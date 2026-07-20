from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from training_free_refine.data import ReferringSegDataset, extract_target_text
from training_free_refine.refiner import (
    TrainingFreeRefineConfig,
    TrainingFreeUncertaintyRefiner,
)

from .export_lisa_masks import (
    _atomic_save_logits,
    _atomic_save_mask,
    _dtype,
    chunked,
    group_record_positions,
    lisa_question,
    preprocess_sam_image,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate LISA with FreeRef inserted before its final SAM decoding pass. "
            "The first SAM pass supplies a coarse spatial prior; FreeRef converts it "
            "into a dense mask prompt for a second pass through the same decoder."
        )
    )
    parser.add_argument("--lisa-code-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--vision-tower", type=Path, required=True)
    parser.add_argument("--eval-json", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--method", default="LISA-7B-v1 + FreeRef-Prompt")
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    parser.add_argument("--conv-type", choices=("llava_v1", "llava_llama_2"), default="llava_v1")
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--model-max-length", type=int, default=512)
    parser.add_argument("--max-expressions-per-call", type=int, default=1)
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--n-segments", type=int, default=1024)
    parser.add_argument("--compactness", type=float, default=10.0)
    parser.add_argument("--slic-sigma", type=float, default=1.0)
    parser.add_argument("--graph-lambda", type=float, default=1.0)
    parser.add_argument("--confidence-power", type=float, default=2.0)
    parser.add_argument("--fusion-power", type=float, default=1.0)
    parser.add_argument("--foreground-seed", type=float, default=0.9)
    parser.add_argument("--background-seed", type=float, default=0.1)
    parser.add_argument("--seed-strength", type=float, default=50.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--mask-logit-epsilon", type=float, default=1e-4)
    return parser.parse_args()


def probability_to_sam_mask_input(
    probability: np.ndarray,
    transform: Any,
    mask_input_size: tuple[int, int],
    epsilon: float = 1e-4,
) -> np.ndarray:
    """Map an original-resolution probability to SAM's padded low-res logit frame."""
    probability = np.asarray(probability, dtype=np.float32)
    if probability.ndim != 2:
        raise ValueError(f"probability must have shape [H, W], got {probability.shape}")
    if not np.isfinite(probability).all():
        raise ValueError("probability contains non-finite values.")
    if not 0 < epsilon < 0.5:
        raise ValueError("epsilon must lie in (0, 0.5).")
    probability = np.clip(probability, epsilon, 1.0 - epsilon)
    logits = np.log(probability) - np.log1p(-probability)
    resized = np.asarray(transform.apply_image(logits[..., None]), dtype=np.float32)
    if resized.ndim == 3:
        resized = resized[..., 0]
    target_h, target_w = (int(value) for value in mask_input_size)
    if resized.shape[0] > target_h or resized.shape[1] > target_w:
        raise ValueError(
            f"Resized mask {resized.shape} exceeds SAM mask input size {mask_input_size}."
        )
    padded = np.zeros((target_h, target_w), dtype=np.float32)
    padded[: resized.shape[0], : resized.shape[1]] = resized
    return padded


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    os.replace(temporary, path)


def _atomic_save_array(path: Path, key: str, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **{key: np.asarray(value, dtype=np.float16)})
    os.replace(temporary, path)


def _artifact_paths(output_dir: Path, index: int) -> dict[str, Path]:
    sample_id = f"{index:08d}"
    return {
        "baseline_logits": output_dir / "baseline_logits" / f"{sample_id}.npz",
        "prompted_logits": output_dir / "prompted_logits" / f"{sample_id}.npz",
        "freeref_prior": output_dir / "freeref_priors" / f"{sample_id}.npz",
        "baseline_mask": output_dir / "baseline_masks" / f"{sample_id}.png",
        "prompted_mask": output_dir / "prompted_masks" / f"{sample_id}.png",
        "gt": output_dir / "gt_masks" / f"{sample_id}.png",
        "metadata": output_dir / "metadata" / f"{sample_id}.json",
    }


def _artifacts_complete(paths: dict[str, Path]) -> bool:
    return all(path.is_file() and path.stat().st_size > 0 for path in paths.values())


class FreeRefPromptDecoder:
    """Temporarily turn one LISA SAM decode into coarse decode + FreeRef re-decode."""

    def __init__(
        self,
        visual_model: Any,
        image: np.ndarray,
        resize_shape: tuple[int, int],
        original_size: tuple[int, int],
        refiner: TrainingFreeUncertaintyRefiner,
        mask_transform: Any,
        epsilon: float,
        call_start: float,
    ) -> None:
        self.visual_model = visual_model
        self.image = image
        self.resize_shape = resize_shape
        self.original_size = original_size
        self.refiner = refiner
        self.mask_transform = mask_transform
        self.epsilon = epsilon
        self.call_start = call_start
        self._decoder = visual_model.mask_decoder
        self._original_forward = self._decoder.forward
        self.calls = 0
        self.baseline_logits: np.ndarray | None = None
        self.freeref_probability: np.ndarray | None = None
        self.base_seconds = 0.0
        self.freeref_seconds = 0.0
        self.second_decoder_seconds = 0.0

    def __enter__(self) -> "FreeRefPromptDecoder":
        self._decoder.forward = self._forward
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self._decoder.forward = self._original_forward

    def _forward(self, *args: Any, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        self.calls += 1
        if self.calls != 1:
            raise RuntimeError(
                "FreeRef-Prompt expects one SAM mask-decoder call per LISA image group. "
                f"Observed call {self.calls}. Use one image group per model invocation."
            )
        if args:
            raise RuntimeError("LISA mask decoder unexpectedly used positional arguments.")

        low_res_masks, _ = self._original_forward(**kwargs)
        baseline_full = self.visual_model.postprocess_masks(
            low_res_masks,
            input_size=self.resize_shape,
            original_size=self.original_size,
        )[:, 0]
        torch.cuda.synchronize()
        self.base_seconds = time.perf_counter() - self.call_start
        self.baseline_logits = baseline_full.detach().float().cpu().numpy()

        start = time.perf_counter()
        refined_probabilities = []
        mask_inputs = []
        mask_input_size = tuple(
            int(value) for value in self.visual_model.prompt_encoder.mask_input_size
        )
        for logits in self.baseline_logits:
            probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
            output = self.refiner.refine_probability(self.image, probability)
            refined = np.asarray(output["refined_probability"], dtype=np.float32)
            refined_probabilities.append(refined)
            mask_inputs.append(
                probability_to_sam_mask_input(
                    refined,
                    self.mask_transform,
                    mask_input_size,
                    epsilon=self.epsilon,
                )
            )
        self.freeref_probability = np.stack(refined_probabilities, axis=0)
        mask_prompts = torch.from_numpy(np.stack(mask_inputs, axis=0)[:, None]).to(
            device=kwargs["image_embeddings"].device,
            dtype=kwargs["dense_prompt_embeddings"].dtype,
        )
        self.freeref_seconds = time.perf_counter() - start

        start = time.perf_counter()
        second_kwargs = dict(kwargs)
        second_kwargs["dense_prompt_embeddings"] = self.visual_model.prompt_encoder._embed_masks(
            mask_prompts
        )
        second_masks, second_iou = self._original_forward(**second_kwargs)
        torch.cuda.synchronize()
        self.second_decoder_seconds = time.perf_counter() - start
        return second_masks, second_iou


def _config(args: argparse.Namespace) -> TrainingFreeRefineConfig:
    return TrainingFreeRefineConfig(
        n_segments=args.n_segments,
        compactness=args.compactness,
        slic_sigma=args.slic_sigma,
        graph_lambda=args.graph_lambda,
        confidence_power=args.confidence_power,
        fusion_power=args.fusion_power,
        foreground_seed=args.foreground_seed,
        background_seed=args.background_seed,
        seed_strength=args.seed_strength,
        threshold=args.threshold,
    )


def _manifest_row(
    dataset: ReferringSegDataset,
    position: int,
    output_dir: Path,
    method: str,
    split: str,
) -> dict[str, Any]:
    record = dataset.records[position]
    paths = _artifact_paths(output_dir, record.index)
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    return {
        "name": f"{record.name}_{record.index}",
        "method": method,
        "split": split,
        "instance_id": str(record.index),
        "image": str(record.image_path),
        "gt_mask": str(paths["gt"]),
        "baseline_prediction": str(paths["baseline_logits"]),
        "prompted_prediction": str(paths["prompted_logits"]),
        "freeref_prior": str(paths["freeref_prior"]),
        "array_key": "logits",
        "threshold": 0.5,
        "query": extract_target_text({}, record.user_text),
        "protocol": "official_teacher_forced_seg_token_freeref_prompt_redecode",
        **metadata,
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    args = parse_args()
    if min(args.limit, args.offset, args.max_expressions_per_call) < 0:
        raise ValueError("limit, offset and max-expressions-per-call must be non-negative.")
    if args.limit == 0:
        print("WARNING: --limit 0 requests a full-split run.")
    if args.image_size <= 0 or args.model_max_length <= 0:
        raise ValueError("image-size and model-max-length must be positive.")
    if not torch.cuda.is_available():
        raise RuntimeError("LISA FreeRef-Prompt evaluation requires a CUDA GPU.")

    code_dir = args.lisa_code_dir.expanduser().resolve()
    model_path = args.model_path.expanduser().resolve()
    vision_tower_path = args.vision_tower.expanduser().resolve()
    for label, path, required in (
        ("LISA code", code_dir, code_dir / "model" / "LISA.py"),
        ("LISA model", model_path, model_path / "config.json"),
        ("CLIP vision tower", vision_tower_path, vision_tower_path / "config.json"),
    ):
        if not required.is_file():
            raise FileNotFoundError(f"{label} is incomplete below {path}; missing {required.name}.")

    sys.path.insert(0, str(code_dir))
    from transformers import AutoTokenizer, CLIPImageProcessor  # type: ignore

    from model.LISA import LISAForCausalLM  # type: ignore[import-not-found]
    from model.llava import conversation as conversation_lib  # type: ignore[import-not-found]
    from model.llava.mm_utils import tokenizer_image_token  # type: ignore[import-not-found]
    from model.llava.model.language_model.llava_llama import (  # type: ignore[import-not-found]
        LlavaConfig,
    )
    from model.segment_anything.utils.transforms import (  # type: ignore[import-not-found]
        ResizeLongestSide,
    )
    from utils.utils import (  # type: ignore[import-not-found]
        DEFAULT_IM_END_TOKEN,
        DEFAULT_IM_START_TOKEN,
        DEFAULT_IMAGE_TOKEN,
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_grad_enabled(False)
    device = torch.device("cuda:0")
    dtype = _dtype(args.precision)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
        local_files_only=True,
    )
    tokenizer.pad_token = tokenizer.unk_token
    seg_token_ids = tokenizer("[SEG]", add_special_tokens=False).input_ids
    if len(seg_token_ids) != 1:
        raise RuntimeError(f"Expected [SEG] to map to one token, got {seg_token_ids}.")

    model_config = LlavaConfig.from_pretrained(model_path, local_files_only=True)
    model_config.vision_tower = str(vision_tower_path)
    model_config.mm_vision_tower = str(vision_tower_path)
    model = LISAForCausalLM.from_pretrained(
        model_path,
        config=model_config,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        vision_tower=str(vision_tower_path),
        seg_token_idx=seg_token_ids[0],
        local_files_only=True,
    )
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.get_model().initialize_vision_modules(model.get_model().config)
    model.to(device=device, dtype=dtype)
    model.get_model().get_vision_tower().to(device=device, dtype=dtype)
    model.eval()

    conversation_lib.default_conversation = conversation_lib.conv_templates[args.conv_type]
    clip_processor = CLIPImageProcessor.from_pretrained(
        vision_tower_path, local_files_only=True
    )
    sam_transform = ResizeLongestSide(args.image_size)
    prompt_transform = ResizeLongestSide(
        max(int(value) for value in model.model.visual_model.prompt_encoder.mask_input_size)
    )
    dataset = ReferringSegDataset(
        args.eval_json,
        data_root=args.data_root,
        limit=args.limit,
        offset=args.offset,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = _config(args)
    refiner = TrainingFreeUncertaintyRefiner(config)

    model_calls = 0
    exported_samples = 0
    reused_samples = 0
    empty_baseline = 0
    empty_prompted = 0
    for positions in tqdm(group_record_positions(dataset), desc="LISA FreeRef-Prompt", dynamic_ncols=True):
        pending = []
        for position in positions:
            paths = _artifact_paths(args.output_dir, dataset.records[position].index)
            if not args.overwrite and _artifacts_complete(paths):
                reused_samples += 1
            else:
                pending.append(position)
        if not pending:
            continue

        for call_positions in chunked(pending, args.max_expressions_per_call):
            samples = [dataset[position] for position in call_positions]
            image_np = np.asarray(samples[0].image, dtype=np.uint8)
            if any(np.asarray(sample.image).shape != image_np.shape for sample in samples):
                raise ValueError("Expressions grouped under one image path produced different image shapes.")

            prompts = []
            for sample in samples:
                target_text = extract_target_text({}, sample.record.user_text)
                prompt = DEFAULT_IMAGE_TOKEN + "\n " + lisa_question(target_text)
                prompt = prompt.replace(
                    DEFAULT_IMAGE_TOKEN,
                    DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN,
                )
                conversation = conversation_lib.default_conversation.copy()
                conversation.messages = []
                conversation.append_message(conversation.roles[0], prompt)
                conversation.append_message(conversation.roles[1], "[SEG].")
                prompts.append(conversation.get_prompt())

            input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in prompts]
            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids,
                batch_first=True,
                padding_value=tokenizer.pad_token_id,
            ).to(device)
            attention_masks = input_ids.ne(tokenizer.pad_token_id)
            image_clip = clip_processor.preprocess(image_np, return_tensors="pt")[
                "pixel_values"
            ].to(device=device, dtype=dtype)
            image_sam, resize_shape = preprocess_sam_image(
                image_np, sam_transform, args.image_size
            )
            image_sam = image_sam.unsqueeze(0).to(device=device, dtype=dtype)
            target_tensors = [sample.mask.to(device=device, dtype=torch.float32) for sample in samples]
            label_shape = tuple(int(value) for value in target_tensors[0].shape)
            if any(tuple(target.shape) != label_shape for target in target_tensors):
                raise ValueError("Expressions from one image have different GT mask shapes.")

            torch.cuda.synchronize()
            call_start = time.perf_counter()
            adapter = FreeRefPromptDecoder(
                model.model.visual_model,
                image_np,
                resize_shape,
                label_shape,
                refiner,
                prompt_transform,
                args.mask_logit_epsilon,
                call_start,
            )
            with adapter:
                output = model(
                    images=image_sam,
                    images_clip=image_clip,
                    input_ids=input_ids,
                    labels=None,
                    attention_masks=attention_masks,
                    offset=torch.tensor([0, len(samples)], dtype=torch.long, device=device),
                    masks_list=[torch.stack(target_tensors)],
                    label_list=[torch.full(label_shape, 255.0, device=device)],
                    resize_list=[resize_shape],
                    inference=True,
                )
            torch.cuda.synchronize()
            total_seconds = time.perf_counter() - call_start
            model_calls += 1

            if adapter.calls != 1 or adapter.baseline_logits is None or adapter.freeref_probability is None:
                raise RuntimeError("FreeRef-Prompt did not capture exactly one LISA decoding pass.")
            pred_masks = output["pred_masks"]
            if len(pred_masks) != 1 or pred_masks[0].shape[0] != len(samples):
                shapes = [tuple(value.shape) for value in pred_masks]
                raise RuntimeError(
                    f"LISA returned mask shapes {shapes} for {len(samples)} expressions."
                )
            prompted_logits = pred_masks[0].detach().float().cpu().numpy()
            share = 1.0 / len(samples)
            residual_seconds = max(
                total_seconds
                - adapter.base_seconds
                - adapter.freeref_seconds
                - adapter.second_decoder_seconds,
                0.0,
            )
            timing = {
                "base_seconds": adapter.base_seconds * share,
                "freeref_seconds": adapter.freeref_seconds * share,
                "second_decoder_seconds": adapter.second_decoder_seconds * share,
                "other_seconds": residual_seconds * share,
                "total_seconds": total_seconds * share,
            }

            for sample, baseline, prior, prompted in zip(
                samples,
                adapter.baseline_logits,
                adapter.freeref_probability,
                prompted_logits,
            ):
                paths = _artifact_paths(args.output_dir, sample.record.index)
                baseline_mask = baseline > 0.0
                prompted_mask = prompted > 0.0
                _atomic_save_logits(paths["baseline_logits"], baseline)
                _atomic_save_logits(paths["prompted_logits"], prompted)
                _atomic_save_array(paths["freeref_prior"], "probability", prior)
                _atomic_save_mask(paths["baseline_mask"], baseline_mask)
                _atomic_save_mask(paths["prompted_mask"], prompted_mask)
                _atomic_save_mask(paths["gt"], sample.mask.cpu().numpy())
                _atomic_write_json(paths["metadata"], timing)
                exported_samples += 1
                empty_baseline += int(not baseline_mask.any())
                empty_prompted += int(not prompted_mask.any())

    rows = [
        _manifest_row(dataset, position, args.output_dir, args.method, args.split)
        for position in range(len(dataset))
    ]
    missing = [
        row["name"]
        for position, row in enumerate(rows)
        if not _artifacts_complete(
            _artifact_paths(args.output_dir, dataset.records[position].index)
        )
    ]
    if missing:
        raise RuntimeError(f"Export finished with {len(missing)} incomplete samples: {missing[:5]}")
    manifest_path = args.output_dir / "manifest.jsonl"
    _write_jsonl(manifest_path, rows)

    def mean_timing(name: str) -> float:
        values = [float(row[name]) for row in rows]
        return float(np.mean(values)) if values else math.nan

    report = {
        "source": "lisa_freeref_prompt_redecode",
        "samples": len(rows),
        "model": str(model_path),
        "vision_tower": str(vision_tower_path),
        "eval_json": str(args.eval_json.resolve()),
        "split": args.split,
        "precision": args.precision,
        "model_calls_in_this_run": model_calls,
        "exported_samples": exported_samples,
        "reused_samples": reused_samples,
        "empty_baseline_predictions_in_new_exports": empty_baseline,
        "empty_prompted_predictions_in_new_exports": empty_prompted,
        "base_seconds_per_sample": mean_timing("base_seconds"),
        "freeref_seconds_per_sample": mean_timing("freeref_seconds"),
        "second_decoder_seconds_per_sample": mean_timing("second_decoder_seconds"),
        "total_seconds_per_sample": mean_timing("total_seconds"),
        "relative_overhead_vs_base": (
            mean_timing("total_seconds") / max(mean_timing("base_seconds"), 1e-12) - 1.0
        ),
        "config": asdict(config),
        "manifest": str(manifest_path.resolve()),
    }
    _atomic_write_json(args.output_dir / "export_summary.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
