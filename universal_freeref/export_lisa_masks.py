from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from training_free_refine.data import ReferringSegDataset, extract_target_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export official LISA SAM-decoder logits on the flat STAMP evaluation JSON."
    )
    parser.add_argument("--lisa-code-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--vision-tower", type=Path, required=True)
    parser.add_argument("--eval-json", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--method", default="LISA-7B-v1")
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    parser.add_argument("--conv-type", choices=("llava_v1", "llava_llama_2"), default="llava_v1")
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--model-max-length", type=int, default=512)
    parser.add_argument("--max-expressions-per-call", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def group_record_positions(dataset: ReferringSegDataset) -> list[list[int]]:
    """Group expressions by image while preserving first-seen image order."""
    groups: OrderedDict[Path, list[int]] = OrderedDict()
    for position, record in enumerate(dataset.records):
        groups.setdefault(record.image_path, []).append(position)
    return list(groups.values())


def chunked(values: list[int], size: int) -> Iterable[list[int]]:
    if size <= 0:
        yield values
        return
    for start in range(0, len(values), size):
        yield values[start : start + size]


def lisa_question(target_text: str) -> str:
    return f"What is {target_text.strip().lower()} in this image? Please output segmentation mask."


def preprocess_sam_image(
    image: np.ndarray,
    transform: Any,
    image_size: int,
) -> tuple[torch.Tensor, tuple[int, int]]:
    resized = transform.apply_image(image)
    resize_shape = tuple(int(value) for value in resized.shape[:2])
    tensor = torch.from_numpy(resized).permute(2, 0, 1).contiguous().float()
    mean = torch.tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    std = torch.tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    tensor = (tensor - mean) / std
    pad_height = image_size - tensor.shape[-2]
    pad_width = image_size - tensor.shape[-1]
    if pad_height < 0 or pad_width < 0:
        raise ValueError(
            f"ResizeLongestSide returned {tuple(tensor.shape[-2:])}, which exceeds {image_size}."
        )
    return F.pad(tensor, (0, pad_width, 0, pad_height)), resize_shape


def _atomic_save_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255, mode="L").save(
        temporary, format="PNG"
    )
    os.replace(temporary, path)


def _atomic_save_logits(path: Path, logits: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, logits=np.asarray(logits, dtype=np.float16))
    os.replace(temporary, path)


def _artifact_paths(output_dir: Path, index: int) -> dict[str, Path]:
    sample_id = f"{index:08d}"
    return {
        "logits": output_dir / "pred_logits" / f"{sample_id}.npz",
        "mask": output_dir / "pred_masks" / f"{sample_id}.png",
        "gt": output_dir / "gt_masks" / f"{sample_id}.png",
    }


def _artifacts_complete(paths: dict[str, Path]) -> bool:
    return all(path.is_file() and path.stat().st_size > 0 for path in paths.values())


def _manifest_row(
    dataset: ReferringSegDataset,
    position: int,
    output_dir: Path,
    method: str,
    split: str,
) -> dict[str, Any]:
    record = dataset.records[position]
    paths = _artifact_paths(output_dir, record.index)
    return {
        "name": f"{record.name}_{record.index}",
        "method": method,
        "split": split,
        "instance_id": str(record.index),
        "image": str(record.image_path),
        "gt_mask": str(paths["gt"]),
        "prediction": str(paths["logits"]),
        "prediction_kind": "logits",
        "array_key": "logits",
        "threshold": 0.5,
        "pred_mask": str(paths["mask"]),
        "query": extract_target_text({}, record.user_text),
        "protocol": "official_teacher_forced_seg_token",
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _dtype(precision: str) -> torch.dtype:
    return {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[precision]


def main() -> int:
    args = parse_args()
    if min(args.limit, args.offset, args.max_expressions_per_call) < 0:
        raise ValueError("limit, offset and max-expressions-per-call must be non-negative.")
    if args.image_size <= 0 or args.model_max_length <= 0:
        raise ValueError("image-size and model-max-length must be positive.")
    if not torch.cuda.is_available():
        raise RuntimeError("LISA export requires a CUDA GPU.")

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

    # LISA uses top-level package names such as `model` and `utils`.
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

    config = LlavaConfig.from_pretrained(model_path, local_files_only=True)
    # The released config stores an online model id. Override both aliases so
    # offline inference always uses the downloaded CLIP directory.
    config.vision_tower = str(vision_tower_path)
    config.mm_vision_tower = str(vision_tower_path)
    model = LISAForCausalLM.from_pretrained(
        model_path,
        config=config,
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
    dataset = ReferringSegDataset(
        args.eval_json,
        data_root=args.data_root,
        limit=args.limit,
        offset=args.offset,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    groups = group_record_positions(dataset)
    model_calls = 0
    exported_samples = 0
    reused_samples = 0
    empty_predictions = 0
    model_seconds = 0.0

    for positions in tqdm(groups, desc="LISA images", dynamic_ncols=True):
        pending = []
        for position in positions:
            record = dataset.records[position]
            paths = _artifact_paths(args.output_dir, record.index)
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
            label_shape = target_tensors[0].shape
            if any(target.shape != label_shape for target in target_tensors):
                raise ValueError("Expressions from one image have different GT mask shapes.")

            torch.cuda.synchronize()
            start = time.perf_counter()
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
            model_seconds += time.perf_counter() - start
            model_calls += 1

            pred_masks = output["pred_masks"]
            if len(pred_masks) != 1 or pred_masks[0].shape[0] != len(samples):
                shapes = [tuple(value.shape) for value in pred_masks]
                raise RuntimeError(
                    f"LISA returned mask shapes {shapes} for {len(samples)} expressions."
                )
            logits_batch = pred_masks[0].detach().float().cpu().numpy()
            for sample, logits in zip(samples, logits_batch):
                paths = _artifact_paths(args.output_dir, sample.record.index)
                hard_mask = logits > 0.0
                _atomic_save_logits(paths["logits"], logits)
                _atomic_save_mask(paths["mask"], hard_mask)
                _atomic_save_mask(paths["gt"], sample.mask.cpu().numpy())
                exported_samples += 1
                empty_predictions += int(not hard_mask.any())

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

    report = {
        "source": "official_lisa_teacher_forced_seg_token",
        "samples": len(rows),
        "model": str(model_path),
        "vision_tower": str(vision_tower_path),
        "eval_json": str(args.eval_json.resolve()),
        "split": args.split,
        "precision": args.precision,
        "prediction_kind": "logits",
        "model_calls": model_calls,
        "exported_samples": exported_samples,
        "reused_samples": reused_samples,
        "empty_predictions_in_new_exports": empty_predictions,
        "model_seconds": model_seconds,
        "model_seconds_per_new_sample": model_seconds / max(exported_samples, 1),
        "manifest": str(manifest_path.resolve()),
    }
    (args.output_dir / "export_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
