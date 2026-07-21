from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from training_free_refine.refiner import (
    TrainingFreeRefineConfig,
    TrainingFreeUncertaintyRefiner,
)

from .eval_lisa_paper_protocol import (
    EXPECTED_EXPRESSIONS,
    PAPER_CIOU,
    _dtype,
    _git_revision,
    _validate_layout,
)
from .export_lisa_freeref_prompt import probability_to_sam_mask_input
from .metrics import boundary_iou, mask_iou, paired_statistics


BRANCHES = ("baseline", "freeref", "baseline_sam", "freeref_sam")
PROTOCOL = "lisa_public_v1_official_refer_freeref_before_second_samh_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate public LISA on the official REFER ValDataset with four paired paths: "
            "LISA, LISA+FreeRef, LISA+second frozen SAM-H, and "
            "LISA+FreeRef+second frozen SAM-H."
        )
    )
    parser.add_argument("--lisa-code-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--vision-tower", type=Path, required=True)
    parser.add_argument("--sam-path", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--val-dataset", default="refcoco|unc|testA")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--paper-row", choices=tuple(PAPER_CIOU), default="finetuned_referseg")
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    parser.add_argument("--conv-type", choices=("llava_v1", "llava_llama_2"), default="llava_v1")
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--model-max-length", type=int, default=512)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit-images", type=int, default=0)
    parser.add_argument("--offset-images", type=int, default=0)
    parser.add_argument("--boundary-tolerance", type=int, default=2)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
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


def _artifact_paths(output_dir: Path, image_index: int, expression_index: int) -> dict[str, Path]:
    stem = f"image_{image_index:06d}_expr_{expression_index:04d}"
    paths = {
        "baseline_logits": output_dir / "baseline_logits" / f"{stem}.npz",
        "freeref_probability": output_dir / "freeref_probabilities" / f"{stem}.npz",
        "baseline_sam_logits": output_dir / "baseline_sam_logits" / f"{stem}.npz",
        "freeref_sam_logits": output_dir / "freeref_sam_logits" / f"{stem}.npz",
        "gt": output_dir / "gt_masks" / f"{stem}.png",
        "metadata": output_dir / "metadata" / f"{stem}.json",
    }
    for branch in BRANCHES:
        paths[f"{branch}_mask"] = output_dir / f"{branch}_masks" / f"{stem}.png"
    return paths


def _complete(paths: dict[str, Path]) -> bool:
    return all(path.is_file() and path.stat().st_size > 0 for path in paths.values())


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=True), encoding="utf-8")
    os.replace(temporary, path)


def _atomic_array(path: Path, key: str, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **{key: np.asarray(value, dtype=np.float16)})
    os.replace(temporary, path)


def _atomic_mask(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    Image.fromarray(np.asarray(value, dtype=np.uint8) * 255, mode="L").save(temporary, format="PNG")
    os.replace(temporary, path)


def _load_array(path: Path, key: str) -> np.ndarray:
    with np.load(path) as value:
        return np.asarray(value[key], dtype=np.float32)


def _load_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8) > 0


class PairedFreeRefSamDecoder:
    """Capture LISA's first SAM mask, then compare paired dense-mask re-decodes."""

    def __init__(
        self,
        visual_model: Any,
        image: np.ndarray,
        resize_shape: tuple[int, int],
        original_size: tuple[int, int],
        refiner: TrainingFreeUncertaintyRefiner,
        mask_transform: Any,
        epsilon: float,
    ) -> None:
        self.visual_model = visual_model
        self.image = image
        self.resize_shape = resize_shape
        self.original_size = original_size
        self.refiner = refiner
        self.mask_transform = mask_transform
        self.epsilon = epsilon
        self.decoder = visual_model.mask_decoder
        self.original_forward = self.decoder.forward
        self.calls = 0
        self.baseline_logits: np.ndarray | None = None
        self.freeref_probability: np.ndarray | None = None
        self.baseline_sam_logits: np.ndarray | None = None
        self.freeref_sam_logits: np.ndarray | None = None
        self.timings = {
            "first_sam_seconds": 0.0,
            "freeref_seconds": 0.0,
            "baseline_second_sam_seconds": 0.0,
            "freeref_second_sam_seconds": 0.0,
        }

    def __enter__(self) -> "PairedFreeRefSamDecoder":
        self.decoder.forward = self._forward
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.decoder.forward = self.original_forward

    def _postprocess(self, low_res_masks: torch.Tensor) -> torch.Tensor:
        return self.visual_model.postprocess_masks(
            low_res_masks,
            input_size=self.resize_shape,
            original_size=self.original_size,
        )[:, 0]

    def _dense_prompt(self, probabilities: np.ndarray, kwargs: dict[str, Any]) -> torch.Tensor:
        mask_input_size = tuple(int(value) for value in self.visual_model.prompt_encoder.mask_input_size)
        mask_inputs = [
            probability_to_sam_mask_input(
                probability,
                self.mask_transform,
                mask_input_size,
                epsilon=self.epsilon,
            )
            for probability in probabilities
        ]
        mask_prompts = torch.from_numpy(np.stack(mask_inputs, axis=0)[:, None]).to(
            device=kwargs["image_embeddings"].device,
            dtype=kwargs["dense_prompt_embeddings"].dtype,
        )
        return self.visual_model.prompt_encoder._embed_masks(mask_prompts)

    def _second_pass(
        self, probabilities: np.ndarray, kwargs: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        second_kwargs = dict(kwargs)
        second_kwargs["dense_prompt_embeddings"] = self._dense_prompt(probabilities, kwargs)
        low_res, quality = self.original_forward(**second_kwargs)
        full = self._postprocess(low_res).detach().float().cpu().numpy()
        return low_res, quality, full

    def _forward(self, *args: Any, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        self.calls += 1
        if self.calls != 1:
            raise RuntimeError(f"Expected one native LISA SAM call, observed {self.calls}.")
        if args:
            raise RuntimeError("LISA mask decoder unexpectedly used positional arguments.")

        torch.cuda.synchronize()
        started = time.perf_counter()
        baseline_low_res, baseline_quality = self.original_forward(**kwargs)
        baseline_full = self._postprocess(baseline_low_res)
        torch.cuda.synchronize()
        self.timings["first_sam_seconds"] = time.perf_counter() - started
        self.baseline_logits = baseline_full.detach().float().cpu().numpy()
        baseline_probability = 1.0 / (1.0 + np.exp(-np.clip(self.baseline_logits, -30.0, 30.0)))

        started = time.perf_counter()
        refined = []
        for probability in baseline_probability:
            output = self.refiner.refine_probability(self.image, probability)
            refined.append(np.asarray(output["refined_probability"], dtype=np.float32))
        self.freeref_probability = np.stack(refined, axis=0)
        self.timings["freeref_seconds"] = time.perf_counter() - started

        torch.cuda.synchronize()
        started = time.perf_counter()
        _, _, self.baseline_sam_logits = self._second_pass(baseline_probability, kwargs)
        torch.cuda.synchronize()
        self.timings["baseline_second_sam_seconds"] = time.perf_counter() - started

        torch.cuda.synchronize()
        started = time.perf_counter()
        freeref_low_res, freeref_quality, self.freeref_sam_logits = self._second_pass(
            self.freeref_probability, kwargs
        )
        torch.cuda.synchronize()
        self.timings["freeref_second_sam_seconds"] = time.perf_counter() - started
        return freeref_low_res, freeref_quality


def _row(
    name: str,
    image_path: Path,
    target: np.ndarray,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
    tolerance: int,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "name": name,
        "image": str(image_path),
        **metadata,
    }
    for branch in BRANCHES:
        mask = np.asarray(arrays[branch], dtype=bool)
        iou, intersection, union = mask_iou(mask, target)
        row[f"{branch}_iou"] = iou
        row[f"{branch}_boundary_iou"] = boundary_iou(mask, target, tolerance)
        row[f"_{branch}_intersection"] = intersection
        row[f"_{branch}_union"] = union
    return row


def summarize_rows(rows: list[dict[str, Any]], bootstrap_samples: int, seed: int) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot summarize an empty LISA result.")
    summary: dict[str, Any] = {"samples": len(rows)}
    for branch in BRANCHES:
        summary[f"{branch}_mean_iou"] = float(np.mean([row[f"{branch}_iou"] for row in rows]))
        intersection = sum(int(row[f"_{branch}_intersection"]) for row in rows)
        union = sum(int(row[f"_{branch}_union"]) for row in rows)
        summary[f"{branch}_cIoU"] = intersection / max(union, 1)
        summary[f"{branch}_boundary_iou"] = float(
            np.mean([row[f"{branch}_boundary_iou"] for row in rows])
        )
    for branch in BRANCHES[1:]:
        deltas = [float(row[f"{branch}_iou"] - row["baseline_iou"]) for row in rows]
        summary[f"{branch}_mean_iou_delta"] = float(np.mean(deltas))
        summary[f"{branch}_cIoU_delta"] = summary[f"{branch}_cIoU"] - summary["baseline_cIoU"]
        summary[f"{branch}_improved_samples"] = sum(delta > 1e-12 for delta in deltas)
        summary[f"{branch}_degraded_samples"] = sum(delta < -1e-12 for delta in deltas)
        if branch == "freeref_sam":
            summary.update(paired_statistics(deltas, bootstrap_samples, seed))
    return summary


def _comparison(summary: dict[str, Any], split: str) -> str:
    labels = {
        "baseline": "LISA",
        "freeref": "LISA + FreeRef",
        "baseline_sam": "LISA + second SAM-H",
        "freeref_sam": "LISA + FreeRef + second SAM-H",
    }
    lines = [
        "# LISA Official REFER + FreeRef Before Second SAM-H",
        "",
        f"Split: `{split}`; public checkpoint paired protocol; N={summary['samples']}.",
        "",
        "| Branch | mIoU | cIoU | Boundary IoU |",
        "|---|---:|---:|---:|",
    ]
    for branch in BRANCHES:
        lines.append(
            f"| {labels[branch]} | {summary[f'{branch}_mean_iou'] * 100:.2f} | "
            f"{summary[f'{branch}_cIoU'] * 100:.2f} | "
            f"{summary[f'{branch}_boundary_iou'] * 100:.2f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    if min(args.workers, args.limit_images, args.offset_images, args.boundary_tolerance) < 0:
        raise ValueError("workers, limits, offsets, and tolerance must be non-negative.")
    if not torch.cuda.is_available():
        raise RuntimeError("LISA paired SAM evaluation requires a CUDA GPU.")
    dataset_name, split_by, split = _validate_layout(args)

    code_dir = args.lisa_code_dir.expanduser().resolve()
    model_path = args.model_path.expanduser().resolve()
    vision_tower_path = args.vision_tower.expanduser().resolve()
    sam_path = args.sam_path.expanduser().resolve()
    dataset_dir = args.dataset_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(code_dir))
    from transformers import AutoTokenizer  # type: ignore

    from model.LISA import LISAForCausalLM  # type: ignore[import-not-found]
    from model.llava import conversation as conversation_lib  # type: ignore[import-not-found]
    from model.llava.model.language_model.llava_llama import LlavaConfig  # type: ignore[import-not-found]
    from model.segment_anything.utils.transforms import ResizeLongestSide  # type: ignore[import-not-found]
    from utils.dataset import ValDataset, collate_fn  # type: ignore[import-not-found]
    from utils.utils import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN  # type: ignore[import-not-found]

    torch.set_grad_enabled(False)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    dtype = _dtype(args.precision)
    device = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
        local_files_only=True,
    )
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.add_tokens("[SEG]")
    seg_token_ids = tokenizer("[SEG]", add_special_tokens=False).input_ids
    if len(seg_token_ids) != 1:
        raise RuntimeError(f"Expected [SEG] to map to one token, got {seg_token_ids}.")
    tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

    config = LlavaConfig.from_pretrained(model_path, local_files_only=True)
    config.vision_tower = str(vision_tower_path)
    config.mm_vision_tower = str(vision_tower_path)
    model, loading_info = LISAForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        train_mask_decoder=True,
        out_dim=256,
        seg_token_idx=seg_token_ids[0],
        vision_pretrained=str(sam_path),
        vision_tower=str(vision_tower_path),
        use_mm_start_end=True,
        local_files_only=True,
        output_loading_info=True,
    )
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.resize_token_embeddings(len(tokenizer))
    model.get_model().initialize_vision_modules(model.get_model().config)
    model.to(device=device, dtype=dtype)
    model.get_model().get_vision_tower().to(device=device, dtype=dtype)
    model.eval()

    conversation_lib.default_conversation = conversation_lib.conv_templates[args.conv_type]
    dataset = ValDataset(
        str(dataset_dir), tokenizer, str(vision_tower_path), args.val_dataset, args.image_size
    )
    start_index = min(args.offset_images, len(dataset))
    end_index = len(dataset) if not args.limit_images else min(start_index + args.limit_images, len(dataset))
    image_indices = list(range(start_index, end_index))
    loader = DataLoader(
        Subset(dataset, image_indices),
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False,
        collate_fn=partial(
            collate_fn,
            tokenizer=tokenizer,
            conv_type=args.conv_type,
            use_mm_start_end=True,
            local_rank=0,
        ),
    )
    refiner_config = _config(args)
    refiner = TrainingFreeUncertaintyRefiner(refiner_config)
    mask_transform = ResizeLongestSide(
        max(int(value) for value in model.model.visual_model.prompt_encoder.mask_input_size)
    )

    rows: list[dict[str, Any]] = []
    model_calls = 0
    reused_expressions = 0
    exported_expressions = 0
    total_seconds = 0.0
    timing_totals = {
        "first_sam_seconds": 0.0,
        "freeref_seconds": 0.0,
        "baseline_second_sam_seconds": 0.0,
        "freeref_second_sam_seconds": 0.0,
    }

    for image_index, batch in tqdm(
        zip(image_indices, loader),
        total=len(image_indices),
        desc=f"LISA paired official {dataset_name} {split}",
        dynamic_ncols=True,
    ):
        image_path = Path(batch["image_paths"][0]).resolve()
        expression_count = len(batch["conversation_list"])
        paths_list = [
            _artifact_paths(output_dir, image_index, expression_index)
            for expression_index in range(expression_count)
        ]
        can_reuse = not args.overwrite and all(_complete(paths) for paths in paths_list)
        if can_reuse:
            reused_expressions += expression_count
        else:
            for key, value in list(batch.items()):
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(device, non_blocking=True)
                elif value and isinstance(value, list) and isinstance(value[0], torch.Tensor):
                    batch[key] = [item.to(device, non_blocking=True) for item in value]
            batch["images"] = batch["images"].to(dtype=dtype)
            batch["images_clip"] = batch["images_clip"].to(dtype=dtype)
            target_group = batch["masks_list"][0]
            original_size = tuple(int(value) for value in target_group.shape[-2:])
            resize_shape = tuple(int(value) for value in batch["resize_list"][0])
            image = np.asarray(Image.open(image_path).convert("RGB"))
            adapter = PairedFreeRefSamDecoder(
                model.model.visual_model,
                image,
                resize_shape,
                original_size,
                refiner,
                mask_transform,
                args.mask_logit_epsilon,
            )
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.inference_mode(), adapter:
                output = model(**batch)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            total_seconds += elapsed
            model_calls += 1
            if any(
                value is None
                for value in (
                    adapter.baseline_logits,
                    adapter.freeref_probability,
                    adapter.baseline_sam_logits,
                    adapter.freeref_sam_logits,
                )
            ):
                raise RuntimeError("LISA paired decoder did not capture all four branches.")
            target_batch = output["gt_masks"][0].detach().cpu().numpy() > 0
            branch_arrays = {
                "baseline": np.asarray(adapter.baseline_logits),
                "freeref": np.asarray(adapter.freeref_probability),
                "baseline_sam": np.asarray(adapter.baseline_sam_logits),
                "freeref_sam": np.asarray(adapter.freeref_sam_logits),
            }
            if any(len(value) != expression_count for value in branch_arrays.values()):
                raise RuntimeError("LISA branch count does not match the expression count.")
            share = 1.0 / max(expression_count, 1)
            per_expression_timing = {
                key: value * share for key, value in adapter.timings.items()
            }
            per_expression_timing["total_seconds"] = elapsed * share
            for key, value in adapter.timings.items():
                timing_totals[key] += value
            for expression_index, paths in enumerate(paths_list):
                baseline_logits = branch_arrays["baseline"][expression_index]
                freeref_probability = branch_arrays["freeref"][expression_index]
                baseline_sam_logits = branch_arrays["baseline_sam"][expression_index]
                freeref_sam_logits = branch_arrays["freeref_sam"][expression_index]
                target = target_batch[expression_index]
                masks = {
                    "baseline": baseline_logits > 0.0,
                    "freeref": freeref_probability >= refiner_config.threshold,
                    "baseline_sam": baseline_sam_logits > 0.0,
                    "freeref_sam": freeref_sam_logits > 0.0,
                }
                _atomic_array(paths["baseline_logits"], "logits", baseline_logits)
                _atomic_array(paths["freeref_probability"], "probability", freeref_probability)
                _atomic_array(paths["baseline_sam_logits"], "logits", baseline_sam_logits)
                _atomic_array(paths["freeref_sam_logits"], "logits", freeref_sam_logits)
                for branch, mask in masks.items():
                    _atomic_mask(paths[f"{branch}_mask"], mask)
                _atomic_mask(paths["gt"], target)
                _atomic_json(paths["metadata"], per_expression_timing)
                exported_expressions += 1

        for expression_index, paths in enumerate(paths_list):
            target = _load_mask(paths["gt"])
            arrays = {branch: _load_mask(paths[f"{branch}_mask"]) for branch in BRANCHES}
            metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
            rows.append(
                _row(
                    f"image_{image_index:06d}_expr_{expression_index:04d}",
                    image_path,
                    target,
                    arrays,
                    metadata,
                    args.boundary_tolerance,
                )
            )

    summary = summarize_rows(rows, args.bootstrap_samples, args.seed)
    full_run = args.limit_images == 0 and args.offset_images == 0
    expected_samples = EXPECTED_EXPRESSIONS[args.val_dataset]
    paper_ciou = PAPER_CIOU[args.paper_row][args.val_dataset]
    baseline_gap = summary["baseline_cIoU"] * 100.0 - paper_ciou
    summary.update(
        {
            "source": "lisa_official_refer_paired_freeref_sam",
            "protocol": PROTOCOL,
            "checkpoint_scope": "public_checkpoint_paired_transfer_not_table3_reproduction",
            "paper_row": args.paper_row,
            "val_dataset": args.val_dataset,
            "dataset": dataset_name,
            "split_by": split_by,
            "split": split,
            "images": len(image_indices),
            "expected_full_samples": expected_samples,
            "sample_count_matches": summary["samples"] == expected_samples if full_run else None,
            "paper_cIoU_percent": paper_ciou,
            "baseline_cIoU_gap_points": baseline_gap,
            "paper_match": (
                summary["samples"] == expected_samples and abs(baseline_gap) <= 0.5
                if full_run
                else None
            ),
            "model": str(model_path),
            "vision_tower": str(vision_tower_path),
            "sam_path": str(sam_path),
            "dataset_dir": str(dataset_dir),
            "lisa_code_revision": _git_revision(code_dir),
            "precision": args.precision,
            "model_calls": model_calls,
            "exported_expressions": exported_expressions,
            "reused_expressions": reused_expressions,
            "seconds_per_sample": float(np.mean([row["total_seconds"] for row in rows])),
            "timing_means": {
                key: float(np.mean([row[key] for row in rows]))
                for key in (
                    "first_sam_seconds",
                    "freeref_seconds",
                    "baseline_second_sam_seconds",
                    "freeref_second_sam_seconds",
                )
            },
            "timing_totals_current_process": timing_totals,
            "config": asdict(refiner_config),
            "loading_info": {
                key: [str(item) for item in loading_info.get(key, [])]
                for key in ("missing_keys", "unexpected_keys", "mismatched_keys")
            },
        }
    )

    rows_path = output_dir / "eval_rows.csv"
    public_rows = [{key: value for key, value in row.items() if not key.startswith("_")} for row in rows]
    with rows_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(public_rows[0]))
        writer.writeheader()
        writer.writerows(public_rows)
    summary["rows_csv"] = str(rows_path)
    _atomic_json(output_dir / "eval_summary.json", summary)
    comparison = _comparison(summary, args.val_dataset)
    (output_dir / "comparison.md").write_text(comparison, encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    print(comparison)
    if full_run and summary["samples"] != expected_samples:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
