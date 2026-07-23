from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from .eval_stamp_dumps import boundary_iou, mask_iou, resolve_asset
from .eval_text4seg_outputs import evaluation_items, resize_float, resize_mask
from .postprocess_baselines import (
    PostprocessBaselineConfig,
    densecrf_probability,
    fast_bilateral_solver_probability,
    guided_filter_probability,
    hard_mask_probability,
    slic_region_average_probability,
)
from .refiner import TrainingFreeRefineConfig, TrainingFreeUncertaintyRefiner, stamp_probability


METHODS = (
    "base",
    "densecrf",
    "guided_filter",
    "fast_bilateral_solver",
    "slic_average",
    "freeref",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare training-free post-processors on paired saved predictions."
    )
    parser.add_argument("--source", choices=("stamp", "text4seg"), required=True)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--split-name", default="refcoco_testA")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--pattern", default="*.pt")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--record-timing", action="store_true")
    parser.add_argument("--freeref-backend", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--hard-mask-confidence", type=float, default=0.95)
    parser.add_argument("--boundary-tolerance", type=int, default=2)
    parser.add_argument("--boundary-sigma", type=float, default=8.0)
    parser.add_argument("--n-segments", type=int, default=1024)
    parser.add_argument("--compactness", type=float, default=10.0)
    parser.add_argument("--slic-sigma", type=float, default=1.0)
    parser.add_argument("--densecrf-iterations", type=int, default=5)
    parser.add_argument("--guided-radius", type=int, default=8)
    parser.add_argument("--guided-epsilon", type=float, default=1e-3)
    parser.add_argument("--fbs-sigma-spatial", type=float, default=16.0)
    parser.add_argument("--fbs-sigma-luma", type=float, default=8.0)
    parser.add_argument("--fbs-sigma-chroma", type=float, default=8.0)
    parser.add_argument("--fbs-lambda", type=float, default=128.0)
    parser.add_argument("--fbs-iterations", type=int, default=25)
    parser.add_argument("--fbs-max-tolerance", type=float, default=1e-5)
    return parser.parse_args()


def _load_stamp_items(args: argparse.Namespace) -> list[dict[str, object]]:
    if args.input_dir is None or args.manifest is not None:
        raise ValueError("STAMP requires --input-dir and does not accept --manifest.")
    paths = sorted(args.input_dir.glob(args.pattern))
    return [{"dump": path, "name": path.stem} for path in paths[: args.limit]]


def _load_text4seg_items(args: argparse.Namespace) -> list[dict[str, object]]:
    if args.manifest is None or args.input_dir is not None:
        raise ValueError("Text4Seg requires --manifest and does not accept --input-dir.")
    items = evaluation_items(args)
    return items[: args.limit]


def _load_stamp_sample(item: dict[str, object]) -> dict[str, object]:
    dump_path = Path(item["dump"])
    payload = torch.load(dump_path, map_location="cpu", weights_only=False)
    image_path = resolve_asset(payload["image_path"], dump_path)
    target_path = resolve_asset(payload["mask_path"], dump_path)
    image = np.asarray(Image.open(image_path).convert("RGB"))
    target = np.asarray(Image.open(target_path).convert("L")) > 0
    ignore = np.zeros(target.shape, dtype=bool)
    if payload.get("ignore_path"):
        ignore_path = resolve_asset(payload["ignore_path"], dump_path)
        ignore = np.asarray(Image.open(ignore_path).convert("L")) > 0
        ignore = resize_mask(ignore, target.shape)
        target &= ~ignore
    if image.shape[:2] != target.shape:
        image = np.asarray(
            Image.fromarray(image).resize(
                (target.shape[1], target.shape[0]), Image.Resampling.BILINEAR
            )
        )
    probability = stamp_probability(
        payload["mask_logits"], tuple(payload["grid_hw"]), image.shape[:2]
    ).numpy()
    return {
        "name": str(payload.get("name", item["name"])),
        "image": image,
        "target": target,
        "ignore": ignore,
        "probability": probability,
        "hard_input": False,
    }


def _load_text4seg_sample(item: dict[str, object], confidence: float) -> dict[str, object]:
    image = np.asarray(Image.open(Path(item["image"])).convert("RGB"))
    target = np.asarray(Image.open(Path(item["gt"])).convert("L")) > 0
    coarse = resize_mask(
        np.asarray(Image.open(Path(item["pred"])).convert("L")) > 0,
        image.shape[:2],
    )
    return {
        "name": str(item["name"]),
        "image": image,
        "target": target,
        "ignore": np.zeros(target.shape, dtype=bool),
        "probability": hard_mask_probability(coarse, confidence),
        "hard_mask": coarse,
        "hard_input": True,
    }


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def main() -> int:
    args = parse_args()
    if args.limit <= 0:
        raise ValueError("limit must be positive for the paired comparison.")
    if args.boundary_tolerance < 0 or args.boundary_sigma <= 0:
        raise ValueError("Invalid boundary configuration.")
    items = _load_stamp_items(args) if args.source == "stamp" else _load_text4seg_items(args)
    if len(items) != args.limit:
        raise ValueError(f"Expected {args.limit} samples, found {len(items)}.")

    baseline_config = PostprocessBaselineConfig(
        threshold=args.threshold,
        hard_mask_confidence=args.hard_mask_confidence,
        densecrf_iterations=args.densecrf_iterations,
        guided_radius=args.guided_radius,
        guided_epsilon=args.guided_epsilon,
        fbs_sigma_spatial=args.fbs_sigma_spatial,
        fbs_sigma_luma=args.fbs_sigma_luma,
        fbs_sigma_chroma=args.fbs_sigma_chroma,
        fbs_lambda=args.fbs_lambda,
        fbs_iterations=args.fbs_iterations,
        fbs_max_tolerance=args.fbs_max_tolerance,
        n_segments=args.n_segments,
        compactness=args.compactness,
        slic_sigma=args.slic_sigma,
    )
    freeref_config = TrainingFreeRefineConfig(
        threshold=args.threshold,
        n_segments=args.n_segments,
        compactness=args.compactness,
        slic_sigma=args.slic_sigma,
    )
    if args.freeref_backend == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError("GPU FreeRef requested but CUDA is unavailable.")
        from .gpu_refiner import GpuTrainingFreeUncertaintyRefiner

        freeref = GpuTrainingFreeUncertaintyRefiner(freeref_config)
    else:
        freeref = TrainingFreeUncertaintyRefiner(freeref_config)

    def apply_method(method: str, sample: dict[str, object]) -> np.ndarray:
        image = np.asarray(sample["image"])
        probability = np.asarray(sample["probability"], dtype=np.float32)
        if method == "base":
            return probability
        if method == "densecrf":
            return densecrf_probability(image, probability, baseline_config)
        if method == "guided_filter":
            return guided_filter_probability(image, probability, baseline_config)
        if method == "fast_bilateral_solver":
            return fast_bilateral_solver_probability(image, probability, baseline_config)
        if method == "slic_average":
            return slic_region_average_probability(image, probability, baseline_config)
        if method == "freeref":
            if bool(sample["hard_input"]):
                output = freeref.refine_hard_mask(
                    image, np.asarray(sample["hard_mask"]), boundary_sigma=args.boundary_sigma
                )
            else:
                output = freeref.refine_probability(image, probability)
            tensor = output["refined_probability"]
            return tensor.detach().float().cpu().numpy()
        raise ValueError(f"Unsupported method: {method}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    totals = {
        method: {"intersection": 0, "union": 0, "iou": 0.0, "boundary": 0.0, "times": []}
        for method in args.methods
    }
    loader: Callable[[dict[str, object]], dict[str, object]]
    if args.source == "stamp":
        loader = _load_stamp_sample
    else:
        loader = lambda item: _load_text4seg_sample(item, args.hard_mask_confidence)

    for item in tqdm(items, desc=f"{args.model_label} post-process baselines", dynamic_ncols=True):
        sample = loader(item)
        target = np.asarray(sample["target"], dtype=bool)
        ignore = np.asarray(sample["ignore"], dtype=bool)
        row: dict[str, object] = {"name": sample["name"]}
        for method in args.methods:
            if args.record_timing and args.freeref_backend == "gpu" and method == "freeref":
                torch.cuda.synchronize()
            start = time.perf_counter()
            output_probability = apply_method(method, sample)
            if args.record_timing and args.freeref_backend == "gpu" and method == "freeref":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            if method == "base" and bool(sample["hard_input"]):
                mask = resize_mask(np.asarray(sample["hard_mask"]), target.shape)
            else:
                probability_at_target = resize_float(output_probability, target.shape)
                mask = probability_at_target >= args.threshold
            mask &= ~ignore
            iou, intersection, union = mask_iou(mask, target)
            boundary = boundary_iou(mask, target, args.boundary_tolerance)
            totals[method]["intersection"] += intersection
            totals[method]["union"] += union
            totals[method]["iou"] += iou
            totals[method]["boundary"] += boundary
            if args.record_timing:
                totals[method]["times"].append(elapsed)
                row[f"{method}_seconds"] = elapsed
            row[f"{method}_iou"] = iou
            row[f"{method}_boundary_iou"] = boundary
        rows.append(row)

    with (args.output_dir / "eval_rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    base = totals["base"] if "base" in totals else None
    summary_methods: dict[str, dict[str, object]] = {}
    for method in args.methods:
        value = totals[method]
        metrics: dict[str, object] = {
            "mIoU": float(value["iou"]) / len(rows),
            "cIoU": int(value["intersection"]) / max(int(value["union"]), 1),
            "bIoU": float(value["boundary"]) / len(rows),
        }
        if base is not None:
            metrics["delta_mIoU"] = float(metrics["mIoU"]) - float(base["iou"]) / len(rows)
            metrics["delta_cIoU"] = float(metrics["cIoU"]) - int(base["intersection"]) / max(
                int(base["union"]), 1
            )
            metrics["delta_bIoU"] = float(metrics["bIoU"]) - float(base["boundary"]) / len(rows)
        times = list(value["times"])
        if times:
            metrics.update(
                {
                    "mean_seconds": statistics.fmean(times),
                    "median_seconds": statistics.median(times),
                    "p95_seconds": _percentile(times, 95),
                }
            )
        summary_methods[method] = metrics

    summary = {
        "protocol": "paired_saved_output_postprocess_baselines_v1",
        "source": args.source,
        "model": args.model_label,
        "split": args.split_name,
        "samples": len(rows),
        "input_representation": "hard_mask" if args.source == "text4seg" else "soft_probability",
        "hard_mask_probability_mapping": (
            [1.0 - args.hard_mask_confidence, args.hard_mask_confidence]
            if args.source == "text4seg"
            else None
        ),
        "record_timing": args.record_timing,
        "timing_scope": "postprocess_only_image_and_saved_prediction_already_loaded",
        "freeref_backend": args.freeref_backend,
        "baseline_config": baseline_config.to_dict(),
        "freeref_config": freeref_config.__dict__,
        "methods": summary_methods,
        "rows_csv": str((args.output_dir / "eval_rows.csv").resolve()),
    }
    (args.output_dir / "eval_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
