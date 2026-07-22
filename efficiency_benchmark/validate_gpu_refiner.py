from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from training_free_refine import (
    GpuTrainingFreeUncertaintyRefiner,
    TrainingFreeRefineConfig,
    TrainingFreeUncertaintyRefiner,
)
from training_free_refine.eval_stamp_dumps import boundary_iou, mask_iou, resolve_asset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare CPU and GPU FreeRef on paired STAMP dumps.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pattern", default="*.pt")
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--boundary-tolerance", type=int, default=2)
    parser.add_argument("--min-mask-iou", type=float, default=0.98)
    parser.add_argument("--max-miou-gap", type=float, default=0.005)
    parser.add_argument("--max-ciou-gap", type=float, default=0.005)
    return parser.parse_args()


def _load_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L")) > 0


def _resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if mask.shape == shape:
        return mask
    return np.asarray(
        Image.fromarray(mask.astype(np.uint8) * 255).resize(
            (shape[1], shape[0]), Image.Resampling.NEAREST
        )
    ) > 0


def _totals() -> dict[str, float]:
    return {"intersection": 0.0, "union": 0.0, "iou": 0.0, "boundary": 0.0}


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("GPU FreeRef validation requires CUDA.")
    if args.limit <= 0:
        raise ValueError("--limit must be positive.")
    dump_paths = sorted(args.input_dir.glob(args.pattern))
    if len(dump_paths) < args.limit:
        raise FileNotFoundError(
            f"Need {args.limit} dumps, but found {len(dump_paths)} in {args.input_dir}."
        )
    dump_paths = random.Random(args.seed).sample(dump_paths, args.limit)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = TrainingFreeRefineConfig()
    cpu_refiner = TrainingFreeUncertaintyRefiner(config)
    gpu_refiner = GpuTrainingFreeUncertaintyRefiner(config)
    totals = {name: _totals() for name in ("coarse", "cpu", "gpu")}
    rows: list[dict[str, object]] = []

    for dump_path in tqdm(dump_paths, desc="CPU/GPU FreeRef validation", dynamic_ncols=True):
        payload = torch.load(dump_path, map_location="cpu", weights_only=False)
        image_path = resolve_asset(payload["image_path"], dump_path)
        target_path = resolve_asset(payload["mask_path"], dump_path)
        with Image.open(image_path) as source:
            image = np.asarray(source.convert("RGB"))
        target = _load_mask(target_path)
        ignore = np.zeros(target.shape, dtype=bool)
        if payload.get("ignore_path"):
            ignore_path = resolve_asset(payload["ignore_path"], dump_path)
            ignore = _resize_mask(_load_mask(ignore_path), target.shape)
            target &= ~ignore
        if image.shape[:2] != target.shape:
            image = np.asarray(
                Image.fromarray(image).resize(
                    (target.shape[1], target.shape[0]), Image.Resampling.BILINEAR
                )
            )

        started = time.perf_counter()
        cpu_output = cpu_refiner.refine(
            image, payload["mask_logits"], tuple(payload["grid_hw"])
        )
        cpu_seconds = time.perf_counter() - started
        torch.cuda.synchronize()
        started = time.perf_counter()
        gpu_output = gpu_refiner.refine(
            image,
            payload["mask_logits"].to("cuda", non_blocking=False),
            tuple(payload["grid_hw"]),
        )
        torch.cuda.synchronize()
        gpu_seconds = time.perf_counter() - started

        coarse = cpu_output["coarse_probability"].numpy() >= config.threshold
        cpu_probability = cpu_output["refined_probability"].numpy()
        gpu_probability = gpu_output["refined_probability"].float().cpu().numpy()
        cpu_mask = cpu_probability >= config.threshold
        gpu_mask = gpu_probability >= config.threshold
        coarse &= ~ignore
        cpu_mask &= ~ignore
        gpu_mask &= ~ignore
        masks = {"coarse": coarse, "cpu": cpu_mask, "gpu": gpu_mask}
        metrics: dict[str, float] = {}
        for name, mask in masks.items():
            iou, intersection, union = mask_iou(mask, target)
            boundary = boundary_iou(mask, target, args.boundary_tolerance)
            totals[name]["intersection"] += intersection
            totals[name]["union"] += union
            totals[name]["iou"] += iou
            totals[name]["boundary"] += boundary
            metrics[f"{name}_iou"] = iou
            metrics[f"{name}_boundary_iou"] = boundary
        mask_agreement_iou, _, _ = mask_iou(cpu_mask, gpu_mask)
        rows.append(
            {
                "name": payload.get("name", dump_path.stem),
                "dump": str(dump_path),
                **metrics,
                "cpu_gpu_mask_iou": mask_agreement_iou,
                "cpu_gpu_pixel_agreement": float(np.mean(cpu_mask == gpu_mask)),
                "probability_mae": float(np.mean(np.abs(cpu_probability - gpu_probability))),
                "cpu_seconds": cpu_seconds,
                "gpu_seconds": gpu_seconds,
            }
        )

    count = len(rows)
    metric_summary = {}
    for name in ("coarse", "cpu", "gpu"):
        metric_summary[name] = {
            "mIoU": totals[name]["iou"] / count,
            "cIoU": totals[name]["intersection"] / max(totals[name]["union"], 1.0),
            "boundary_iou": totals[name]["boundary"] / count,
        }
    miou_gap = metric_summary["gpu"]["mIoU"] - metric_summary["cpu"]["mIoU"]
    ciou_gap = metric_summary["gpu"]["cIoU"] - metric_summary["cpu"]["cIoU"]
    boundary_gap = (
        metric_summary["gpu"]["boundary_iou"] - metric_summary["cpu"]["boundary_iou"]
    )
    mean_mask_iou = float(np.mean([row["cpu_gpu_mask_iou"] for row in rows]))
    validation_pass = (
        mean_mask_iou >= args.min_mask_iou
        and abs(miou_gap) <= args.max_miou_gap
        and abs(ciou_gap) <= args.max_ciou_gap
    )
    rows_path = args.output_dir / "rows.csv"
    with rows_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "source": "paired_cpu_gpu_freeref_validation",
        "samples": count,
        "seed": args.seed,
        "device": torch.cuda.get_device_name(0),
        "cpu_backend": "skimage-scipy",
        "gpu_backend": "cucim-cupy-local-slic",
        "metrics": metric_summary,
        "gpu_minus_cpu_mIoU": miou_gap,
        "gpu_minus_cpu_cIoU": ciou_gap,
        "gpu_minus_cpu_boundary_iou": boundary_gap,
        "mean_cpu_gpu_mask_iou": mean_mask_iou,
        "mean_cpu_gpu_pixel_agreement": float(
            np.mean([row["cpu_gpu_pixel_agreement"] for row in rows])
        ),
        "mean_probability_mae": float(np.mean([row["probability_mae"] for row in rows])),
        "cpu_seconds_per_sample": float(np.mean([row["cpu_seconds"] for row in rows])),
        "gpu_seconds_per_sample": float(np.mean([row["gpu_seconds"] for row in rows])),
        "thresholds": {
            "min_mean_mask_iou": args.min_mask_iou,
            "max_absolute_mIoU_gap": args.max_miou_gap,
            "max_absolute_cIoU_gap": args.max_ciou_gap,
        },
        "validation_pass": validation_pass,
        "config": asdict(config),
        "rows_csv": str(rows_path.resolve()),
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report = [
        "# CPU/GPU FreeRef Consistency",
        "",
        f"Samples: {count}; device: {summary['device']}; pass: **{validation_pass}**.",
        "",
        "| Backend | mIoU | cIoU | Boundary IoU |",
        "|---|---:|---:|---:|",
        f"| Coarse | {metric_summary['coarse']['mIoU'] * 100:.2f} | {metric_summary['coarse']['cIoU'] * 100:.2f} | {metric_summary['coarse']['boundary_iou'] * 100:.2f} |",
        f"| CPU FreeRef | {metric_summary['cpu']['mIoU'] * 100:.2f} | {metric_summary['cpu']['cIoU'] * 100:.2f} | {metric_summary['cpu']['boundary_iou'] * 100:.2f} |",
        f"| GPU FreeRef | {metric_summary['gpu']['mIoU'] * 100:.2f} | {metric_summary['gpu']['cIoU'] * 100:.2f} | {metric_summary['gpu']['boundary_iou'] * 100:.2f} |",
        "",
        f"GPU-CPU gaps: mIoU {miou_gap * 100:+.3f}, cIoU {ciou_gap * 100:+.3f}, Boundary IoU {boundary_gap * 100:+.3f} points.",
        f"Mean CPU/GPU mask IoU: {mean_mask_iou * 100:.3f}%; pixel agreement: {summary['mean_cpu_gpu_pixel_agreement'] * 100:.3f}%.",
    ]
    (args.output_dir / "comparison.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("\n".join(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
