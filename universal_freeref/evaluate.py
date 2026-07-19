from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from tqdm import tqdm

from training_free_refine.refiner import (
    TrainingFreeRefineConfig,
    TrainingFreeUncertaintyRefiner,
    boundary_uncertainty,
    probability_uncertainty,
)

from .io import load_mask, load_probability, load_rgb, resize_mask
from .metrics import boundary_iou, mask_iou, paired_statistics
from .schema import ManifestItem, load_manifest
from .visualize import save_panel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply FreeRef to predictions from any segmentation model.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save-visualizations", type=int, default=12)
    parser.add_argument("--boundary-sigma", type=float, default=8.0)
    parser.add_argument("--boundary-tolerance", type=int, default=2)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
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
    return parser.parse_args()


def _slug(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return value[:100] or "sample"


def _run_signature(args: argparse.Namespace, config: TrainingFreeRefineConfig) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "manifest": str(args.manifest.expanduser().resolve()),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "boundary_sigma": args.boundary_sigma,
        "boundary_tolerance": args.boundary_tolerance,
        "config": asdict(config),
    }


def _validate_files(item: ManifestItem) -> None:
    for label, path in (
        ("image", item.image),
        ("gt_mask", item.gt_mask),
        ("prediction", item.prediction),
        ("ignore_mask", item.ignore_mask),
        ("uncertainty", item.uncertainty),
    ):
        if path is not None and not path.is_file():
            raise FileNotFoundError(f"{label} does not exist for {item.name!r}: {path}")


def _constant_output(probability: np.ndarray, uncertainty: np.ndarray) -> dict[str, Any]:
    return {
        "refined_probability": probability.astype(np.float32),
        "uncertainty_map": uncertainty.astype(np.float32),
        "diagnostics": {
            "segments": 0.0,
            "edges": 0.0,
            "color_scale": 0.0,
            "foreground_seeds": 0.0,
            "background_seeds": 0.0,
            "solver_info": 0.0,
        },
    }


def _evaluate_item(
    item: ManifestItem,
    refiner: TrainingFreeUncertaintyRefiner,
    config: TrainingFreeRefineConfig,
    args: argparse.Namespace,
    index: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    _validate_files(item)
    target = load_mask(item.gt_mask)
    image = load_rgb(item.image)
    if image.shape[:2] != target.shape:
        image = np.asarray(
            Image.fromarray(image).resize((target.shape[1], target.shape[0]), Image.Resampling.BILINEAR)
        )
    ignore = (
        load_mask(item.ignore_mask, target.shape)
        if item.ignore_mask is not None
        else np.zeros(target.shape, dtype=bool)
    )
    target = target & ~ignore
    probability, supplied_uncertainty = load_probability(item, target.shape)
    threshold = item.threshold if item.threshold != 0.5 else config.threshold
    coarse = probability >= threshold

    start = time.perf_counter()
    if item.prediction_kind == "mask":
        uncertainty = (
            supplied_uncertainty
            if supplied_uncertainty is not None
            else boundary_uncertainty(coarse, args.boundary_sigma)
        )
        if not coarse.any() or coarse.all():
            output = _constant_output(probability, uncertainty)
        else:
            output = refiner.refine_probability(image, probability, uncertainty=uncertainty)
    else:
        uncertainty = (
            supplied_uncertainty
            if supplied_uncertainty is not None
            else probability_uncertainty(probability)
        )
        if item.no_target and not coarse.any():
            output = _constant_output(probability, uncertainty)
        else:
            output = refiner.refine_probability(image, probability, uncertainty=uncertainty)
    elapsed = time.perf_counter() - start
    refined_probability = np.asarray(output["refined_probability"], dtype=np.float32)
    refined = refined_probability >= threshold
    coarse &= ~ignore
    refined &= ~ignore

    coarse_iou, coarse_inter, coarse_union = mask_iou(coarse, target)
    refined_iou, refined_inter, refined_union = mask_iou(refined, target)
    corrected = (coarse != target) & (refined == target) & ~ignore
    regressed = (coarse == target) & (refined != target) & ~ignore
    valid_pixels = max(int((~ignore).sum()), 1)
    uncertainty_values = np.asarray(output["uncertainty_map"], dtype=np.float32)
    diagnostics = dict(output["diagnostics"])
    row: dict[str, Any] = {
        "index": index,
        "name": item.name,
        "method": item.method,
        "split": item.split,
        "instance_id": item.instance_id or "",
        "prediction_kind": item.prediction_kind,
        "image": str(item.image),
        "gt_mask": str(item.gt_mask),
        "prediction": str(item.prediction),
        "no_target": item.no_target,
        "coarse_iou": coarse_iou,
        "refined_iou": refined_iou,
        "iou_delta": refined_iou - coarse_iou,
        "coarse_boundary_iou": boundary_iou(coarse, target, args.boundary_tolerance),
        "refined_boundary_iou": boundary_iou(refined, target, args.boundary_tolerance),
        "uncertain_fraction": float((uncertainty_values >= 0.5).sum() / valid_pixels),
        "corrected_fraction": float(corrected.sum() / valid_pixels),
        "regressed_fraction": float(regressed.sum() / valid_pixels),
        "object_fraction": float(target.sum() / valid_pixels),
        "seconds": elapsed,
        "_coarse_intersection": coarse_inter,
        "_coarse_union": coarse_union,
        "_refined_intersection": refined_inter,
        "_refined_union": refined_union,
        **diagnostics,
    }
    arrays = {
        "image": image,
        "target": target,
        "coarse": coarse,
        "refined": refined,
        "uncertainty": uncertainty_values,
    }
    return row, arrays


def _write_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255).save(path)


def _summary(rows: list[dict[str, Any]], config: TrainingFreeRefineConfig, args: argparse.Namespace) -> dict[str, Any]:
    count = len(rows)
    coarse_inter = sum(int(row["_coarse_intersection"]) for row in rows)
    coarse_union = sum(int(row["_coarse_union"]) for row in rows)
    refined_inter = sum(int(row["_refined_intersection"]) for row in rows)
    refined_union = sum(int(row["_refined_union"]) for row in rows)
    deltas = [float(row["iou_delta"]) for row in rows]
    coarse_mean_iou = float(np.mean([row["coarse_iou"] for row in rows]))
    refined_mean_iou = float(np.mean([row["refined_iou"] for row in rows]))
    improved = sum(delta > 1e-12 for delta in deltas)
    degraded = sum(delta < -1e-12 for delta in deltas)
    no_target_rows = [row for row in rows if bool(row["no_target"])]
    summary: dict[str, Any] = {
        "source": "universal_freeref_manifest",
        "method": sorted({str(row["method"]) for row in rows}),
        "split": sorted({str(row["split"]) for row in rows}),
        "samples": count,
        "coarse_mean_iou": coarse_mean_iou,
        "refined_mean_iou": refined_mean_iou,
        "mean_iou_delta": float(np.mean(deltas)),
        "coarse_gIoU": coarse_mean_iou,
        "refined_gIoU": refined_mean_iou,
        "gIoU_delta": refined_mean_iou - coarse_mean_iou,
        "coarse_cIoU": coarse_inter / max(coarse_union, 1),
        "refined_cIoU": refined_inter / max(refined_union, 1),
        "cIoU_delta": refined_inter / max(refined_union, 1) - coarse_inter / max(coarse_union, 1),
        "coarse_boundary_iou": float(np.mean([row["coarse_boundary_iou"] for row in rows])),
        "refined_boundary_iou": float(np.mean([row["refined_boundary_iou"] for row in rows])),
        "boundary_iou_delta": float(
            np.mean([row["refined_boundary_iou"] - row["coarse_boundary_iou"] for row in rows])
        ),
        "improved_samples": improved,
        "degraded_samples": degraded,
        "unchanged_samples": count - improved - degraded,
        "improvement_rate": improved / count,
        "seconds_per_sample": float(np.mean([row["seconds"] for row in rows])),
        "mean_uncertain_fraction": float(np.mean([row["uncertain_fraction"] for row in rows])),
        "mean_corrected_fraction": float(np.mean([row["corrected_fraction"] for row in rows])),
        "mean_regressed_fraction": float(np.mean([row["regressed_fraction"] for row in rows])),
        "no_target_samples": len(no_target_rows),
        "no_target_coarse_gIoU": (
            float(np.mean([row["coarse_iou"] for row in no_target_rows])) if no_target_rows else None
        ),
        "no_target_refined_gIoU": (
            float(np.mean([row["refined_iou"] for row in no_target_rows])) if no_target_rows else None
        ),
        "boundary_sigma": args.boundary_sigma,
        "config": asdict(config),
        **paired_statistics(deltas, args.bootstrap_samples, args.seed),
    }
    return summary


def main() -> int:
    args = parse_args()
    if min(args.limit, args.save_visualizations, args.boundary_tolerance, args.bootstrap_samples) < 0:
        raise ValueError("limit, save-visualizations, boundary-tolerance and bootstrap-samples must be non-negative.")
    items = load_manifest(args.manifest)
    if args.limit:
        items = items[: args.limit]
    config = TrainingFreeRefineConfig(
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
    refiner = TrainingFreeUncertaintyRefiner(config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    signature = _run_signature(args, config)
    run_config_path = args.output_dir / "run_config.json"
    if run_config_path.exists() and not args.overwrite:
        existing = json.loads(run_config_path.read_text(encoding="utf-8"))
        if existing != signature:
            raise RuntimeError(
                f"Output directory contains a different run configuration: {args.output_dir}. "
                "Use --overwrite or choose another directory."
            )
    run_config_path.write_text(json.dumps(signature, indent=2, ensure_ascii=True), encoding="utf-8")

    rows: list[dict[str, Any]] = []
    for index, item in enumerate(tqdm(items, desc="universal FreeRef", dynamic_ncols=True)):
        stem = f"{index:08d}_{_slug(item.name)}"
        cache_path = args.output_dir / "row_cache" / f"{stem}.json"
        refined_path = args.output_dir / "refined_masks" / f"{stem}.png"
        if args.resume and cache_path.is_file() and refined_path.is_file() and not args.overwrite:
            rows.append(json.loads(cache_path.read_text(encoding="utf-8")))
            continue
        row, arrays = _evaluate_item(item, refiner, config, args, index)
        _write_mask(args.output_dir / "coarse_masks" / f"{stem}.png", arrays["coarse"])
        _write_mask(refined_path, arrays["refined"])
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(row, indent=2, ensure_ascii=True), encoding="utf-8")
        if index < args.save_visualizations:
            save_panel(
                args.output_dir / "visualizations" / f"{stem}.png",
                arrays["image"],
                arrays["target"],
                arrays["coarse"],
                arrays["refined"],
                arrays["uncertainty"],
            )
        rows.append(row)

    rows.sort(key=lambda row: int(row["index"]))
    csv_rows = [{key: value for key, value in row.items() if not key.startswith("_")} for row in rows]
    rows_path = args.output_dir / "eval_rows.csv"
    with rows_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames: list[str] = []
        for row in csv_rows:
            fieldnames.extend(key for key in row if key not in fieldnames)
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    summary = _summary(rows, config, args)
    summary["rows_csv"] = str(rows_path.resolve())
    summary["manifest"] = str(args.manifest.resolve())
    summary_path = args.output_dir / "eval_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
