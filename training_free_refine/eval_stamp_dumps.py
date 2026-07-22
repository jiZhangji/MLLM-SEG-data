from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion
from tqdm import tqdm

from .refiner import TrainingFreeRefineConfig, TrainingFreeUncertaintyRefiner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate training-free uncertainty refinement on STAMP dumps.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pattern", default="*.pt")
    parser.add_argument("--limit", type=int, default=0)
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
    parser.add_argument(
        "--uncertainty-aware-anchoring",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--appearance-weighted-graph",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--selective-fusion",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--boundary-tolerance", type=int, default=2)
    parser.add_argument("--save-visualizations", type=int, default=8)
    return parser.parse_args()


def resolve_asset(path_value: str | Path, dump_path: Path) -> Path:
    path = Path(path_value)
    candidates = [
        path,
        dump_path.parent / path,
        dump_path.parent / path.name,
        Path.cwd() / path,
        Path.cwd().parent / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not resolve {path_value!r} for dump {dump_path}.")


def mask_iou(prediction: np.ndarray, target: np.ndarray) -> tuple[float, int, int]:
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    intersection = int(np.logical_and(prediction, target).sum())
    union = int(np.logical_or(prediction, target).sum())
    return (float(intersection / union) if union else 1.0), intersection, union


def boundary_iou(prediction: np.ndarray, target: np.ndarray, tolerance: int) -> float:
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    pred_boundary = np.logical_xor(prediction, binary_erosion(prediction))
    target_boundary = np.logical_xor(target, binary_erosion(target))
    if tolerance > 0:
        pred_boundary = binary_dilation(pred_boundary, iterations=tolerance)
        target_boundary = binary_dilation(target_boundary, iterations=tolerance)
    union = np.logical_or(pred_boundary, target_boundary).sum()
    if not union:
        return 1.0
    return float(np.logical_and(pred_boundary, target_boundary).sum() / union)


def overlay(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> Image.Image:
    base = image.astype(np.float32)
    tint = np.zeros_like(base)
    tint[...] = np.asarray(color, dtype=np.float32)
    alpha = mask.astype(np.float32)[..., None] * 0.45
    return Image.fromarray(np.clip(base * (1.0 - alpha) + tint * alpha, 0, 255).astype(np.uint8))


def save_visualization(
    output_path: Path,
    image: np.ndarray,
    target: np.ndarray,
    coarse: np.ndarray,
    refined: np.ndarray,
    uncertainty: np.ndarray,
) -> None:
    uncertainty_rgb = np.stack(
        [uncertainty, np.sqrt(np.clip(uncertainty, 0.0, 1.0)), 1.0 - uncertainty], axis=-1
    )
    panels = [
        Image.fromarray(image),
        overlay(image, target, (40, 220, 80)),
        overlay(image, coarse, (230, 60, 50)),
        Image.fromarray(np.clip(uncertainty_rgb * 255.0, 0, 255).astype(np.uint8)),
        overlay(image, refined, (50, 120, 240)),
    ]
    width, height = panels[0].size
    canvas = Image.new("RGB", (width * len(panels), height))
    for index, panel in enumerate(panels):
        canvas.paste(panel, (index * width, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def main() -> int:
    args = parse_args()
    if args.limit < 0 or args.boundary_tolerance < 0 or args.save_visualizations < 0:
        raise ValueError("limit, boundary-tolerance, and save-visualizations must be non-negative.")
    dump_paths = sorted(args.input_dir.glob(args.pattern))
    if args.limit:
        dump_paths = dump_paths[: args.limit]
    if not dump_paths:
        raise FileNotFoundError(f"No dumps matched {args.pattern!r} in {args.input_dir}.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

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
        uncertainty_aware_anchoring=args.uncertainty_aware_anchoring,
        appearance_weighted_graph=args.appearance_weighted_graph,
        selective_fusion=args.selective_fusion,
    )
    refiner = TrainingFreeUncertaintyRefiner(config)
    rows = []
    coarse_intersection = coarse_union = 0
    refined_intersection = refined_union = 0
    elapsed_total = 0.0

    for sample_index, dump_path in enumerate(tqdm(dump_paths, desc="training-free refine", dynamic_ncols=True)):
        payload = torch.load(dump_path, map_location="cpu", weights_only=False)
        image_path = resolve_asset(payload["image_path"], dump_path)
        mask_path = resolve_asset(payload["mask_path"], dump_path)
        image = np.asarray(Image.open(image_path).convert("RGB"))
        target = np.asarray(Image.open(mask_path).convert("L")) > 0
        ignore_path = None
        ignore = np.zeros(target.shape, dtype=bool)
        if payload.get("ignore_path"):
            ignore_path = resolve_asset(payload["ignore_path"], dump_path)
            ignore = np.asarray(Image.open(ignore_path).convert("L")) > 0
            if ignore.shape != target.shape:
                ignore = np.asarray(
                    Image.fromarray(ignore.astype(np.uint8) * 255).resize(
                        (target.shape[1], target.shape[0]), Image.Resampling.NEAREST
                    )
                ) > 0
            target &= ~ignore
        if image.shape[:2] != target.shape:
            image = np.asarray(Image.fromarray(image).resize((target.shape[1], target.shape[0]), Image.Resampling.BILINEAR))

        start = time.perf_counter()
        output = refiner.refine(image, payload["mask_logits"], tuple(payload["grid_hw"]))
        elapsed = time.perf_counter() - start
        elapsed_total += elapsed
        coarse_probability = output["coarse_probability"].numpy()
        refined_probability = output["refined_probability"].numpy()
        coarse_mask = coarse_probability >= config.threshold
        refined_mask = refined_probability >= config.threshold
        coarse_mask &= ~ignore
        refined_mask &= ~ignore
        coarse_iou, coarse_inter, coarse_uni = mask_iou(coarse_mask, target)
        refined_iou, refined_inter, refined_uni = mask_iou(refined_mask, target)
        coarse_intersection += coarse_inter
        coarse_union += coarse_uni
        refined_intersection += refined_inter
        refined_union += refined_uni
        diagnostics = output["diagnostics"]
        rows.append(
            {
                "name": payload.get("name", dump_path.stem),
                "dump": str(dump_path),
                "image": str(image_path),
                "mask": str(mask_path),
                "ignore_mask": str(ignore_path) if ignore_path is not None else "",
                "no_target": bool(payload.get("source_item", {}).get("no_target", False)),
                "empty_prediction": bool(payload.get("empty_prediction", False)),
                "prediction_error": str(payload.get("prediction_error", "")),
                "grid_h": int(payload["grid_hw"][0]),
                "grid_w": int(payload["grid_hw"][1]),
                "coarse_iou": coarse_iou,
                "refined_iou": refined_iou,
                "iou_delta": refined_iou - coarse_iou,
                "coarse_boundary_iou": boundary_iou(coarse_mask, target, args.boundary_tolerance),
                "refined_boundary_iou": boundary_iou(refined_mask, target, args.boundary_tolerance),
                "seconds": elapsed,
                **diagnostics,
            }
        )
        if sample_index < args.save_visualizations:
            save_visualization(
                args.output_dir / "visualizations" / f"{dump_path.stem}.png",
                image,
                target,
                coarse_mask,
                refined_mask,
                output["uncertainty_map"].numpy(),
            )

    rows_path = args.output_dir / "eval_rows.csv"
    with rows_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    count = len(rows)
    improved = sum(row["iou_delta"] > 1e-12 for row in rows)
    degraded = sum(row["iou_delta"] < -1e-12 for row in rows)
    no_target_rows = [row for row in rows if row["no_target"]]
    summary = {
        "samples": count,
        "coarse_mean_iou": sum(row["coarse_iou"] for row in rows) / count,
        "refined_mean_iou": sum(row["refined_iou"] for row in rows) / count,
        "mean_iou_delta": sum(row["iou_delta"] for row in rows) / count,
        "coarse_cIoU": coarse_intersection / max(coarse_union, 1),
        "refined_cIoU": refined_intersection / max(refined_union, 1),
        "cIoU_delta": refined_intersection / max(refined_union, 1) - coarse_intersection / max(coarse_union, 1),
        "coarse_boundary_iou": sum(row["coarse_boundary_iou"] for row in rows) / count,
        "refined_boundary_iou": sum(row["refined_boundary_iou"] for row in rows) / count,
        "boundary_iou_delta": sum(row["refined_boundary_iou"] - row["coarse_boundary_iou"] for row in rows) / count,
        "improved_samples": improved,
        "degraded_samples": degraded,
        "unchanged_samples": count - improved - degraded,
        "no_target_samples": len(no_target_rows),
        "no_target_coarse_gIoU": (
            sum(row["coarse_iou"] for row in no_target_rows) / len(no_target_rows) if no_target_rows else None
        ),
        "no_target_refined_gIoU": (
            sum(row["refined_iou"] for row in no_target_rows) / len(no_target_rows) if no_target_rows else None
        ),
        "empty_prediction_samples": sum(bool(row["empty_prediction"]) for row in rows),
        "seconds_per_sample": elapsed_total / count,
        "config": asdict(config),
        "rows_csv": str(rows_path),
    }
    summary_path = args.output_dir / "eval_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
