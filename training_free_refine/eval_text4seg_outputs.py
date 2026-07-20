from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from .eval_stamp_dumps import boundary_iou, mask_iou, save_visualization
from .refiner import TrainingFreeRefineConfig, TrainingFreeUncertaintyRefiner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate training-free refinement on official Text4Seg masks.")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--input-dir", type=Path)
    inputs.add_argument("--manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pattern", default="*_pred_mask.png")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--boundary-sigma", type=float, default=8.0)
    parser.add_argument("--boundary-tolerance", type=int, default=2)
    parser.add_argument("--save-visualizations", type=int, default=8)
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


def sibling(path: Path, old_suffix: str, new_suffix: str) -> Path:
    if not path.name.endswith(old_suffix):
        raise ValueError(f"{path} does not end with {old_suffix!r}.")
    return path.with_name(path.name[: -len(old_suffix)] + new_suffix)


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def load_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > 0


def resolve_manifest_path(value: str, manifest: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else manifest.parent / path


def evaluation_items(args: argparse.Namespace) -> list[dict[str, object]]:
    if args.manifest is not None:
        items = []
        for line_number, line in enumerate(args.manifest.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"Manifest line {line_number} is not a JSON object.")
            items.append(
                {
                    "name": str(value.get("name") or f"sample_{line_number:08d}"),
                    "pred": resolve_manifest_path(str(value["pred_mask"]), args.manifest),
                    "gt": resolve_manifest_path(str(value["gt_mask"]), args.manifest),
                    "image": resolve_manifest_path(str(value["image"]), args.manifest),
                    "sam": resolve_manifest_path(str(value["sam_mask"]), args.manifest)
                    if value.get("sam_mask")
                    else None,
                }
            )
        return items

    paths = sorted(path for path in args.input_dir.rglob(args.pattern) if path.is_file())
    return [
        {
            "name": str(path.relative_to(args.input_dir)),
            "pred": path,
            "gt": sibling(path, "_pred_mask.png", "_gt_mask.png"),
            "image": sibling(path, "_pred_mask.png", "_image.png"),
            "sam": sibling(path, "_pred_mask.png", "_sam_mask.png"),
        }
        for path in paths
    ]


def resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if mask.shape == shape:
        return mask.astype(bool)
    image = Image.fromarray(mask.astype(np.uint8) * 255)
    return np.asarray(image.resize((shape[1], shape[0]), Image.Resampling.NEAREST)) > 0


def resize_float(values: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if values.shape == shape:
        return values
    image = Image.fromarray(values.astype(np.float32), mode="F")
    return np.asarray(image.resize((shape[1], shape[0]), Image.Resampling.BILINEAR))


def main() -> int:
    args = parse_args()
    if args.limit < 0 or args.save_visualizations < 0 or args.boundary_tolerance < 0:
        raise ValueError("limit, save-visualizations, and boundary-tolerance must be non-negative.")
    items = evaluation_items(args)
    if args.limit:
        items = items[: args.limit]
    if not items:
        raise FileNotFoundError("No Text4Seg evaluation items were found.")

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
    rows: list[dict[str, object]] = []
    totals = {"coarse_inter": 0, "coarse_union": 0, "refined_inter": 0, "refined_union": 0}
    sam_totals = {"inter": 0, "union": 0, "count": 0, "mean_iou": 0.0}
    elapsed_total = 0.0

    for index, item in enumerate(tqdm(items, desc="Text4Seg training-free refine", dynamic_ncols=True)):
        pred_path = Path(item["pred"])
        gt_path = Path(item["gt"])
        image_path = Path(item["image"])
        sam_path = Path(item["sam"]) if item["sam"] is not None else None
        if not gt_path.exists() or not image_path.exists():
            raise FileNotFoundError(f"Missing GT/image sibling for {pred_path}.")

        image = load_rgb(image_path)
        target = load_mask(gt_path)
        coarse_at_image = resize_mask(load_mask(pred_path), image.shape[:2])
        start = time.perf_counter()
        output = refiner.refine_hard_mask(image, coarse_at_image, boundary_sigma=args.boundary_sigma)
        elapsed = time.perf_counter() - start
        elapsed_total += elapsed

        coarse = resize_mask(coarse_at_image, target.shape)
        refined_probability = resize_float(output["refined_probability"].numpy(), target.shape)
        refined = refined_probability >= config.threshold
        coarse_iou, coarse_inter, coarse_union = mask_iou(coarse, target)
        refined_iou, refined_inter, refined_union = mask_iou(refined, target)
        totals["coarse_inter"] += coarse_inter
        totals["coarse_union"] += coarse_union
        totals["refined_inter"] += refined_inter
        totals["refined_union"] += refined_union

        row: dict[str, object] = {
            "name": str(item["name"]),
            "pred_mask": str(pred_path),
            "image": str(image_path),
            "gt_mask": str(gt_path),
            "coarse_iou": coarse_iou,
            "refined_iou": refined_iou,
            "iou_delta": refined_iou - coarse_iou,
            "coarse_boundary_iou": boundary_iou(coarse, target, args.boundary_tolerance),
            "refined_boundary_iou": boundary_iou(refined, target, args.boundary_tolerance),
            "seconds": elapsed,
            **output["diagnostics"],
        }
        if sam_path is not None and sam_path.exists():
            sam = resize_mask(load_mask(sam_path), target.shape)
            sam_iou, sam_inter, sam_union = mask_iou(sam, target)
            sam_totals["inter"] += sam_inter
            sam_totals["union"] += sam_union
            sam_totals["count"] += 1
            sam_totals["mean_iou"] += sam_iou
            row["sam_iou"] = sam_iou
        rows.append(row)

        if index < args.save_visualizations:
            display_image = np.asarray(
                Image.fromarray(image).resize((target.shape[1], target.shape[0]), Image.Resampling.BILINEAR)
            )
            uncertainty = resize_float(output["uncertainty_map"].numpy(), target.shape)
            save_visualization(
                args.output_dir / "visualizations" / f"sample_{index:05d}.png",
                display_image,
                target,
                coarse,
                refined,
                uncertainty,
            )

    rows_path = args.output_dir / "eval_rows.csv"
    with rows_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(rows[0])
        for row in rows[1:]:
            fieldnames.extend(key for key in row if key not in fieldnames)
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    count = len(rows)
    coarse_ciou = totals["coarse_inter"] / max(totals["coarse_union"], 1)
    refined_ciou = totals["refined_inter"] / max(totals["refined_union"], 1)
    improved = sum(float(row["iou_delta"]) > 1e-12 for row in rows)
    degraded = sum(float(row["iou_delta"]) < -1e-12 for row in rows)
    export_metadata: dict[str, object] = {}
    if args.manifest is not None:
        export_summary_path = args.manifest.parent / "export_summary.json"
        if export_summary_path.is_file():
            value = json.loads(export_summary_path.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                for key in (
                    "model",
                    "eval_json",
                    "descriptor_grid_size",
                    "descriptor_count",
                    "checkpoint_grid_size",
                    "evaluation_protocol",
                ):
                    if key in value:
                        export_metadata[key] = value[key]

    summary: dict[str, object] = {
        "source": "official_text4seg_saved_masks",
        "base_export": export_metadata,
        "samples": count,
        "coarse_mean_iou": sum(float(row["coarse_iou"]) for row in rows) / count,
        "refined_mean_iou": sum(float(row["refined_iou"]) for row in rows) / count,
        "mean_iou_delta": sum(float(row["iou_delta"]) for row in rows) / count,
        "coarse_cIoU": coarse_ciou,
        "refined_cIoU": refined_ciou,
        "cIoU_delta": refined_ciou - coarse_ciou,
        "coarse_boundary_iou": sum(float(row["coarse_boundary_iou"]) for row in rows) / count,
        "refined_boundary_iou": sum(float(row["refined_boundary_iou"]) for row in rows) / count,
        "boundary_iou_delta": sum(
            float(row["refined_boundary_iou"]) - float(row["coarse_boundary_iou"]) for row in rows
        )
        / count,
        "improved_samples": improved,
        "degraded_samples": degraded,
        "unchanged_samples": count - improved - degraded,
        "seconds_per_sample": elapsed_total / count,
        "boundary_sigma": args.boundary_sigma,
        "config": asdict(config),
        "rows_csv": str(rows_path),
    }
    if sam_totals["count"]:
        summary.update(
            {
                "sam_samples": sam_totals["count"],
                "sam_mean_iou": sam_totals["mean_iou"] / sam_totals["count"],
                "sam_cIoU": sam_totals["inter"] / max(sam_totals["union"], 1),
            }
        )
    summary_path = args.output_dir / "eval_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
