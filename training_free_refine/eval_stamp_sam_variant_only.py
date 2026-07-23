from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from .eval_stamp_dumps import boundary_iou, mask_iou, resolve_asset
from .eval_stamp_sam_h import load_sam, sam_protocol_name, sam_refine, stable_sample_seed
from .eval_text4seg_outputs import resize_mask
from .refiner import stamp_probability


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one frozen-SAM variant on coarse STAMP masks.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stamp-code-dir", type=Path, required=True)
    parser.add_argument("--sam-path", type=Path, required=True)
    parser.add_argument("--sam-model-type", choices=("vit_b", "vit_l", "vit_h"), required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--split-name", required=True)
    parser.add_argument("--pattern", default="*.pt")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--point-count", type=int, default=10)
    parser.add_argument("--cascade-steps", type=int, default=2)
    parser.add_argument("--boundary-tolerance", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > 0


def save_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255).save(path)


def main() -> int:
    args = parse_args()
    if args.limit < 0 or args.boundary_tolerance < 0 or args.cascade_steps < 0:
        raise ValueError("limit, boundary-tolerance, and cascade-steps must be non-negative.")
    if args.point_count != 10:
        raise ValueError("The released STAMP frozen-SAM protocol requires 10 points per class.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for frozen-SAM evaluation.")
    minimum_bytes = {"vit_b": 300_000_000, "vit_l": 900_000_000, "vit_h": 2_000_000_000}
    if not args.sam_path.is_file() or args.sam_path.stat().st_size < minimum_bytes[args.sam_model_type]:
        raise FileNotFoundError(f"Incomplete SAM checkpoint: {args.sam_path}")
    dumps = sorted(args.input_dir.glob(args.pattern))
    if args.limit:
        dumps = dumps[: args.limit]
    if not dumps:
        raise FileNotFoundError(f"No dumps found below {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sam_mask_dir = args.output_dir / "sam_masks"
    predictor, compute_logits, sample_points, official_utils = load_sam(
        args.stamp_code_dir, args.sam_path, args.sam_model_type
    )
    totals = {
        "coarse": {"intersection": 0, "union": 0, "iou": 0.0, "boundary": 0.0},
        "sam": {"intersection": 0, "union": 0, "iou": 0.0, "boundary": 0.0},
    }
    rows: list[dict[str, object]] = []
    failures = 0

    for dump_path in tqdm(dumps, desc=f"SAM {args.sam_model_type} {args.model_label} {args.split_name}"):
        payload = torch.load(dump_path, map_location="cpu", weights_only=False)
        image_path = resolve_asset(payload["image_path"], dump_path)
        target_path = resolve_asset(payload["mask_path"], dump_path)
        image = np.asarray(Image.open(image_path).convert("RGB"))
        target = load_mask(target_path)
        if image.shape[:2] != target.shape:
            image = np.asarray(
                Image.fromarray(image).resize((target.shape[1], target.shape[0]), Image.Resampling.BILINEAR)
            )
        ignore = np.zeros(target.shape, dtype=bool)
        if payload.get("ignore_path"):
            ignore_path = resolve_asset(payload["ignore_path"], dump_path)
            ignore = resize_mask(load_mask(ignore_path), target.shape)
            target &= ~ignore
        coarse = (
            stamp_probability(payload["mask_logits"], tuple(payload["grid_hw"]), image.shape[:2]).numpy()
            >= args.threshold
        )
        coarse &= ~ignore
        sam_path = sam_mask_dir / f"{dump_path.stem}.png"
        elapsed = 0.0
        error_text = ""
        if sam_path.is_file():
            sam_mask = resize_mask(load_mask(sam_path), target.shape) & ~ignore
        else:
            try:
                torch.cuda.synchronize()
                started = time.perf_counter()
                if coarse.any():
                    predictor.set_image(image)
                    sam_mask = sam_refine(
                        predictor,
                        compute_logits,
                        sample_points,
                        coarse,
                        args.cascade_steps,
                        stable_sample_seed(args.seed, dump_path.stem),
                    )
                else:
                    sam_mask = coarse.copy()
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
            except Exception as error:
                failures += 1
                sam_mask = coarse.copy()
                error_text = repr(error)
            sam_mask &= ~ignore
            save_mask(sam_path, sam_mask)

        row: dict[str, object] = {
            "name": payload.get("name", dump_path.stem),
            "dump": str(dump_path),
            "image": str(image_path),
            "target": str(target_path),
            "sam_seconds": elapsed,
            "sam_error": error_text,
        }
        for branch, mask in (("coarse", coarse), ("sam", sam_mask)):
            iou, intersection, union = mask_iou(mask, target)
            boundary = boundary_iou(mask, target, args.boundary_tolerance)
            totals[branch]["intersection"] += intersection
            totals[branch]["union"] += union
            totals[branch]["iou"] += iou
            totals[branch]["boundary"] += boundary
            row[f"{branch}_iou"] = iou
            row[f"{branch}_boundary_iou"] = boundary
        row["sam_delta"] = float(row["sam_iou"]) - float(row["coarse_iou"])
        rows.append(row)

    rows_path = args.output_dir / "eval_rows.csv"
    with rows_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    count = len(rows)
    summary: dict[str, object] = {
        "model": args.model_label,
        "split": args.split_name,
        "samples": count,
        "protocol": sam_protocol_name(args.sam_model_type),
        "evaluation_scope": "coarse_to_frozen_sam_only",
        "sam_model_type": args.sam_model_type,
        "sam_checkpoint": str(args.sam_path),
        "official_stamp_utils": str(official_utils),
        "sam_failures_with_input_fallback": failures,
        "rows_csv": str(rows_path),
    }
    for branch in ("coarse", "sam"):
        summary[f"{branch}_mean_iou"] = totals[branch]["iou"] / count
        summary[f"{branch}_cIoU"] = totals[branch]["intersection"] / max(totals[branch]["union"], 1)
        summary[f"{branch}_boundary_iou"] = totals[branch]["boundary"] / count
    summary["sam_mean_iou_delta"] = float(summary["sam_mean_iou"]) - float(summary["coarse_mean_iou"])
    summary["sam_cIoU_delta"] = float(summary["sam_cIoU"]) - float(summary["coarse_cIoU"])
    summary["sam_boundary_iou_delta"] = float(summary["sam_boundary_iou"]) - float(
        summary["coarse_boundary_iou"]
    )
    summary["seconds_per_sample"] = sum(float(row["sam_seconds"]) for row in rows) / count
    (args.output_dir / "eval_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
