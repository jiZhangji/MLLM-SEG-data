from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from tqdm import tqdm

from .metrics import boundary_iou, mask_iou, paired_statistics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare LISA's first SAM pass with FreeRef-guided SAM re-decoding."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--boundary-tolerance", type=int, default=2)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        row["_line_number"] = line_number
        rows.append(row)
    if not rows:
        raise ValueError(f"Manifest contains no samples: {path}")
    return rows


def _load_logits(path: Path) -> np.ndarray:
    with np.load(path) as payload:
        if "logits" not in payload:
            raise KeyError(f"Missing 'logits' array in {path}")
        logits = np.asarray(payload["logits"], dtype=np.float32).squeeze()
    if logits.ndim != 2:
        raise ValueError(f"Expected [H, W] logits in {path}, got {logits.shape}")
    return logits


def _load_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8) > 127


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _evaluate_row(row: dict[str, Any], tolerance: int) -> dict[str, Any]:
    baseline_path = Path(row["baseline_prediction"])
    prompted_path = Path(row["prompted_prediction"])
    target_path = Path(row["gt_mask"])
    for path in (baseline_path, prompted_path, target_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing artifact for {row['name']}: {path}")
    baseline = _load_logits(baseline_path) > 0.0
    prompted = _load_logits(prompted_path) > 0.0
    target = _load_mask(target_path)
    if baseline.shape != target.shape or prompted.shape != target.shape:
        raise ValueError(
            f"Shape mismatch for {row['name']}: baseline={baseline.shape}, "
            f"prompted={prompted.shape}, target={target.shape}"
        )
    baseline_iou, baseline_intersection, baseline_union = mask_iou(baseline, target)
    prompted_iou, prompted_intersection, prompted_union = mask_iou(prompted, target)
    return {
        "name": row["name"],
        "instance_id": row.get("instance_id", ""),
        "split": row.get("split", ""),
        "query": row.get("query", ""),
        "baseline_iou": baseline_iou,
        "prompted_iou": prompted_iou,
        "iou_delta": prompted_iou - baseline_iou,
        "baseline_boundary_iou": boundary_iou(baseline, target, tolerance),
        "prompted_boundary_iou": boundary_iou(prompted, target, tolerance),
        "base_seconds": float(row.get("base_seconds", 0.0)),
        "freeref_seconds": float(row.get("freeref_seconds", 0.0)),
        "second_decoder_seconds": float(row.get("second_decoder_seconds", 0.0)),
        "total_seconds": float(row.get("total_seconds", 0.0)),
        "_baseline_intersection": baseline_intersection,
        "_baseline_union": baseline_union,
        "_prompted_intersection": prompted_intersection,
        "_prompted_union": prompted_union,
    }


def summarize_rows(
    rows: list[dict[str, Any]], bootstrap_samples: int, seed: int
) -> dict[str, Any]:
    count = len(rows)
    deltas = [float(row["iou_delta"]) for row in rows]
    baseline_miou = float(np.mean([row["baseline_iou"] for row in rows]))
    prompted_miou = float(np.mean([row["prompted_iou"] for row in rows]))
    baseline_intersection = sum(int(row["_baseline_intersection"]) for row in rows)
    baseline_union = sum(int(row["_baseline_union"]) for row in rows)
    prompted_intersection = sum(int(row["_prompted_intersection"]) for row in rows)
    prompted_union = sum(int(row["_prompted_union"]) for row in rows)
    baseline_ciou = baseline_intersection / max(baseline_union, 1)
    prompted_ciou = prompted_intersection / max(prompted_union, 1)
    improved = sum(delta > 1e-12 for delta in deltas)
    degraded = sum(delta < -1e-12 for delta in deltas)
    base_seconds = float(np.mean([row["base_seconds"] for row in rows]))
    total_seconds = float(np.mean([row["total_seconds"] for row in rows]))
    return {
        "source": "lisa_freeref_prompt_paired",
        "samples": count,
        "baseline_mean_iou": baseline_miou,
        "prompted_mean_iou": prompted_miou,
        "mean_iou_delta": prompted_miou - baseline_miou,
        "baseline_cIoU": baseline_ciou,
        "prompted_cIoU": prompted_ciou,
        "cIoU_delta": prompted_ciou - baseline_ciou,
        "baseline_boundary_iou": float(
            np.mean([row["baseline_boundary_iou"] for row in rows])
        ),
        "prompted_boundary_iou": float(
            np.mean([row["prompted_boundary_iou"] for row in rows])
        ),
        "boundary_iou_delta": float(
            np.mean(
                [row["prompted_boundary_iou"] - row["baseline_boundary_iou"] for row in rows]
            )
        ),
        "improved_samples": improved,
        "degraded_samples": degraded,
        "unchanged_samples": count - improved - degraded,
        "improvement_rate": improved / count,
        "base_seconds_per_sample": base_seconds,
        "freeref_seconds_per_sample": float(
            np.mean([row["freeref_seconds"] for row in rows])
        ),
        "second_decoder_seconds_per_sample": float(
            np.mean([row["second_decoder_seconds"] for row in rows])
        ),
        "total_seconds_per_sample": total_seconds,
        "relative_overhead_vs_base": total_seconds / max(base_seconds, 1e-12) - 1.0,
        **paired_statistics(deltas, bootstrap_samples, seed),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    public_rows = [
        {key: value for key, value in row.items() if not key.startswith("_")} for row in rows
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(public_rows[0]))
        writer.writeheader()
        writer.writerows(public_rows)
    os.replace(temporary, path)


def _markdown(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# LISA FreeRef-Prompt Quick Evaluation",
            "",
            "| Samples | Baseline mIoU | Prompted mIoU | Delta | Baseline cIoU | Prompted cIoU | Delta | Boundary Delta |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
            (
                f"| {summary['samples']} | {summary['baseline_mean_iou'] * 100:.2f} | "
                f"{summary['prompted_mean_iou'] * 100:.2f} | "
                f"{summary['mean_iou_delta'] * 100:+.2f} | "
                f"{summary['baseline_cIoU'] * 100:.2f} | "
                f"{summary['prompted_cIoU'] * 100:.2f} | "
                f"{summary['cIoU_delta'] * 100:+.2f} | "
                f"{summary['boundary_iou_delta'] * 100:+.2f} |"
            ),
            "",
            (
                f"Timing: base {summary['base_seconds_per_sample']:.3f} s/sample, "
                f"FreeRef {summary['freeref_seconds_per_sample']:.3f}, "
                f"second decoder {summary['second_decoder_seconds_per_sample']:.3f}, "
                f"total {summary['total_seconds_per_sample']:.3f}."
            ),
            "",
        ]
    )


def main() -> int:
    args = parse_args()
    if args.boundary_tolerance < 0 or args.bootstrap_samples < 0:
        raise ValueError("boundary-tolerance and bootstrap-samples must be non-negative.")
    manifest_rows = _read_jsonl(args.manifest.expanduser().resolve())
    rows = [
        _evaluate_row(row, args.boundary_tolerance)
        for row in tqdm(manifest_rows, desc="LISA paired metrics", dynamic_ncols=True)
    ]
    summary = summarize_rows(rows, args.bootstrap_samples, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "eval_rows.csv", rows)
    _atomic_write(
        args.output_dir / "eval_summary.json",
        json.dumps(summary, indent=2, ensure_ascii=True),
    )
    _atomic_write(args.output_dir / "comparison.md", _markdown(summary))
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
