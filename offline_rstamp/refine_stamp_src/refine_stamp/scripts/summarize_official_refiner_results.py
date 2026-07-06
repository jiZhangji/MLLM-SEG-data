from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_summary(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["summary"]


def fmt(value: float | int | str | None) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--train-limit", type=int, default=1000)
    parser.add_argument("--selectors", nargs="+", default=["random", "uncertainty", "boundary", "hybrid"])
    parser.add_argument("--splits", nargs="+", default=["refcocog_val", "refcocog_test"])
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    rows = []
    for split in args.splits:
        for selector in args.selectors:
            path = (
                args.root
                / "outputs"
                / "refine_stamp_refiner_eval"
                / f"official_{split}_train{args.train_limit}_{selector}"
                / "refiner_eval_summary.json"
            )
            if not path.exists():
                rows.append({"split": split, "selector": selector, "missing": str(path)})
                continue
            summary = load_summary(path)
            rows.append({"split": split, "selector": selector, **summary})

    lines = [
        f"# Refine-STAMP Official Comparison Summary (train={args.train_limit})",
        "",
        "| Split | Selector | Samples | Coarse mean IoU | Refined mean IoU | mean IoU Delta | Coarse cIoU | Refined cIoU | cIoU Delta | Coarse Boundary IoU | Refined Boundary IoU | Boundary Delta |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if "missing" in row:
            lines.append(
                f"| {row['split']} | {row['selector']} | MISSING | `{row['missing']}` |  |  |  |  |  |  |  |  |"
            )
            continue
        lines.append(
            "| {split} | {selector} | {samples} | {coarse_mean_iou} | {refined_mean_iou} | {mean_iou_delta} | "
            "{coarse_cIoU} | {refined_cIoU} | {cIoU_delta} | {coarse_boundary_iou} | {refined_boundary_iou} | {boundary_iou_delta} |".format(
                split=row["split"],
                selector=row["selector"],
                samples=fmt(row.get("samples")),
                coarse_mean_iou=fmt(row.get("coarse_mean_iou")),
                refined_mean_iou=fmt(row.get("refined_mean_iou")),
                mean_iou_delta=fmt(row.get("mean_iou_delta")),
                coarse_cIoU=fmt(row.get("coarse_cIoU")),
                refined_cIoU=fmt(row.get("refined_cIoU")),
                cIoU_delta=fmt(row.get("cIoU_delta")),
                coarse_boundary_iou=fmt(row.get("coarse_boundary_iou")),
                refined_boundary_iou=fmt(row.get("refined_boundary_iou")),
                boundary_iou_delta=fmt(row.get("boundary_iou_delta")),
            )
        )

    lines.extend(
        [
            "",
            "Notes:",
            "",
            "- `Coarse` is the STAMP mask reconstructed from exported Phase-2 logits.",
            "- `Refined` is STAMP plus the local high-resolution refiner.",
            "- For paper comparison, focus first on official `refcocog_val` and `refcocog_test` cIoU and Boundary IoU deltas.",
        ]
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
