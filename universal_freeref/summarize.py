from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine universal FreeRef paired evaluation summaries.")
    parser.add_argument("--summary", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--title", default="Universal FreeRef Paired Evaluation")
    return parser.parse_args()


def _load(specification: str) -> dict[str, Any]:
    if "=" not in specification:
        raise ValueError(f"Expected LABEL=PATH, got {specification!r}.")
    label, raw_path = specification.split("=", 1)
    path = Path(raw_path)
    value = json.loads(path.read_text(encoding="utf-8"))
    return {
        "label": label,
        "samples": int(value["samples"]),
        "coarse_mean_iou": float(value["coarse_mean_iou"]),
        "refined_mean_iou": float(value["refined_mean_iou"]),
        "mean_iou_delta": float(value["mean_iou_delta"]),
        "coarse_cIoU": float(value["coarse_cIoU"]),
        "refined_cIoU": float(value["refined_cIoU"]),
        "cIoU_delta": float(value["cIoU_delta"]),
        "boundary_iou_delta": float(value["boundary_iou_delta"]),
        "improved_samples": int(value["improved_samples"]),
        "degraded_samples": int(value["degraded_samples"]),
        "seconds_per_sample": float(value["seconds_per_sample"]),
        "source": str(path.resolve()),
    }


def main() -> int:
    args = parse_args()
    rows = [_load(specification) for specification in args.summary]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_path = args.output_dir / "comparison.json"
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=True), encoding="utf-8")
    lines = [
        f"# {args.title}",
        "",
        "| Method / Split | Samples | Original mIoU | FreeRef mIoU | Delta | Original cIoU | FreeRef cIoU | Delta | Boundary delta | Improved | Degraded | Sec/sample |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['label']} | {row['samples']} | {row['coarse_mean_iou']:.4f} | "
            f"{row['refined_mean_iou']:.4f} | {row['mean_iou_delta']:+.4f} | "
            f"{row['coarse_cIoU']:.4f} | {row['refined_cIoU']:.4f} | "
            f"{row['cIoU_delta']:+.4f} | {row['boundary_iou_delta']:+.4f} | "
            f"{row['improved_samples']} | {row['degraded_samples']} | {row['seconds_per_sample']:.3f} |"
        )
    markdown = "\n".join(lines) + "\n"
    (args.output_dir / "comparison.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
