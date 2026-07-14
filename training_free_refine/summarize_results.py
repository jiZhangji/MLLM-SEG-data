from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine training-free refinement val/test summaries.")
    parser.add_argument("--val-summary", type=Path, required=True)
    parser.add_argument("--test-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_summary(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}.")
    return value


def result_row(split: str, summary: dict[str, object]) -> dict[str, object]:
    return {
        "split": split,
        "samples": int(summary["samples"]),
        "coarse_mean_iou": float(summary["coarse_mean_iou"]),
        "refined_mean_iou": float(summary["refined_mean_iou"]),
        "mean_iou_delta": float(summary["mean_iou_delta"]),
        "coarse_cIoU": float(summary["coarse_cIoU"]),
        "refined_cIoU": float(summary["refined_cIoU"]),
        "cIoU_delta": float(summary["cIoU_delta"]),
        "coarse_boundary_iou": float(summary["coarse_boundary_iou"]),
        "refined_boundary_iou": float(summary["refined_boundary_iou"]),
        "boundary_iou_delta": float(summary["boundary_iou_delta"]),
        "improved_samples": int(summary["improved_samples"]),
        "degraded_samples": int(summary["degraded_samples"]),
        "seconds_per_sample": float(summary["seconds_per_sample"]),
    }


def markdown_table(rows: list[dict[str, object]]) -> str:
    lines = [
        "# Training-Free Refinement Full Evaluation",
        "",
        "| Split | Samples | Coarse mIoU | Refined mIoU | Delta | Coarse cIoU | Refined cIoU | Delta | Boundary delta | Improved | Degraded | Sec/sample |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {split} | {samples} | {coarse_mean_iou:.4f} | {refined_mean_iou:.4f} | "
            "{mean_iou_delta:+.4f} | {coarse_cIoU:.4f} | {refined_cIoU:.4f} | "
            "{cIoU_delta:+.4f} | {boundary_iou_delta:+.4f} | {improved_samples} | "
            "{degraded_samples} | {seconds_per_sample:.3f} |".format(**row)
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    rows = [
        result_row("val", load_summary(args.val_summary)),
        result_row("test", load_summary(args.test_summary)),
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    combined = {
        "method": "training_free_uncertainty_graph_refinement",
        "splits": rows,
        "val_summary": str(args.val_summary),
        "test_summary": str(args.test_summary),
    }
    json_path = args.output_dir / "combined_summary.json"
    markdown_path = args.output_dir / "combined_summary.md"
    json_path.write_text(json.dumps(combined, indent=2, ensure_ascii=True), encoding="utf-8")
    table = markdown_table(rows)
    markdown_path.write_text(table, encoding="utf-8")
    print(table, end="")
    print(f"Combined JSON: {json_path}")
    print(f"Combined Markdown: {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
