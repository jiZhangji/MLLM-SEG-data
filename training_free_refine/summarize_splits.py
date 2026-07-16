from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_summary_spec(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Summary must use SPLIT=/path/to/eval_summary.json")
    label, raw_path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("Summary split label cannot be empty.")
    return label, Path(raw_path).expanduser()


def load_row(label: str, path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return {
        "split": label,
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
        "unchanged_samples": int(value["unchanged_samples"]),
        "seconds_per_sample": float(value["seconds_per_sample"]),
        "summary": str(path),
    }


def markdown_table(title: str, rows: list[dict[str, object]]) -> str:
    lines = [
        f"# {title}",
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
    parser = argparse.ArgumentParser(description="Combine any number of Training-Free split summaries.")
    parser.add_argument("--summary", type=parse_summary_spec, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--title", default="Training-Free Multi-Split Evaluation")
    args = parser.parse_args()
    rows = [load_row(label, path.resolve()) for label, path in args.summary]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    combined = {"method": "training_free_uncertainty_graph_refinement", "splits": rows}
    json_path = args.output_dir / "combined_summary.json"
    markdown_path = args.output_dir / "combined_summary.md"
    json_path.write_text(json.dumps(combined, indent=2, ensure_ascii=True), encoding="utf-8")
    table = markdown_table(args.title, rows)
    markdown_path.write_text(table, encoding="utf-8")
    print(table, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
