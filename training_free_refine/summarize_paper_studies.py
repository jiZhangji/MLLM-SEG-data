from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize FreeRef paper ablations and sweeps.")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for path in sorted(args.input_root.glob("*/*/*/eval_summary.json")):
        relative = path.relative_to(args.input_root)
        model, study, setting = relative.parts[:3]
        value = json.loads(path.read_text(encoding="utf-8"))
        config = value.get("config", {})
        rows.append(
            {
                "model": model,
                "study": study,
                "setting": setting,
                "samples": int(value["samples"]),
                "coarse_mIoU_pct": 100.0 * float(value["coarse_mean_iou"]),
                "refined_mIoU_pct": 100.0 * float(value["refined_mean_iou"]),
                "delta_mIoU_points": 100.0 * float(value["mean_iou_delta"]),
                "coarse_cIoU_pct": 100.0 * float(value["coarse_cIoU"]),
                "refined_cIoU_pct": 100.0 * float(value["refined_cIoU"]),
                "delta_cIoU_points": 100.0 * float(value["cIoU_delta"]),
                "delta_boundary_iou_points": 100.0 * float(value["boundary_iou_delta"]),
                "seconds_per_sample": float(value["seconds_per_sample"]),
                "n_segments": config.get("n_segments"),
                "graph_lambda": config.get("graph_lambda"),
                "confidence_power": config.get("confidence_power"),
                "boundary_sigma": value.get("boundary_sigma"),
                "uncertainty_aware_anchoring": config.get("uncertainty_aware_anchoring"),
                "appearance_weighted_graph": config.get("appearance_weighted_graph"),
                "selective_fusion": config.get("selective_fusion"),
                "summary": str(path),
            }
        )
    if not rows:
        raise FileNotFoundError(f"No study summaries found below {args.input_root}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "paper_studies.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "paper_studies.json").write_text(
        json.dumps({"rows": rows}, indent=2), encoding="utf-8"
    )

    lines = [
        "# FreeRef Paper Ablations and Sensitivity Studies",
        "",
        "All values are percentage points on a fixed paired subset.",
        "",
        "| Model | Study | Setting | N | Refined mIoU | Delta mIoU | Refined cIoU | Delta cIoU | Delta bIoU | Sec/sample |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {model} | {study} | {setting} | {samples} | {refined_mIoU_pct:.2f} | "
            "{delta_mIoU_points:+.2f} | {refined_cIoU_pct:.2f} | "
            "{delta_cIoU_points:+.2f} | {delta_boundary_iou_points:+.2f} | "
            "{seconds_per_sample:.4f} |".format(**row)
        )
    markdown = "\n".join(lines) + "\n"
    (args.output_dir / "paper_studies.md").write_text(markdown, encoding="utf-8")
    print(markdown, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
