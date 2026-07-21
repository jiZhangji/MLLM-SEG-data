from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and summarize paired PolyFormer + FreeRef metrics.")
    parser.add_argument("--export-summary", type=Path, required=True)
    parser.add_argument("--freeref-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paper-miou", type=float)
    parser.add_argument("--consistency-tolerance", type=float, default=1e-6)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    export = json.loads(args.export_summary.read_text(encoding="utf-8"))
    refined = json.loads(args.freeref_summary.read_text(encoding="utf-8"))
    if int(export["samples"]) != int(refined["samples"]):
        raise RuntimeError("PolyFormer export and FreeRef summaries contain different sample counts.")
    baseline_gap = abs(float(export["coarse_mean_iou"]) - float(refined["coarse_mean_iou"]))
    if baseline_gap > args.consistency_tolerance:
        raise RuntimeError(
            f"Rasterized baseline mismatch: export vs FreeRef differs by {baseline_gap:.8f}."
        )

    baseline_miou = float(refined["coarse_mean_iou"]) * 100.0
    freeref_miou = float(refined["refined_mean_iou"]) * 100.0
    baseline_ciou = float(refined["coarse_cIoU"]) * 100.0
    freeref_ciou = float(refined["refined_cIoU"]) * 100.0
    boundary_before = float(refined["coarse_boundary_iou"]) * 100.0
    boundary_after = float(refined["refined_boundary_iou"]) * 100.0
    lines = [
        "# PolyFormer-L + FreeRef Paired Evaluation",
        "",
        f"Protocol: official REFER TSV and released PolyFormer-L polygon rasterization; N={refined['samples']}.",
        "",
        "| Samples | Baseline mIoU | FreeRef mIoU | Delta | Baseline cIoU | FreeRef cIoU | Delta | Boundary delta |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| {refined['samples']} | {baseline_miou:.2f} | {freeref_miou:.2f} | "
            f"{freeref_miou - baseline_miou:+.2f} | {baseline_ciou:.2f} | {freeref_ciou:.2f} | "
            f"{freeref_ciou - baseline_ciou:+.2f} | {boundary_after - boundary_before:+.2f} |"
        ),
        "",
        f"Improved/degraded/unchanged: {refined['improved_samples']}/{refined['degraded_samples']}/{refined['unchanged_samples']}.",
        f"Paired mIoU 95% CI: [{float(refined['delta_ci95_low']) * 100:+.2f}, {float(refined['delta_ci95_high']) * 100:+.2f}] points.",
        f"Wilcoxon p-value: {refined['wilcoxon_p']}.",
    ]
    if args.paper_miou is not None:
        lines.extend(
            [
                "",
                f"Paper full-split mIoU reference: {args.paper_miou:.2f}. This is descriptive only for a subset run.",
            ]
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
