from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


METHODS = (
    ("base", "None", "-", "-", "CPU / threshold"),
    ("densecrf", "DenseCRF", "No", "None", "CPU / pydensecrf"),
    ("guided_filter", "Guided Filter", "No", "None", "CPU / SciPy"),
    ("slic_average", "SLIC Averaging", "No", "None", "CPU / scikit-image"),
    ("sam_h", "SAM-H", "No", "SAM-H", "H100 / SAM-H"),
    ("freeref", "FreeRef", "No", "None", "H100 / CuPy-cuCIM"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize paired post-processing baselines.")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def sam_metrics(summary: dict[str, object]) -> dict[str, float]:
    return {
        "mIoU": float(summary["coarse_sam_mean_iou"]),
        "cIoU": float(summary["coarse_sam_cIoU"]),
        "bIoU": float(summary["coarse_sam_boundary_iou"]),
        "delta_mIoU": float(summary["coarse_sam_mean_iou"])
        - float(summary["coarse_mean_iou"]),
        "delta_cIoU": float(summary["coarse_sam_cIoU"]) - float(summary["coarse_cIoU"]),
        "delta_bIoU": float(summary["coarse_sam_boundary_iou"])
        - float(summary["coarse_boundary_iou"]),
    }


def sam_mean_seconds(summary: dict[str, object]) -> float:
    rows_path = Path(str(summary["rows_csv"]))
    if not rows_path.is_absolute():
        rows_path = rows_path.resolve()
    with rows_path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No SAM timing rows in {rows_path}")
    values = [
        float(row["sam_image_seconds"]) + float(row["sam_prompt_seconds"]) / 2.0
        for row in rows
    ]
    return statistics.fmean(values)


def assert_paired_base(
    model_label: str,
    baseline: dict[str, object],
    sam_summary: dict[str, object],
) -> None:
    pairs = (
        ("mIoU", "coarse_mean_iou"),
        ("cIoU", "coarse_cIoU"),
        ("bIoU", "coarse_boundary_iou"),
    )
    for baseline_key, sam_key in pairs:
        gap = abs(float(baseline[baseline_key]) - float(sam_summary[sam_key]))
        if gap > 1e-6:
            raise ValueError(
                f"{model_label} is not paired: Base {baseline_key} differs from the "
                f"SAM input branch by {gap:.8f}."
            )


def main() -> int:
    args = parse_args()
    root = args.input_root
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_specs = (
        ("stamp7b", "STAMP-7B"),
        ("text4seg_p24", "Text4Seg-p24"),
    )
    all_rows: list[dict[str, object]] = []
    for slug, model_label in model_specs:
        accuracy = load_json(root / "accuracy" / slug / "baselines" / "eval_summary.json")
        timing = load_json(root / "timing" / slug / "baselines" / "eval_summary.json")
        sam_accuracy_summary = load_json(root / "accuracy" / slug / "sam_h" / "eval_summary.json")
        sam_timing_summary = load_json(root / "timing" / slug / "sam_h" / "eval_summary.json")
        if int(accuracy["samples"]) != int(timing["samples"]):
            raise ValueError(f"Accuracy/timing sample mismatch for {model_label}")
        accuracy_methods = dict(accuracy["methods"])
        timing_methods = dict(timing["methods"])
        assert_paired_base(model_label, dict(accuracy_methods["base"]), sam_accuracy_summary)
        for key, display, training, external, backend in METHODS:
            metrics = sam_metrics(sam_accuracy_summary) if key == "sam_h" else dict(accuracy_methods[key])
            if key == "sam_h":
                seconds = sam_mean_seconds(sam_timing_summary)
            else:
                seconds = float(timing_methods[key]["mean_seconds"])
            throughput = 1.0 / seconds if seconds > 1e-8 else None
            all_rows.append(
                {
                    "model": model_label,
                    "refiner": display,
                    "training": training,
                    "external_model": external,
                    "mIoU": 100.0 * float(metrics["mIoU"]),
                    "delta_mIoU": 100.0 * float(metrics["delta_mIoU"]),
                    "cIoU": 100.0 * float(metrics["cIoU"]),
                    "delta_cIoU": 100.0 * float(metrics["delta_cIoU"]),
                    "bIoU": 100.0 * float(metrics["bIoU"]),
                    "delta_bIoU": 100.0 * float(metrics["delta_bIoU"]),
                    "seconds_per_sample": seconds,
                    "throughput_samples_per_second": throughput,
                    "backend": backend,
                }
            )

    csv_path = args.output_dir / "postprocess_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)

    lines = [
        "# Training-Free Post-processing Comparison (N=500)",
        "",
        "Accuracy uses paired saved outputs on RefCOCO testA. Runtime is measured in a separate "
        "serial stage with image/model-output loading excluded.",
        "",
        "| Model | Refiner | Training | External model | mIoU | Delta mIoU | cIoU | Delta cIoU | bIoU | Delta bIoU | Latency (s/sample) | Throughput (samples/s) | Backend |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in all_rows:
        throughput = row["throughput_samples_per_second"]
        throughput_text = "-" if throughput is None else f"{float(throughput):.2f}"
        lines.append(
            "| {model} | {refiner} | {training} | {external_model} | {mIoU:.2f} | "
            "{delta_mIoU:+.2f} | {cIoU:.2f} | {delta_cIoU:+.2f} | {bIoU:.2f} | "
            "{delta_bIoU:+.2f} | {seconds_per_sample:.4f} | {throughput} | "
            "{backend} |".format(throughput=throughput_text, **row)
        )
    lines.extend(
        [
            "",
            "Text4Seg hard masks are mapped to foreground/background probabilities 0.95/0.05 for "
            "DenseCRF, Guided Filter, and SLIC Averaging. FreeRef accuracy uses the reference CPU "
            "implementation; its reported runtime uses the metric-equivalent GPU implementation. "
            "SAM-H timing is image encoding plus one of the two paired prompt branches.",
            "",
        ]
    )
    markdown = "\n".join(lines)
    (args.output_dir / "postprocess_comparison.md").write_text(markdown, encoding="utf-8")
    (args.output_dir / "postprocess_comparison.json").write_text(
        json.dumps({"rows": all_rows}, indent=2), encoding="utf-8"
    )
    print(markdown, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
