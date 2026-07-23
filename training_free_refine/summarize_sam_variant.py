from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


BRANCHES = (
    ("coarse", "Base"),
    ("freeref", "FreeRef"),
    ("coarse_sam", "SAM"),
    ("freeref_sam", "FreeRef -> SAM"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize a paired frozen-SAM variant run.")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def mean_timings(summary: dict[str, object]) -> dict[str, float]:
    with Path(str(summary["rows_csv"])).open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("No timing rows were found.")
    freeref = [float(row["freeref_seconds"]) for row in rows]
    sam = [
        float(row["sam_image_seconds"]) + float(row["sam_prompt_seconds"]) / 2.0
        for row in rows
    ]
    return {
        "coarse": 0.0,
        "freeref": statistics.fmean(freeref),
        "coarse_sam": statistics.fmean(sam),
        "freeref_sam": statistics.fmean(
            free_seconds + sam_seconds
            for free_seconds, sam_seconds in zip(freeref, sam)
        ),
    }


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for slug, model_label in (("stamp7b", "STAMP-7B"), ("text4seg_p24", "Text4Seg-p24")):
        accuracy = load_json(args.input_root / "accuracy" / slug / "eval_summary.json")
        timing = load_json(args.input_root / "timing" / slug / "eval_summary.json")
        if int(accuracy["samples"]) != int(timing["samples"]):
            raise ValueError(f"Accuracy/timing sample mismatch for {model_label}")
        if accuracy.get("sam_model_type") != timing.get("sam_model_type"):
            raise ValueError(f"Accuracy/timing SAM model mismatch for {model_label}")
        latency = mean_timings(timing)
        base = {
            "mIoU": float(accuracy["coarse_mean_iou"]),
            "cIoU": float(accuracy["coarse_cIoU"]),
            "bIoU": float(accuracy["coarse_boundary_iou"]),
        }
        for branch, branch_label in BRANCHES:
            seconds = latency[branch]
            values = {
                "mIoU": float(accuracy[f"{branch}_mean_iou"]),
                "cIoU": float(accuracy[f"{branch}_cIoU"]),
                "bIoU": float(accuracy[f"{branch}_boundary_iou"]),
            }
            rows.append(
                {
                    "model": model_label,
                    "sam_model_type": accuracy["sam_model_type"],
                    "branch": branch_label,
                    "mIoU": 100.0 * values["mIoU"],
                    "delta_mIoU": 100.0 * (values["mIoU"] - base["mIoU"]),
                    "cIoU": 100.0 * values["cIoU"],
                    "delta_cIoU": 100.0 * (values["cIoU"] - base["cIoU"]),
                    "bIoU": 100.0 * values["bIoU"],
                    "delta_bIoU": 100.0 * (values["bIoU"] - base["bIoU"]),
                    "latency_seconds": seconds,
                    "throughput_samples_per_second": 1.0 / seconds if seconds > 1e-8 else None,
                }
            )

    csv_path = args.output_dir / "sam_variant_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Frozen SAM Variant Comparison (N=500)",
        "",
        "| Model | SAM type | Branch | mIoU | Delta mIoU | cIoU | Delta cIoU | bIoU | Delta bIoU | Latency (s/sample) | Throughput (samples/s) |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        throughput = row["throughput_samples_per_second"]
        throughput_text = "-" if throughput is None else f"{float(throughput):.2f}"
        lines.append(
            "| {model} | {sam_model_type} | {branch} | {mIoU:.2f} | {delta_mIoU:+.2f} | "
            "{cIoU:.2f} | {delta_cIoU:+.2f} | {bIoU:.2f} | {delta_bIoU:+.2f} | "
            "{latency_seconds:.4f} | {throughput} |".format(
                throughput=throughput_text, **row
            )
        )
    markdown = "\n".join(lines) + "\n"
    (args.output_dir / "sam_variant_comparison.md").write_text(markdown, encoding="utf-8")
    (args.output_dir / "sam_variant_comparison.json").write_text(
        json.dumps({"rows": rows}, indent=2), encoding="utf-8"
    )
    print(markdown, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
