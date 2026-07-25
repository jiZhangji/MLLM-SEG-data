from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


SPLITS = (
    ("refcoco_val", 10834),
    ("refcoco_testA", 5657),
    ("refcoco_testB", 5095),
    ("refcoco+_val", 10758),
    ("refcoco+_testA", 5726),
    ("refcoco+_testB", 4889),
    ("refcocog_val", 4896),
    ("refcocog_test", 9602),
)
MODELS = (("stamp7b", "STAMP-7B"), ("text4seg_p24", "Text4Seg-p24"))
METHODS = (
    ("base", "Base"),
    ("densecrf", "DenseCRF"),
    ("guided_filter", "Guided Filter"),
    ("fast_bilateral_solver", "Fast Bilateral Solver"),
    ("slic_average", "SLIC Averaging"),
    ("freeref", "FreeRef"),
)

SPLIT_LABELS = {
    "refcoco_val": "RefCOCO val",
    "refcoco_testA": "RefCOCO testA",
    "refcoco_testB": "RefCOCO testB",
    "refcoco+_val": "RefCOCO+ val",
    "refcoco+_testA": "RefCOCO+ testA",
    "refcoco+_testB": "RefCOCO+ testB",
    "refcocog_val": "RefCOCOg val(U)",
    "refcocog_test": "RefCOCOg test(U)",
}


def split_slug(value: str) -> str:
    return value.replace("+", "plus")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize full eight-split post-processing runs.")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def detailed_metric_table(
    detailed: list[dict[str, object]],
    macro: list[dict[str, object]],
    metric: str,
) -> list[str]:
    split_names = [split for split, _ in SPLITS]
    labels = [SPLIT_LABELS[split] for split in split_names]
    title = {"mIoU": "mIoU", "cIoU": "cIoU", "bIoU": "Boundary IoU (bIoU)"}[metric]
    lines = [
        f"## Per-Split {title} (%)",
        "",
        "| Base output | Refiner | " + " | ".join(labels) + " | Avg. |",
        "|---|---|" + "---:|" * (len(labels) + 1),
    ]
    for _, model_label in MODELS:
        for _, method_label in METHODS:
            selected = {
                str(row["split"]): row
                for row in detailed
                if row["model"] == model_label and row["method"] == method_label
            }
            macro_row = next(
                row
                for row in macro
                if row["model"] == model_label and row["method"] == method_label
            )
            method_cell = f"**{method_label}**" if method_label == "FreeRef" else method_label
            values = [f"{float(selected[split][metric]):.2f}" for split in split_names]
            average = f"{float(macro_row[metric]):.2f}"
            if method_label == "FreeRef":
                values = [f"**{value}**" for value in values]
                average = f"**{average}**"
            lines.append(
                f"| {model_label} | {method_cell} | " + " | ".join(values) + f" | {average} |"
            )
    lines.append("")
    return lines


def main() -> int:
    args = parse_args()
    detailed: list[dict[str, object]] = []
    for model_slug, model_label in MODELS:
        for split, expected in SPLITS:
            path = args.input_root / model_slug / split_slug(split) / "eval_summary.json"
            if not path.is_file():
                raise FileNotFoundError(path)
            summary = json.loads(path.read_text(encoding="utf-8"))
            if int(summary.get("samples", 0)) != expected:
                raise ValueError(f"Incomplete split {path}: {summary.get('samples')} != {expected}")
            values = dict(summary["methods"])
            for method, display in METHODS:
                metrics = dict(values[method])
                detailed.append(
                    {
                        "model": model_label,
                        "split": split,
                        "samples": expected,
                        "method": display,
                        "mIoU": 100.0 * float(metrics["mIoU"]),
                        "delta_mIoU": 100.0 * float(metrics["delta_mIoU"]),
                        "cIoU": 100.0 * float(metrics["cIoU"]),
                        "delta_cIoU": 100.0 * float(metrics["delta_cIoU"]),
                        "bIoU": 100.0 * float(metrics["bIoU"]),
                        "delta_bIoU": 100.0 * float(metrics["delta_bIoU"]),
                        "summary": str(path),
                    }
                )

    macro: list[dict[str, object]] = []
    for _, model_label in MODELS:
        for _, display in METHODS:
            selected = [
                row for row in detailed if row["model"] == model_label and row["method"] == display
            ]
            macro.append(
                {
                    "model": model_label,
                    "method": display,
                    **{
                        key: statistics.fmean(float(row[key]) for row in selected)
                        for key in (
                            "mIoU",
                            "delta_mIoU",
                            "cIoU",
                            "delta_cIoU",
                            "bIoU",
                            "delta_bIoU",
                        )
                    },
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("postprocess_full_splits", detailed), ("postprocess_macro", macro)):
        with (args.output_dir / f"{name}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (args.output_dir / "postprocess_full.json").write_text(
        json.dumps({"split_rows": detailed, "macro_rows": macro}, indent=2), encoding="utf-8"
    )
    lines = [
        "# Full Eight-Split Training-Free Post-processing Comparison",
        "",
        "Values are macro-averaged over the eight complete RefCOCO-family splits.",
        "",
        "| Base output | Refiner | mIoU | Delta mIoU | cIoU | Delta cIoU | **bIoU** | **Delta bIoU** |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in macro:
        lines.append(
            "| {model} | {method} | {mIoU:.2f} | {delta_mIoU:+.2f} | {cIoU:.2f} | "
            "{delta_cIoU:+.2f} | **{bIoU:.2f}** | **{delta_bIoU:+.2f}** |".format(**row)
        )
    lines.extend([""])
    lines.extend(detailed_metric_table(detailed, macro, "mIoU"))
    lines.extend(detailed_metric_table(detailed, macro, "cIoU"))
    lines.extend(detailed_metric_table(detailed, macro, "bIoU"))
    markdown = "\n".join(lines) + "\n"
    (args.output_dir / "postprocess_full.md").write_text(markdown, encoding="utf-8")
    print(markdown, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
