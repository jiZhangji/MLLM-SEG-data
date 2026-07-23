from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


METHOD_ORDER = ("DenseCRF", "Guided Filter", "SLIC Averaging", "SAM-H", "FreeRef")
METHOD_COLORS = {
    "DenseCRF": "#9a9a9a",
    "Guided Filter": "#2a9d8f",
    "SLIC Averaging": "#e9c46a",
    "SAM-H": "#d1495b",
    "FreeRef": "#277da1",
}
ANNOTATION_OFFSETS = {
    "DenseCRF": (7, -16),
    "Guided Filter": (7, 9),
    "SLIC Averaging": (-78, 9),
    "SAM-H": (7, 10),
    "FreeRef": (7, 9),
}
METRIC_SPECS = (
    ("delta_mIoU", r"$\Delta$mIoU", "#2a9d8f"),
    ("delta_cIoU", r"$\Delta$cIoU", "#277da1"),
    ("delta_bIoU", r"$\Delta$bIoU", "#e07a3f"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot paired post-processing baseline results.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, object]]:
    numeric = (
        "mIoU",
        "delta_mIoU",
        "cIoU",
        "delta_cIoU",
        "bIoU",
        "delta_bIoU",
        "seconds_per_sample",
    )
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    converted: list[dict[str, object]] = []
    for row in rows:
        value: dict[str, object] = dict(row)
        for key in numeric:
            value[key] = float(row[key])
        converted.append(value)
    return converted


def _model_rows(rows: list[dict[str, object]], model: str) -> list[dict[str, object]]:
    by_method = {str(row["refiner"]): row for row in rows if row["model"] == model}
    missing = [method for method in METHOD_ORDER if method not in by_method]
    if missing:
        raise ValueError(f"{model} is missing methods: {missing}")
    return [by_method[method] for method in METHOD_ORDER]


def save_accuracy_gain(rows: list[dict[str, object]], output_dir: Path, dpi: int) -> None:
    import matplotlib.pyplot as plt

    models = tuple(dict.fromkeys(str(row["model"]) for row in rows))
    fig, axes = plt.subplots(1, len(models), figsize=(12.6, 4.2), sharey=False)
    axes = np.atleast_1d(axes)
    x = np.arange(len(METHOD_ORDER), dtype=np.float64)
    width = 0.24
    for axis, model in zip(axes, models):
        values = _model_rows(rows, model)
        for metric_index, (key, label, color) in enumerate(METRIC_SPECS):
            offsets = x + (metric_index - 1) * width
            axis.bar(
                offsets,
                [float(row[key]) for row in values],
                width=width,
                label=label,
                color=color,
                edgecolor="white",
                linewidth=0.5,
            )
        axis.axhline(0.0, color="#222222", linewidth=0.8)
        axis.set_title(model, fontsize=11, fontweight="bold")
        axis.set_xticks(x)
        axis.set_xticklabels(METHOD_ORDER, rotation=18, ha="right")
        axis.set_ylabel("Gain over base (percentage points)")
        axis.grid(axis="y", color="#dddddd", linewidth=0.6)
        axis.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=3,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"postprocess_accuracy_gains.{suffix}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_accuracy_latency(rows: list[dict[str, object]], output_dir: Path, dpi: int) -> None:
    import matplotlib.pyplot as plt

    models = tuple(dict.fromkeys(str(row["model"]) for row in rows))
    fig, axes = plt.subplots(1, len(models), figsize=(12.6, 4.2), sharex=True)
    axes = np.atleast_1d(axes)
    for axis, model in zip(axes, models):
        values = _model_rows(rows, model)
        for row in values:
            method = str(row["refiner"])
            marker = "*" if method == "SAM-H" else ("D" if method == "FreeRef" else "o")
            size = 150 if method == "SAM-H" else (75 if method == "FreeRef" else 55)
            x = float(row["seconds_per_sample"])
            y = float(row["delta_cIoU"])
            axis.scatter(
                x,
                y,
                s=size,
                marker=marker,
                color=METHOD_COLORS[method],
                edgecolor="white",
                linewidth=0.7,
                zorder=3,
            )
            axis.annotate(
                method,
                (x, y),
                xytext=ANNOTATION_OFFSETS[method],
                textcoords="offset points",
                fontsize=8,
                bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "none", "alpha": 0.9},
            )
        axis.axhline(0.0, color="#222222", linewidth=0.8)
        axis.set_title(model, fontsize=11, fontweight="bold")
        axis.set_xlabel("Post-processing time (seconds/sample)")
        axis.set_ylabel(r"$\Delta$cIoU (percentage points)")
        axis.grid(color="#dddddd", linewidth=0.6)
        axis.set_axisbelow(True)
        axis.margins(x=0.16, y=0.22)
    fig.suptitle("Accuracy-latency trade-off (method-native backend)", y=1.02, fontsize=12)
    fig.text(
        0.5,
        -0.01,
        "CPU: DenseCRF, Guided Filter, SLIC Averaging; H100: SAM-H and FreeRef.",
        ha="center",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"postprocess_accuracy_latency.{suffix}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    if args.dpi <= 0:
        raise ValueError("dpi must be positive.")
    rows = load_rows(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_accuracy_gain(rows, args.output_dir, args.dpi)
    save_accuracy_latency(rows, args.output_dir, args.dpi)
    for path in sorted(args.output_dir.glob("postprocess_*.*")):
        print(path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
