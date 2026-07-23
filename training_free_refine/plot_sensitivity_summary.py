from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


STUDIES = (
    ("confidence_power", r"Confidence power $\gamma$", "#2a9d8f"),
    ("graph_regularization", r"Graph regularization $\lambda$", "#277da1"),
    ("superpixels", r"Superpixels $K$", "#e07a3f"),
    ("boundary_uncertainty", r"Boundary uncertainty $\sigma$", "#d1495b"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot the FreeRef sensitivity summary.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        source = list(csv.DictReader(handle))
    if not source:
        raise ValueError(f"No sensitivity rows found in {path}")
    rows: list[dict[str, object]] = []
    for row in source:
        rows.append(
            {
                **row,
                "delta_ciou": float(row["delta_ciou"]),
                "is_default": str(row["is_default"]).strip().lower() in {"1", "true", "yes"},
            }
        )
    return rows


def save_plot(rows: list[dict[str, object]], output_dir: Path, dpi: int) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11.6, 7.2), constrained_layout=True)
    for axis, (study, title, color) in zip(axes.reshape(-1), STUDIES):
        values = [row for row in rows if row["study"] == study]
        if not values:
            raise ValueError(f"Missing sensitivity study: {study}")
        x = np.arange(len(values), dtype=np.float64)
        y = np.asarray([float(row["delta_ciou"]) for row in values])
        labels = [str(row["value"]) for row in values]
        axis.plot(x, y, color=color, linewidth=2.0, marker="o", markersize=5.5)
        default_indices = [index for index, row in enumerate(values) if bool(row["is_default"])]
        if len(default_indices) != 1:
            raise ValueError(f"{study} must contain exactly one default point")
        default_index = default_indices[0]
        axis.scatter(
            [x[default_index]],
            [y[default_index]],
            marker="*",
            s=190,
            color="#f4b400",
            edgecolor="#222222",
            linewidth=0.7,
            zorder=4,
        )
        axis.annotate(
            "Default",
            (x[default_index], y[default_index]),
            xytext=(8, 10),
            textcoords="offset points",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.9},
        )
        axis.axhline(0.0, color="#333333", linewidth=0.8, linestyle="--")
        axis.set_title(f"{title}\n{values[0]['model_input']}", fontsize=10, fontweight="bold")
        axis.set_xticks(x)
        rotation = 28 if len(labels) > 5 else 0
        axis.set_xticklabels(labels, rotation=rotation, ha="right" if rotation else "center")
        axis.set_xlabel(str(values[0]["parameter"]))
        axis.set_ylabel(r"$\Delta$cIoU (percentage points)")
        axis.grid(axis="y", color="#dddddd", linewidth=0.6)
        axis.set_axisbelow(True)
        margin = max(0.12, 0.12 * max(float(y.max() - y.min()), 0.5))
        axis.set_ylim(min(0.0, float(y.min()) - margin), float(y.max()) + 2.0 * margin)
    fig.suptitle("FreeRef hyperparameter sensitivity", fontsize=13, fontweight="bold")
    for suffix in ("png", "pdf"):
        fig.savefig(
            output_dir / f"freeref_hyperparameter_sensitivity.{suffix}",
            dpi=dpi,
            bbox_inches="tight",
        )
    plt.close(fig)


def main() -> int:
    args = parse_args()
    if args.dpi <= 0:
        raise ValueError("dpi must be positive.")
    rows = load_rows(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_plot(rows, args.output_dir, args.dpi)
    for path in sorted(args.output_dir.glob("freeref_hyperparameter_sensitivity.*")):
        print(path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
