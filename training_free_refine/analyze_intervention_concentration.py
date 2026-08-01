from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt
from tqdm import tqdm

from .visualize_comparison import (
    add_refiner_arguments,
    build_refiner,
    load_view,
    read_csv,
)


CONFIDENCE_EDGES = np.asarray([0.0, 0.2, 0.4, 0.6, 0.8, 1.000001])
DISTANCE_EDGES = np.asarray([0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, np.inf])
GT_BOUNDARY_RADII = (1, 2, 3, 5, 10)


def interval_labels(edges: np.ndarray, *, integer: bool = False) -> list[str]:
    labels = []
    for left, right in zip(edges[:-1], edges[1:]):
        left_text = f"{left:.0f}" if integer else f"{left:.1f}"
        if np.isinf(right):
            labels.append(f">={left_text}")
        else:
            right_text = f"{right:.0f}" if integer else f"{right:.1f}"
            labels.append(f"[{left_text},{right_text})")
    return labels


@dataclass
class BinnedAccumulator:
    edges: np.ndarray
    labels: list[str]

    def __post_init__(self) -> None:
        count = len(self.edges) - 1
        self.pixel_count = np.zeros(count, dtype=np.int64)
        self.abs_change_sum = np.zeros(count, dtype=np.float64)
        self.flip_count = np.zeros(count, dtype=np.int64)

    def update(self, values: np.ndarray, abs_change: np.ndarray, flips: np.ndarray) -> None:
        indices = np.searchsorted(self.edges, values.reshape(-1), side="right") - 1
        indices = np.clip(indices, 0, len(self.pixel_count) - 1)
        self.pixel_count += np.bincount(indices, minlength=len(self.pixel_count))
        self.abs_change_sum += np.bincount(
            indices,
            weights=abs_change.reshape(-1),
            minlength=len(self.pixel_count),
        )
        self.flip_count += np.bincount(
            indices,
            weights=flips.reshape(-1).astype(np.float64),
            minlength=len(self.pixel_count),
        ).astype(np.int64)

    def rows(self, axis: str) -> list[dict[str, Any]]:
        rows = []
        for index, label in enumerate(self.labels):
            count = int(self.pixel_count[index])
            rows.append(
                {
                    "axis": axis,
                    "bin_index": index,
                    "bin": label,
                    "pixel_count": count,
                    "pixel_fraction": count / max(int(self.pixel_count.sum()), 1),
                    "mean_abs_probability_change": self.abs_change_sum[index] / max(count, 1),
                    "label_flip_rate": self.flip_count[index] / max(count, 1),
                    "flip_count": int(self.flip_count[index]),
                }
            )
        return rows


class InterventionAccumulator:
    def __init__(self, threshold: float, boundary_radii: tuple[int, ...]) -> None:
        self.threshold = threshold
        self.boundary_radii = boundary_radii
        self.confidence = BinnedAccumulator(
            CONFIDENCE_EDGES,
            interval_labels(CONFIDENCE_EDGES),
        )
        self.distance = BinnedAccumulator(
            DISTANCE_EDGES,
            interval_labels(DISTANCE_EDGES, integer=True),
        )
        self.samples = 0
        self.total_pixels = 0
        self.total_flips = 0
        self.total_abs_change = 0.0
        self.total_abs_change_mass = 0.0
        self.boundary_band_pixels = {radius: 0 for radius in boundary_radii}
        self.flips_in_boundary_band = {radius: 0 for radius in boundary_radii}
        self.abs_change_in_boundary_band = {radius: 0.0 for radius in boundary_radii}

    def update(
        self,
        coarse_probability: np.ndarray,
        refined_probability: np.ndarray,
        target: np.ndarray,
    ) -> None:
        coarse = np.asarray(coarse_probability, dtype=np.float64)
        refined = np.asarray(refined_probability, dtype=np.float64)
        target_mask = np.asarray(target, dtype=bool)
        if coarse.shape != refined.shape or coarse.shape != target_mask.shape:
            raise ValueError("coarse, refined, and target must have the same shape")
        if not np.isfinite(coarse).all() or not np.isfinite(refined).all():
            raise ValueError("probability maps contain non-finite values")

        coarse = np.clip(coarse, 0.0, 1.0)
        refined = np.clip(refined, 0.0, 1.0)
        abs_change = np.abs(refined - coarse)
        coarse_mask = coarse >= self.threshold
        refined_mask = refined >= self.threshold
        flips = coarse_mask != refined_mask
        confidence = np.abs(2.0 * coarse - 1.0)

        coarse_boundary = np.logical_xor(coarse_mask, binary_erosion(coarse_mask))
        if coarse_boundary.any():
            distance = distance_transform_edt(~coarse_boundary)
        else:
            distance = np.full(coarse.shape, np.inf, dtype=np.float64)

        self.confidence.update(confidence, abs_change, flips)
        self.distance.update(distance, abs_change, flips)
        self.samples += 1
        self.total_pixels += int(coarse.size)
        self.total_flips += int(flips.sum())
        self.total_abs_change += float(abs_change.sum())
        self.total_abs_change_mass += float(abs_change.sum())

        gt_boundary = np.logical_xor(target_mask, binary_erosion(target_mask))
        for radius in self.boundary_radii:
            band = binary_dilation(gt_boundary, iterations=radius) if gt_boundary.any() else gt_boundary
            self.boundary_band_pixels[radius] += int(band.sum())
            self.flips_in_boundary_band[radius] += int(np.logical_and(flips, band).sum())
            self.abs_change_in_boundary_band[radius] += float(abs_change[band].sum())

    def boundary_rows(self) -> list[dict[str, Any]]:
        rows = []
        for radius in self.boundary_radii:
            band_pixels = self.boundary_band_pixels[radius]
            flips_inside = self.flips_in_boundary_band[radius]
            area_fraction = band_pixels / max(self.total_pixels, 1)
            change_share = flips_inside / max(self.total_flips, 1)
            outside_pixels = self.total_pixels - band_pixels
            flips_outside = self.total_flips - flips_inside
            rows.append(
                {
                    "radius_pixels": radius,
                    "boundary_band_pixel_fraction": area_fraction,
                    "changed_pixels_in_band": flips_inside,
                    "changed_pixel_share_in_band": change_share,
                    "change_enrichment": change_share / max(area_fraction, 1e-12),
                    "flip_rate_inside_band": flips_inside / max(band_pixels, 1),
                    "flip_rate_outside_band": flips_outside / max(outside_pixels, 1),
                    "abs_change_mass_share_in_band": self.abs_change_in_boundary_band[radius]
                    / max(self.total_abs_change_mass, 1e-12),
                }
            )
        return rows

    def summary(self) -> dict[str, Any]:
        confidence_rows = self.confidence.rows("base_confidence")
        distance_rows = self.distance.rows("distance_to_coarse_boundary")
        return {
            "samples": self.samples,
            "total_pixels": self.total_pixels,
            "changed_pixels": self.total_flips,
            "overall_label_flip_rate": self.total_flips / max(self.total_pixels, 1),
            "overall_mean_abs_probability_change": self.total_abs_change / max(self.total_pixels, 1),
            "confidence_bins": confidence_rows,
            "distance_bins": distance_rows,
            "gt_boundary_concentration": self.boundary_rows(),
        }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(summary: dict[str, Any], output_dir: Path, label: str, dpi: int) -> None:
    confidence = summary["confidence_bins"]
    distance = summary["distance_bins"]
    boundary = summary["gt_boundary_concentration"]
    figure, axes = plt.subplots(2, 3, figsize=(16.5, 8.8), constrained_layout=True)

    def line_panel(axis, rows, key, title, ylabel, percent=False):
        values = np.asarray([float(row[key]) for row in rows])
        if percent:
            values *= 100.0
        axis.plot(range(len(rows)), values, marker="o", color="#1769AA", linewidth=2)
        axis.set_xticks(range(len(rows)), [row["bin"] for row in rows], rotation=25, ha="right")
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)

    line_panel(
        axes[0, 0],
        confidence,
        "mean_abs_probability_change",
        "Change magnitude by base confidence",
        r"Mean $|p_{ref}-p_{base}|$",
    )
    line_panel(
        axes[0, 1],
        confidence,
        "label_flip_rate",
        "Label flips by base confidence",
        "Flip rate (%)",
        percent=True,
    )
    line_panel(
        axes[0, 2],
        distance,
        "mean_abs_probability_change",
        "Change magnitude by boundary distance",
        r"Mean $|p_{ref}-p_{base}|$",
    )
    line_panel(
        axes[1, 0],
        distance,
        "label_flip_rate",
        "Label flips by boundary distance",
        "Flip rate (%)",
        percent=True,
    )

    radii = [str(row["radius_pixels"]) for row in boundary]
    change_share = 100 * np.asarray([float(row["changed_pixel_share_in_band"]) for row in boundary])
    area_share = 100 * np.asarray([float(row["boundary_band_pixel_fraction"]) for row in boundary])
    x = np.arange(len(boundary))
    width = 0.36
    axes[1, 1].bar(x - width / 2, change_share, width, label="Changed pixels", color="#1769AA")
    axes[1, 1].bar(x + width / 2, area_share, width, label="Image area", color="#A8B4BB")
    axes[1, 1].set_xticks(x, radii)
    axes[1, 1].set_xlabel("GT boundary radius (pixels)")
    axes[1, 1].set_ylabel("Share (%)")
    axes[1, 1].set_title("Changes near the GT boundary")
    axes[1, 1].legend(frameon=False)
    axes[1, 1].grid(axis="y", alpha=0.25)

    enrichment = [float(row["change_enrichment"]) for row in boundary]
    axes[1, 2].plot(x, enrichment, marker="o", linewidth=2, color="#C45B45")
    axes[1, 2].axhline(1.0, color="#777777", linestyle="--", linewidth=1)
    axes[1, 2].set_xticks(x, radii)
    axes[1, 2].set_xlabel("GT boundary radius (pixels)")
    axes[1, 2].set_ylabel("Change-share / area-share")
    axes[1, 2].set_title("Boundary concentration enrichment")
    axes[1, 2].grid(alpha=0.25)

    figure.suptitle(
        f"{label}: semantic preservation and intervention concentration (N={summary['samples']})",
        fontsize=15,
        fontweight="bold",
    )
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"intervention_concentration.{suffix}", dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def markdown_report(summary: dict[str, Any], label: str, kind: str) -> str:
    lines = [
        f"# {label}: Semantic Preservation and Intervention Concentration",
        "",
        f"- Samples: {summary['samples']}",
        f"- Overall mean |p_refined - p_base|: {summary['overall_mean_abs_probability_change']:.6f}",
        f"- Overall binary label flip rate: {100 * summary['overall_label_flip_rate']:.4f}%",
        f"- Input kind: {kind}",
        "",
    ]
    if kind == "text4seg":
        lines.extend(
            [
                "> Text4Seg provides a hard mask rather than native probability confidence. "
                "Its confidence bins are therefore degenerate; use the distance-to-boundary analysis.",
                "",
            ]
        )
    lines.extend(
        [
            "## Base-confidence bins",
            "",
            "| Confidence | Pixels | Mean abs. change | Label flip rate |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in summary["confidence_bins"]:
        lines.append(
            f"| {row['bin']} | {row['pixel_count']} | "
            f"{row['mean_abs_probability_change']:.6f} | {100 * row['label_flip_rate']:.4f}% |"
        )
    lines.extend(
        [
            "",
            "## Distance to coarse boundary",
            "",
            "| Distance (px) | Pixels | Mean abs. change | Label flip rate |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in summary["distance_bins"]:
        lines.append(
            f"| {row['bin']} | {row['pixel_count']} | "
            f"{row['mean_abs_probability_change']:.6f} | {100 * row['label_flip_rate']:.4f}% |"
        )
    lines.extend(
        [
            "",
            "## Changes near the GT boundary",
            "",
            "| Radius (px) | Boundary area | Changed pixels in band | Change share | Enrichment |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["gt_boundary_concentration"]:
        lines.append(
            f"| {row['radius_pixels']} | {100 * row['boundary_band_pixel_fraction']:.3f}% | "
            f"{row['changed_pixels_in_band']} | {100 * row['changed_pixel_share_in_band']:.3f}% | "
            f"{row['change_enrichment']:.2f}x |"
        )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure whether FreeRef interventions stay in uncertain and boundary-local regions."
    )
    parser.add_argument("--kind", choices=("stamp", "text4seg"), required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--boundary-sigma", type=float, default=8.0)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--max-errors", type=int, default=20)
    add_refiner_arguments(parser)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.limit < 0 or args.dpi <= 0 or args.max_errors < 0:
        raise ValueError("limit/max-errors must be non-negative and dpi must be positive")
    args.rows = args.rows.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if not args.rows.is_file():
        raise FileNotFoundError(args.rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_csv(args.rows)
    if args.limit:
        rows = rows[: args.limit]
    refiner = build_refiner(args)
    accumulator = InterventionAccumulator(args.threshold, GT_BOUNDARY_RADII)
    failures = []
    for row in tqdm(rows, desc=f"{args.label} concentration", dynamic_ncols=True):
        try:
            view = load_view(
                args.kind,
                row,
                args.rows,
                args.label,
                refiner,
                args.boundary_sigma,
            )
            accumulator.update(
                view.coarse_probability,
                view.refined_probability,
                view.target,
            )
        except Exception as error:
            failures.append({"name": row.get("name", ""), "error": str(error)})
            if len(failures) > args.max_errors:
                raise RuntimeError(f"Too many sample failures: {failures[:3]}") from error

    if accumulator.samples == 0:
        raise RuntimeError("No samples were analyzed")
    summary = accumulator.summary()
    summary.update(
        {
            "label": args.label,
            "kind": args.kind,
            "source_rows": str(args.rows),
            "threshold": args.threshold,
            "failures": failures,
        }
    )
    (args.output_dir / "intervention_concentration.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    write_csv(args.output_dir / "confidence_bins.csv", summary["confidence_bins"])
    write_csv(args.output_dir / "distance_bins.csv", summary["distance_bins"])
    write_csv(
        args.output_dir / "gt_boundary_concentration.csv",
        summary["gt_boundary_concentration"],
    )
    (args.output_dir / "intervention_concentration.md").write_text(
        markdown_report(summary, args.label, args.kind), encoding="utf-8"
    )
    plot_summary(summary, args.output_dir, args.label, args.dpi)
    print(markdown_report(summary, args.label, args.kind), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
