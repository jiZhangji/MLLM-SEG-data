from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from training_free_refine.refiner import boundary_uncertainty
from training_free_refine.visualize_comparison import (
    SampleView,
    add_refiner_arguments,
    build_refiner,
    load_view,
    read_csv,
    safe_name,
    transition_map,
    view_metrics,
    write_csv,
)


def parse_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def parse_bool(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def row_iou_delta(row: dict[str, str]) -> float:
    if str(row.get("iou_delta", "")).strip():
        return parse_float(row, "iou_delta")
    return parse_float(row, "refined_iou") - parse_float(row, "coarse_iou")


def row_boundary_delta(row: dict[str, str]) -> float:
    return parse_float(row, "refined_boundary_iou") - parse_float(
        row, "coarse_boundary_iou"
    )


def row_is_loadable(row: dict[str, str], kind: str) -> bool:
    if parse_bool(row.get("no_target")) or parse_bool(row.get("empty_prediction")):
        return False
    if str(row.get("prediction_error", "")).strip():
        return False
    required = ("dump",) if kind == "stamp" else ("image", "gt_mask", "pred_mask")
    return all(str(row.get(key, "")).strip() for key in required)


def _representative_order(rows: list[tuple[int, dict[str, str]]]) -> list[tuple[int, dict[str, str]]]:
    if not rows:
        return []
    gains = np.asarray([row_iou_delta(row) for _, row in rows], dtype=float)
    boundary = np.asarray([row_boundary_delta(row) for _, row in rows], dtype=float)
    coarse = np.asarray([parse_float(row, "coarse_iou", 0.5) for _, row in rows], dtype=float)
    center = np.asarray(
        [np.median(gains), np.median(boundary), np.median(coarse)], dtype=float
    )
    scale = np.asarray(
        [
            max(float(np.quantile(gains, 0.75) - np.quantile(gains, 0.25)), 1e-4),
            max(float(np.quantile(boundary, 0.75) - np.quantile(boundary, 0.25)), 1e-4),
            max(float(np.quantile(coarse, 0.75) - np.quantile(coarse, 0.25)), 1e-3),
        ]
    )

    def score(item: tuple[int, dict[str, str]]) -> float:
        _, row = item
        values = np.asarray(
            [
                row_iou_delta(row),
                row_boundary_delta(row),
                parse_float(row, "coarse_iou", 0.5),
            ]
        )
        distance = float(np.mean(np.abs(values - center) / scale))
        coarse_iou = values[2]
        if coarse_iou < 0.25 or coarse_iou > 0.92:
            distance += 2.0
        return distance

    return sorted(rows, key=lambda item: (score(item), row_iou_delta(item[1])))


def preselect_rows(
    rows: list[dict[str, str]],
    kind: str,
    pool_size: int,
    sample_name: str = "",
    selection: str = "representative_success",
) -> list[tuple[int, dict[str, str]]]:
    """Use existing evaluation metrics to avoid recomputing the whole validation set."""
    available = [
        (index, row)
        for index, row in enumerate(rows)
        if row_is_loadable(row, kind)
    ]
    if sample_name:
        exact = [
            item
            for item in available
            if str(item[1].get("name", "")).strip() == sample_name
        ]
        matches = exact or [
            item
            for item in available
            if sample_name.lower() in str(item[1].get("name", "")).lower()
        ]
        if not matches:
            raise ValueError(f"No loadable row matches --sample-name={sample_name!r}.")
        return matches[:1]

    strict = [
        item
        for item in available
        if row_iou_delta(item[1]) > 1e-4
        and row_boundary_delta(item[1]) > 1e-4
        and 0.15 <= parse_float(item[1], "coarse_iou", 0.5) <= 0.95
    ]
    relaxed = [item for item in available if row_iou_delta(item[1]) > 1e-4]
    if selection == "best_iou":
        return sorted(
            relaxed or available,
            key=lambda item: row_iou_delta(item[1]),
            reverse=True,
        )[:pool_size]
    if selection == "best_boundary":
        boundary_positive = [
            item for item in available if row_boundary_delta(item[1]) > 1e-4
        ]
        return sorted(
            boundary_positive or available,
            key=lambda item: row_boundary_delta(item[1]),
            reverse=True,
        )[:pool_size]
    candidates = strict or relaxed or available
    return _representative_order(candidates)[:pool_size]


def representative_rank(
    loaded: list[tuple[int, dict[str, str], SampleView, dict[str, float | int]]],
    selection: str,
) -> list[tuple[int, dict[str, str], SampleView, dict[str, float | int]]]:
    if selection == "best_iou":
        return sorted(loaded, key=lambda item: float(item[3]["iou_delta"]), reverse=True)
    if selection == "best_boundary":
        return sorted(
            loaded,
            key=lambda item: float(item[3]["boundary_iou_delta"]),
            reverse=True,
        )

    successful = [
        item
        for item in loaded
        if float(item[3]["iou_delta"]) > 0
        and float(item[3]["boundary_iou_delta"]) > 0
        and float(item[3]["corrected_fraction"])
        > float(item[3]["regressed_fraction"])
        and 0.01 <= float(item[3]["object_fraction"]) <= 0.75
        and 5e-4 <= float(item[3]["changed_fraction"]) <= 0.25
    ]
    candidates = successful or [
        item for item in loaded if float(item[3]["iou_delta"]) > 0
    ] or loaded
    keys = ("iou_delta", "boundary_iou_delta", "coarse_iou", "object_fraction")
    values = np.asarray(
        [[float(item[3][key]) for key in keys] for item in candidates], dtype=float
    )
    centers = np.median(values, axis=0)
    scales = np.maximum(np.quantile(values, 0.75, axis=0) - np.quantile(values, 0.25, axis=0), 1e-5)
    distances = np.mean(np.abs(values - centers) / scales, axis=1)
    return [
        item
        for _, item in sorted(
            zip(distances.tolist(), candidates), key=lambda pair: pair[0]
        )
    ]


def draw_mask_contours(axis: plt.Axes, mask: np.ndarray, color: str) -> None:
    if np.any(mask) and not np.all(mask):
        axis.contour(mask.astype(float), levels=[0.5], colors=[color], linewidths=1.0)


def save_contact_sheet(
    ranked: list[tuple[int, dict[str, str], SampleView, dict[str, float | int]]],
    output_path: Path,
    threshold: float,
    count: int,
    dpi: int,
) -> None:
    shown = ranked[:count]
    if not shown:
        return
    figure, axes = plt.subplots(
        len(shown),
        4,
        figsize=(12.8, max(2.4 * len(shown), 3.0)),
        squeeze=False,
        constrained_layout=True,
    )
    for rank, (_, _, view, metrics) in enumerate(shown, start=1):
        coarse = view.coarse_probability >= threshold
        refined = view.refined_probability >= threshold
        transition, _ = transition_map(view, threshold)
        axes[rank - 1, 0].imshow(view.image)
        axes[rank - 1, 0].set_title(f"#{rank} {view.name}", fontsize=8)
        axes[rank - 1, 1].imshow(view.image)
        draw_mask_contours(axes[rank - 1, 1], view.target, "white")
        draw_mask_contours(axes[rank - 1, 1], coarse, "#ef553b")
        axes[rank - 1, 1].set_title(
            f"Baseline IoU {float(metrics['coarse_iou']):.3f}", fontsize=8
        )
        axes[rank - 1, 2].imshow(view.image)
        draw_mask_contours(axes[rank - 1, 2], view.target, "white")
        draw_mask_contours(axes[rank - 1, 2], refined, "#2684ff")
        axes[rank - 1, 2].set_title(
            f"FreeRef IoU {float(metrics['refined_iou']):.3f}", fontsize=8
        )
        axes[rank - 1, 3].imshow(transition)
        axes[rank - 1, 3].set_title(
            f"IoU {float(metrics['iou_delta']):+.3f} | "
            f"Boundary {float(metrics['boundary_iou_delta']):+.3f}",
            fontsize=8,
        )
        for axis in axes[rank - 1]:
            axis.axis("off")
    figure.suptitle(
        "Real evaluation candidates (white: GT, red: baseline, blue: FreeRef)",
        fontsize=11,
        fontweight="bold",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def export_bundle(
    output_dir: Path,
    view: SampleView,
    metrics: dict[str, float | int],
    source_row: dict[str, str],
    source_index: int,
    args: argparse.Namespace,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    hard = view.coarse_probability >= args.threshold
    hard_uncertainty = boundary_uncertainty(hard, sigma=args.boundary_sigma)
    bundle_path = output_dir / "selected_real_sample.npz"
    np.savez_compressed(
        bundle_path,
        scene=view.image,
        target=view.target.astype(np.uint8),
        hard=hard.astype(np.uint8),
        p=np.clip(view.coarse_probability, 0.0, 1.0).astype(np.float32),
        u=np.clip(view.uncertainty, 0.0, 1.0).astype(np.float32),
        u_hard=hard_uncertainty.astype(np.float32),
        r=np.clip(view.graph_probability, 0.0, 1.0).astype(np.float32),
        refined=np.clip(view.refined_probability, 0.0, 1.0).astype(np.float32),
        changed=np.abs(view.refined_probability - view.coarse_probability).astype(np.float32),
        superpixels=view.superpixels.astype(np.int32),
        name=np.asarray(view.name),
        query=np.asarray(view.query),
        model=np.asarray(view.label),
    )
    manifest_path = output_dir / "selected_real_sample.json"
    manifest = {
        "real_experiment_data": True,
        "selection": args.selection,
        "selection_rank": args.rank,
        "kind": args.kind,
        "model": view.label,
        "name": view.name,
        "query": view.query,
        "source_rows": str(args.rows.resolve()),
        "source_row_index": source_index,
        "source_row": source_row,
        "metrics": metrics,
        "refiner_config": asdict(build_refiner(args).config),
        "boundary_sigma": args.boundary_sigma,
        "note": (
            "Ground truth is used only to select and verify a representative sample; "
            "the FreeRef refinement and the framework figure do not consume it."
        ),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return bundle_path, manifest_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select a real evaluation sample and export FreeRef framework assets."
    )
    parser.add_argument("--kind", choices=["stamp", "text4seg"], default="stamp")
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", default="")
    parser.add_argument("--sample-name", default="")
    parser.add_argument(
        "--selection",
        choices=["representative_success", "best_iou", "best_boundary"],
        default="representative_success",
    )
    parser.add_argument("--rank", type=int, default=1, help="1-based candidate rank.")
    parser.add_argument("--candidate-pool", type=int, default=16)
    parser.add_argument("--contact-sheet-count", type=int, default=6)
    parser.add_argument("--boundary-sigma", type=float, default=8.0)
    parser.add_argument("--boundary-tolerance", type=int, default=2)
    parser.add_argument("--dpi", type=int, default=160)
    add_refiner_arguments(parser)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.rank <= 0 or args.candidate_pool <= 0 or args.contact_sheet_count <= 0:
        raise ValueError("rank, candidate-pool, and contact-sheet-count must be positive.")
    args.rows = args.rows.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.label = args.label or ("STAMP-7B" if args.kind == "stamp" else "Text4Seg-7B-p24")
    rows = read_csv(args.rows)
    preselected = preselect_rows(
        rows,
        args.kind,
        args.candidate_pool,
        args.sample_name,
        args.selection,
    )
    if not preselected:
        raise RuntimeError(f"No usable samples found in {args.rows}.")

    refiner = build_refiner(args)
    loaded: list[
        tuple[int, dict[str, str], SampleView, dict[str, float | int]]
    ] = []
    failures: list[dict[str, Any]] = []
    for source_index, row in preselected:
        try:
            view = load_view(
                args.kind,
                row,
                args.rows,
                args.label,
                refiner,
                args.boundary_sigma,
            )
            metrics = view_metrics(view, args.threshold, args.boundary_tolerance)
            loaded.append((source_index, row, view, metrics))
        except Exception as error:
            failures.append(
                {
                    "source_row_index": source_index,
                    "name": row.get("name", ""),
                    "error": f"{type(error).__name__}: {error}",
                }
            )
    if not loaded:
        raise RuntimeError(
            "Every preselected sample failed to load. "
            + json.dumps(failures[:3], ensure_ascii=False)
        )

    ranked = representative_rank(loaded, args.selection)
    if args.rank > len(ranked):
        raise ValueError(
            f"--rank={args.rank} exceeds the {len(ranked)} successfully loaded candidates."
        )
    selected_index, selected_row, selected_view, selected_metrics = ranked[args.rank - 1]
    ranked_rows = [
        {
            "selection_rank": rank,
            "source_row_index": source_index,
            "name": view.name,
            **metrics,
        }
        for rank, (source_index, _, view, metrics) in enumerate(ranked, start=1)
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "framework_candidates.csv", ranked_rows)
    if failures:
        (args.output_dir / "candidate_load_failures.json").write_text(
            json.dumps(failures, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    save_contact_sheet(
        ranked,
        args.output_dir / "framework_candidate_contact_sheet.png",
        args.threshold,
        args.contact_sheet_count,
        args.dpi,
    )
    bundle_path, manifest_path = export_bundle(
        args.output_dir,
        selected_view,
        selected_metrics,
        selected_row,
        selected_index,
        args,
    )
    print(
        json.dumps(
            {
                "selected": selected_view.name,
                "bundle": str(bundle_path),
                "manifest": str(manifest_path),
                "contact_sheet": str(
                    args.output_dir / "framework_candidate_contact_sheet.png"
                ),
                "metrics": selected_metrics,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
