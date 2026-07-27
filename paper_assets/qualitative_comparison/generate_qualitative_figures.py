from __future__ import annotations

import argparse
import csv
import hashlib
import json
import textwrap
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion
from tqdm import tqdm

from paper_assets.intro_figure.generate_intro_motivation_figure import (
    PairedCandidate,
    canonical_instance_id,
    diverse_samples,
    load_candidate,
    manifest_index,
    pair_candidates,
    read_csv,
    resize_rgb,
)
from training_free_refine.eval_stamp_dumps import boundary_iou, mask_iou
from training_free_refine.postprocess_baselines import (
    PostprocessBaselineConfig,
    densecrf_probability,
    fast_bilateral_solver_probability,
    guided_filter_probability,
    slic_region_average_probability,
)
from training_free_refine.visualize_comparison import (
    add_refiner_arguments,
    build_refiner,
    load_stamp_view,
)
from universal_freeref.io import load_rgb


BLUE = np.asarray((33, 122, 196), dtype=np.float32)
BLUE_EDGE = np.asarray((4, 55, 118), dtype=np.uint8)
GREEN = np.asarray((37, 166, 102), dtype=np.float32)
GREEN_EDGE = np.asarray((10, 91, 55), dtype=np.uint8)
INK = "#17242D"
MUTED = "#5B6870"
OURS = "#1769AA"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate paired main-table and post-processing qualitative figures."
    )
    parser.add_argument("--stamp-rows", type=Path, required=True)
    parser.add_argument("--text4seg-rows", type=Path, required=True)
    parser.add_argument("--pixellm-rows", type=Path, required=True)
    parser.add_argument("--pixellm-manifest", type=Path, required=True)
    parser.add_argument("--postprocess-rows", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=4)
    parser.add_argument("--rows-per-page", type=int, default=4)
    parser.add_argument("--zoom-rows-per-page", type=int, default=2)
    parser.add_argument("--candidate-pool", type=int, default=96)
    parser.add_argument(
        "--render-style",
        choices=("overlay", "binary_zoom", "both", "masks_only"),
        default="overlay",
    )
    parser.add_argument(
        "--main-selection-mode",
        choices=("balanced", "hard_recovery"),
        default="balanced",
    )
    parser.add_argument(
        "--post-selection-mode",
        choices=("balanced", "hard_recovery"),
        default="balanced",
    )
    parser.add_argument("--hard-max-base-iou", type=float, default=0.78)
    parser.add_argument("--hard-min-final-iou", type=float, default=0.72)
    parser.add_argument("--hard-min-iou-gain", type=float, default=0.04)
    parser.add_argument("--hard-min-improved-models", type=int, default=2)
    parser.add_argument("--main-sample-id", action="append", default=[])
    parser.add_argument("--post-sample-id", action="append", default=[])
    parser.add_argument("--boundary-tolerance", type=int, default=2)
    parser.add_argument("--boundary-sigma", type=float, default=8.0)
    parser.add_argument("--dpi", type=int, default=260)
    add_refiner_arguments(parser)
    return parser.parse_args()


def normalize_ids(values: list[str]) -> list[str]:
    return [str(int(value)) if value.isdigit() else value for value in values]


def mask_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    color: np.ndarray,
    edge_color: np.ndarray,
    alpha: float = 0.48,
) -> np.ndarray:
    image = np.asarray(image, dtype=np.uint8)
    mask = np.asarray(mask, dtype=bool)
    result = image.astype(np.float32).copy()
    result[mask] = (1.0 - alpha) * result[mask] + alpha * color
    result = np.clip(result, 0, 255).astype(np.uint8)
    boundary = binary_dilation(mask) ^ binary_erosion(mask)
    result[boundary] = edge_color
    return result


def metrics(mask: np.ndarray, target: np.ndarray, tolerance: int) -> dict[str, float]:
    iou, _, _ = mask_iou(mask, target)
    return {
        "IoU": iou,
        "bIoU": boundary_iou(mask, target, tolerance),
    }


def row_value(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def recovery_values(row: dict[str, str]) -> dict[str, float]:
    base_iou = row_value(row, "coarse_iou")
    final_iou = row_value(row, "refined_iou")
    base_boundary = row_value(row, "coarse_boundary_iou")
    final_boundary = row_value(row, "refined_boundary_iou")
    return {
        "base_iou": base_iou,
        "final_iou": final_iou,
        "iou_gain": final_iou - base_iou,
        "base_boundary_iou": base_boundary,
        "final_boundary_iou": final_boundary,
        "boundary_gain": final_boundary - base_boundary,
    }


def hard_recovery_score(values: list[dict[str, float]]) -> float:
    mean_base = float(np.mean([value["base_iou"] for value in values]))
    mean_final = float(np.mean([value["final_iou"] for value in values]))
    mean_gain = float(np.mean([value["iou_gain"] for value in values]))
    mean_boundary_gain = float(np.mean([value["boundary_gain"] for value in values]))
    hardness = max(0.0, 0.85 - mean_base)
    final_quality = max(0.0, mean_final - 0.60)
    return (
        3.0 * mean_gain
        + 2.2 * mean_boundary_gain
        + 0.8 * hardness
        + 0.8 * final_quality
    )


def pair_hard_recovery_candidates(
    stamp_rows: list[dict[str, str]],
    text_rows: list[dict[str, str]],
    pixel_rows: list[dict[str, str]],
    max_base_iou: float,
    min_final_iou: float,
    min_iou_gain: float,
    min_improved_models: int,
) -> list[PairedCandidate]:
    sources = []
    for rows in (stamp_rows, text_rows, pixel_rows):
        sources.append(
            {
                instance_id: row
                for row in rows
                if (instance_id := canonical_instance_id(row))
            }
        )
    candidates: list[PairedCandidate] = []
    for instance_id in sources[0].keys() & sources[1].keys() & sources[2].keys():
        rows = [source[instance_id] for source in sources]
        values = [recovery_values(row) for row in rows]
        mean_base = float(np.mean([value["base_iou"] for value in values]))
        mean_final = float(np.mean([value["final_iou"] for value in values]))
        mean_gain = float(np.mean([value["iou_gain"] for value in values]))
        improved = sum(value["iou_gain"] >= min_iou_gain for value in values)
        if not 0.02 <= mean_base <= max_base_iou:
            continue
        if mean_final < min_final_iou or mean_gain < min_iou_gain:
            continue
        if improved < min_improved_models:
            continue
        candidates.append(
            PairedCandidate(
                instance_id,
                rows[0],
                rows[1],
                rows[2],
                hard_recovery_score(values),
            )
        )
    return sorted(candidates, key=lambda candidate: candidate.score, reverse=True)


def save_grid(
    rows: list[dict[str, Any]],
    titles: list[str],
    output_stem: Path,
    dpi: int,
    ours_columns: set[int],
) -> None:
    count = len(rows)
    figure, axes = plt.subplots(
        count,
        len(titles),
        figsize=(16.2, max(2.45 * count, 3.2)),
        squeeze=False,
    )
    figure.subplots_adjust(
        left=0.13, right=0.995, top=0.93, bottom=0.03, wspace=0.025, hspace=0.09
    )
    for row_index, row in enumerate(rows):
        for column, (title, panel) in enumerate(zip(titles, row["panels"])):
            axis = axes[row_index, column]
            axis.imshow(panel)
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_linewidth(1.4 if column in ours_columns else 0.45)
                spine.set_color(OURS if column in ours_columns else "#C5CDD2")
            if row_index == 0:
                axis.set_title(
                    title,
                    fontsize=9.0,
                    fontweight="bold" if column in ours_columns else "semibold",
                    color=OURS if column in ours_columns else INK,
                    pad=5,
                )
        position = axes[row_index, 0].get_position()
        prompt = textwrap.fill(str(row["prompt"]), width=27)
        figure.text(
            0.008,
            0.5 * (position.y0 + position.y1),
            f"#{row_index + 1}\n{prompt}",
            ha="left",
            va="center",
            fontsize=7.4,
            color=MUTED,
            linespacing=1.25,
        )
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(
            output_stem.with_suffix(f".{suffix}"),
            dpi=dpi,
            facecolor="white",
            bbox_inches="tight",
            pad_inches=0.04,
        )
    plt.close(figure)


def save_grid_pages(
    rows: list[dict[str, Any]],
    titles: list[str],
    output_stem: Path,
    dpi: int,
    ours_columns: set[int],
    rows_per_page: int,
) -> list[str]:
    pages: list[str] = []
    page_count = (len(rows) + rows_per_page - 1) // rows_per_page
    for page_index in range(page_count):
        start = page_index * rows_per_page
        chunk = rows[start : start + rows_per_page]
        stem = (
            output_stem
            if page_count == 1
            else output_stem.with_name(f"{output_stem.name}_page_{page_index + 1:02d}")
        )
        save_grid(chunk, titles, stem, dpi, ours_columns)
        pages.append(str(stem))
    return pages


def binary_panel(mask: np.ndarray) -> np.ndarray:
    value = np.asarray(mask, dtype=bool).astype(np.uint8) * 255
    return np.repeat(value[..., None], 3, axis=2)


def window_argmax(score: np.ndarray, height: int, width: int) -> tuple[int, int]:
    score = np.asarray(score, dtype=np.float64)
    integral = np.pad(score, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    sums = (
        integral[height:, width:]
        - integral[:-height, width:]
        - integral[height:, :-width]
        + integral[:-height, :-width]
    )
    y, x = np.unravel_index(int(np.argmax(sums)), sums.shape)
    return int(y), int(x)


def best_zoom_box(
    target: np.ndarray,
    base_masks: list[np.ndarray],
    refined_masks: list[np.ndarray],
    fraction: float = 0.30,
) -> tuple[int, int, int, int]:
    target = np.asarray(target, dtype=bool)
    height, width = target.shape
    crop_height = min(height, max(48, int(round(height * fraction))))
    crop_width = min(width, max(48, int(round(width * fraction))))
    improvement = np.zeros_like(target, dtype=np.float64)
    regression = np.zeros_like(target, dtype=np.float64)
    for base, refined in zip(base_masks, refined_masks):
        base_error = np.asarray(base, dtype=bool) != target
        refined_error = np.asarray(refined, dtype=bool) != target
        improvement += base_error & ~refined_error
        regression += ~base_error & refined_error
    boundary = binary_dilation(target) ^ binary_erosion(target)
    boundary_band = binary_dilation(
        boundary, iterations=max(2, int(round(min(height, width) * 0.025)))
    )
    score = (2.5 * improvement - regression) * (1.0 + boundary_band)
    if float(score.max()) <= 0.0:
        score = boundary_band.astype(np.float64)
    y0, x0 = window_argmax(score, crop_height, crop_width)
    return x0, y0, x0 + crop_width, y0 + crop_height


def save_binary_zoom_grid(
    rows: list[dict[str, Any]],
    titles: list[str],
    output_stem: Path,
    dpi: int,
    ours_columns: set[int],
) -> None:
    sample_count = len(rows)
    figure, axes = plt.subplots(
        sample_count * 2,
        len(titles),
        figsize=(16.2, max(4.25 * sample_count, 4.8)),
        squeeze=False,
    )
    figure.subplots_adjust(
        left=0.125, right=0.995, top=0.94, bottom=0.02, wspace=0.025, hspace=0.035
    )
    roi_color = "#D62F2F"
    for sample_index, row in enumerate(rows):
        full_row = sample_index * 2
        zoom_row = full_row + 1
        x0, y0, x1, y1 = row["zoom_box"]
        for column, (title, panel) in enumerate(zip(titles, row["binary_panels"])):
            full_axis = axes[full_row, column]
            zoom_axis = axes[zoom_row, column]
            interpolation = "bilinear" if column == 0 else "nearest"
            full_axis.imshow(panel, interpolation=interpolation)
            full_axis.add_patch(
                Rectangle(
                    (x0, y0),
                    x1 - x0,
                    y1 - y0,
                    fill=False,
                    edgecolor=roi_color,
                    linewidth=1.35,
                )
            )
            zoom_axis.imshow(panel[y0:y1, x0:x1], interpolation=interpolation)
            for axis in (full_axis, zoom_axis):
                axis.set_xticks([])
                axis.set_yticks([])
                for spine in axis.spines.values():
                    spine.set_linewidth(1.45 if column in ours_columns else 0.5)
                    spine.set_color(OURS if column in ours_columns else "#BFC8CE")
            for spine in zoom_axis.spines.values():
                spine.set_color(OURS if column in ours_columns else roi_color)
                spine.set_linewidth(1.45 if column in ours_columns else 0.85)
            if sample_index == 0:
                full_axis.set_title(
                    title,
                    fontsize=9.0,
                    fontweight="bold" if column in ours_columns else "semibold",
                    color=OURS if column in ours_columns else INK,
                    pad=5,
                )
        full_position = axes[full_row, 0].get_position()
        zoom_position = axes[zoom_row, 0].get_position()
        prompt = textwrap.fill(str(row["prompt"]), width=26)
        figure.text(
            0.008,
            0.5 * (full_position.y0 + full_position.y1),
            f"#{sample_index + 1} Full\n{prompt}",
            ha="left",
            va="center",
            fontsize=7.2,
            color=MUTED,
            linespacing=1.22,
        )
        figure.text(
            0.095,
            0.5 * (zoom_position.y0 + zoom_position.y1),
            "Zoom",
            ha="right",
            va="center",
            fontsize=7.4,
            fontweight="bold",
            color=roi_color,
        )
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(
            output_stem.with_suffix(f".{suffix}"),
            dpi=dpi,
            facecolor="white",
            bbox_inches="tight",
            pad_inches=0.04,
        )
    plt.close(figure)


def save_binary_zoom_grid_pages(
    rows: list[dict[str, Any]],
    titles: list[str],
    output_stem: Path,
    dpi: int,
    ours_columns: set[int],
    rows_per_page: int,
) -> list[str]:
    pages: list[str] = []
    page_count = (len(rows) + rows_per_page - 1) // rows_per_page
    for page_index in range(page_count):
        start = page_index * rows_per_page
        chunk = rows[start : start + rows_per_page]
        stem = (
            output_stem
            if page_count == 1
            else output_stem.with_name(f"{output_stem.name}_page_{page_index + 1:02d}")
        )
        save_binary_zoom_grid(chunk, titles, stem, dpi, ours_columns)
        pages.append(str(stem))
    return pages


def save_panels(root: Path, sample_id: str, names: list[str], panels: list[np.ndarray]) -> None:
    output = root / f"sample_{int(sample_id):06d}"
    output.mkdir(parents=True, exist_ok=True)
    for name, panel in zip(names, panels):
        Image.fromarray(np.asarray(panel, dtype=np.uint8)).save(output / f"{name}.png")


def save_binary_masks(
    root: Path,
    sample_id: str,
    names: list[str],
    masks: list[np.ndarray],
) -> None:
    output = root / f"sample_{int(sample_id):06d}"
    output.mkdir(parents=True, exist_ok=True)
    for name, mask in zip(names, masks):
        value = np.asarray(mask, dtype=bool).astype(np.uint8) * 255
        Image.fromarray(value, mode="L").save(output / f"{name}.png")


def main_table_rows(args: argparse.Namespace, refiner: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    stamp_rows = read_csv(args.stamp_rows)
    text_rows = read_csv(args.text4seg_rows)
    pixel_rows = read_csv(args.pixellm_rows)
    if args.main_selection_mode == "hard_recovery":
        paired = pair_hard_recovery_candidates(
            stamp_rows,
            text_rows,
            pixel_rows,
            args.hard_max_base_iou,
            args.hard_min_final_iou,
            args.hard_min_iou_gain,
            args.hard_min_improved_models,
        )
    else:
        paired = pair_candidates(stamp_rows, text_rows, pixel_rows)
    requested = normalize_ids(args.main_sample_id)
    if requested:
        by_id = {candidate.instance_id: candidate for candidate in paired}
        missing = [value for value in requested if value not in by_id]
        if missing:
            raise ValueError(f"Main-table sample IDs are not eligible: {missing}")
        paired = [by_id[value] for value in requested]
    manifest = manifest_index(args.pixellm_manifest)
    loaded = []
    failures = []
    limit = len(paired) if requested else min(len(paired), args.candidate_pool)
    for candidate in tqdm(
        paired[:limit], desc="Loading paired main-table candidates", dynamic_ncols=True
    ):
        try:
            loaded.append(
                load_candidate(
                    candidate,
                    args.stamp_rows,
                    args.text4seg_rows,
                    manifest,
                    refiner,
                    "STAMP-7B",
                    args.threshold,
                    0.35,
                    args.boundary_sigma,
                )
            )
        except Exception as error:
            failures.append({"instance_id": candidate.instance_id, "error": str(error)})
    if not loaded:
        raise RuntimeError(f"No main-table samples loaded: {failures[:5]}")
    if requested:
        loaded_by_id = {sample.candidate.instance_id: sample for sample in loaded}
        selected = [loaded_by_id[value] for value in requested]
    else:
        loaded.sort(key=lambda sample: sample.score, reverse=True)
        selected = diverse_samples(loaded)[: args.sample_count]
    rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    panel_names = [
        "input",
        "ground_truth",
        "stamp_base",
        "stamp_freeref",
        "text4seg_base",
        "text4seg_freeref",
        "pixellm_base",
        "pixellm_freeref",
    ]
    for sample in selected:
        target = sample.stamp.target.astype(bool)
        pixel_image = resize_rgb(load_rgb(sample.pixellm_item.image), target.shape)
        if not np.array_equal(target, sample.text4seg.target.astype(bool)):
            raise ValueError(f"GT mismatch for sample {sample.candidate.instance_id}")
        pixel_output = refiner.refine_probability(pixel_image, sample.pixellm_probability)
        masks = {
            "stamp_base": sample.stamp.coarse_probability >= args.threshold,
            "stamp_freeref": sample.stamp.refined_probability >= args.threshold,
            "text4seg_base": sample.text4seg.coarse_probability >= args.threshold,
            "text4seg_freeref": sample.text4seg.refined_probability >= args.threshold,
            "pixellm_base": sample.pixellm_probability >= sample.pixellm_item.threshold,
            "pixellm_freeref": pixel_output["refined_probability"].numpy() >= args.threshold,
        }
        panels = [sample.stamp.image, mask_overlay(sample.stamp.image, target, GREEN, GREEN_EDGE)]
        panels.extend(mask_overlay(sample.stamp.image, masks[name], BLUE, BLUE_EDGE) for name in panel_names[2:])
        row_metrics = {
            name: metrics(mask, target, args.boundary_tolerance) for name, mask in masks.items()
        }
        query = sample.stamp.query or sample.pixellm_query
        base_masks = [masks["stamp_base"], masks["text4seg_base"], masks["pixellm_base"]]
        refined_masks = [
            masks["stamp_freeref"],
            masks["text4seg_freeref"],
            masks["pixellm_freeref"],
        ]
        rows.append(
            {
                "sample_id": sample.candidate.instance_id,
                "prompt": query,
                "panels": panels,
                "binary_panels": [sample.stamp.image]
                + [binary_panel(target)]
                + [binary_panel(masks[name]) for name in panel_names[2:]],
                "zoom_box": best_zoom_box(target, base_masks, refined_masks),
            }
        )
        base_names = ("stamp_base", "text4seg_base", "pixellm_base")
        final_names = ("stamp_freeref", "text4seg_freeref", "pixellm_freeref")
        mean_base_iou = float(np.mean([row_metrics[name]["IoU"] for name in base_names]))
        mean_final_iou = float(np.mean([row_metrics[name]["IoU"] for name in final_names]))
        mean_base_boundary = float(
            np.mean([row_metrics[name]["bIoU"] for name in base_names])
        )
        mean_final_boundary = float(
            np.mean([row_metrics[name]["bIoU"] for name in final_names])
        )
        record = {
            "sample_id": sample.candidate.instance_id,
            "prompt": query,
            "selection_score": sample.candidate.score,
            "mean_base_iou": mean_base_iou,
            "mean_final_iou": mean_final_iou,
            "mean_iou_gain": mean_final_iou - mean_base_iou,
            "mean_base_boundary_iou": mean_base_boundary,
            "mean_final_boundary_iou": mean_final_boundary,
            "mean_boundary_gain": mean_final_boundary - mean_base_boundary,
            "metrics": row_metrics,
        }
        records.append(record)
        save_panels(args.output_dir / "main_table_panels", sample.candidate.instance_id, panel_names, panels)
        mask_names = ["ground_truth"] + panel_names[2:]
        save_binary_masks(
            args.output_dir / "main_table_binary_masks",
            sample.candidate.instance_id,
            mask_names,
            [target] + [masks[name] for name in panel_names[2:]],
        )
    return rows, records


def postprocess_score(row: dict[str, str], mode: str = "balanced") -> float:
    competitors = ("densecrf", "guided_filter", "fast_bilateral_solver", "slic_average")
    free_iou = float(row.get("freeref_iou", 0.0))
    free_boundary = float(row.get("freeref_boundary_iou", 0.0))
    best_iou = max(float(row.get(f"{name}_iou", 0.0)) for name in competitors)
    best_boundary = max(float(row.get(f"{name}_boundary_iou", 0.0)) for name in competitors)
    base_iou = float(row.get("base_iou", 0.0))
    base_boundary = float(row.get("base_boundary_iou", 0.0))
    if not 0.01 <= base_iou <= 0.98:
        return -1e9
    if mode == "hard_recovery":
        hardness = max(0.0, 0.85 - base_iou)
        final_quality = max(0.0, free_iou - 0.60)
        return (
            3.0 * (free_iou - base_iou)
            + 2.4 * (free_boundary - base_boundary)
            + 1.5 * (free_iou - best_iou)
            + 1.8 * (free_boundary - best_boundary)
            + 0.8 * hardness
            + 0.8 * final_quality
        )
    return 2.0 * (free_boundary - best_boundary) + (free_iou - best_iou) + 0.4 * (
        free_boundary - base_boundary
    ) + 0.2 * (free_iou - base_iou)


def postprocess_rows(args: argparse.Namespace, refiner: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    stamp_by_id = {
        value: row
        for row in read_csv(args.stamp_rows)
        if (value := canonical_instance_id(row))
    }
    metric_by_id = {
        value: row
        for row in read_csv(args.postprocess_rows)
        if (value := canonical_instance_id(row))
    }
    requested = normalize_ids(args.post_sample_id)
    if requested:
        candidate_ids = requested
    else:
        eligible_ids = stamp_by_id.keys() & metric_by_id.keys()
        if args.post_selection_mode == "hard_recovery":
            eligible_ids = {
                value
                for value in eligible_ids
                if row_value(metric_by_id[value], "base_iou") <= args.hard_max_base_iou
                and row_value(metric_by_id[value], "freeref_iou") >= args.hard_min_final_iou
                and row_value(metric_by_id[value], "freeref_iou")
                - row_value(metric_by_id[value], "base_iou")
                >= args.hard_min_iou_gain
            }
        candidate_ids = sorted(
            eligible_ids,
            key=lambda value: postprocess_score(
                metric_by_id[value], args.post_selection_mode
            ),
            reverse=True,
        )
    config = PostprocessBaselineConfig(
        threshold=args.threshold,
        n_segments=args.n_segments,
        compactness=args.compactness,
        slic_sigma=args.slic_sigma,
    )
    selected: list[tuple[str, Any]] = []
    seen: set[str] = set()
    for sample_id in tqdm(
        candidate_ids, desc="Selecting post-processing candidates", dynamic_ncols=True
    ):
        if sample_id not in stamp_by_id:
            if requested:
                raise ValueError(f"STAMP has no sample ID {sample_id}")
            continue
        view = load_stamp_view(stamp_by_id[sample_id], args.stamp_rows, "STAMP-7B", refiner)
        key = hashlib.sha1(view.image.tobytes()).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        selected.append((sample_id, view))
        if len(selected) >= (len(requested) if requested else args.sample_count):
            break
    if not selected:
        raise RuntimeError("No post-processing samples were selected.")
    titles = [
        "input",
        "ground_truth",
        "coarse",
        "densecrf",
        "guided_filter",
        "fbs",
        "slic_average",
        "freeref",
    ]
    rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for sample_id, view in tqdm(
        selected, desc="Rendering post-processing methods", dynamic_ncols=True
    ):
        image = view.image
        target = view.target.astype(bool)
        probability = view.coarse_probability.astype(np.float32)
        probabilities = {
            "coarse": probability,
            "densecrf": densecrf_probability(image, probability, config),
            "guided_filter": guided_filter_probability(image, probability, config),
            "fbs": fast_bilateral_solver_probability(image, probability, config),
            "slic_average": slic_region_average_probability(image, probability, config),
            "freeref": view.refined_probability,
        }
        masks = {name: value >= args.threshold for name, value in probabilities.items()}
        panels = [image, mask_overlay(image, target, GREEN, GREEN_EDGE)]
        panels.extend(mask_overlay(image, masks[name], BLUE, BLUE_EDGE) for name in titles[2:])
        row_metrics = {
            name: metrics(mask, target, args.boundary_tolerance) for name, mask in masks.items()
        }
        competitor_names = ("coarse", "densecrf", "guided_filter", "fbs", "slic_average")
        rows.append(
            {
                "sample_id": sample_id,
                "prompt": view.query,
                "panels": panels,
                "binary_panels": [image]
                + [binary_panel(target)]
                + [binary_panel(masks[name]) for name in titles[2:]],
                "zoom_box": best_zoom_box(
                    target,
                    [masks[name] for name in competitor_names],
                    [masks["freeref"] for _ in competitor_names],
                ),
            }
        )
        records.append(
            {
                "sample_id": sample_id,
                "prompt": view.query,
                "selection_score": postprocess_score(
                    metric_by_id.get(sample_id, {}), args.post_selection_mode
                ),
                "metrics": row_metrics,
            }
        )
        save_panels(args.output_dir / "postprocess_panels", sample_id, titles, panels)
        mask_names = ["ground_truth"] + titles[2:]
        save_binary_masks(
            args.output_dir / "postprocess_binary_masks",
            sample_id,
            mask_names,
            [target] + [masks[name] for name in titles[2:]],
        )
    return rows, records


def flatten_records(records: list[dict[str, Any]], path: Path) -> None:
    rows = []
    for record in records:
        row: dict[str, Any] = {
            "sample_id": record["sample_id"],
            "prompt": record["prompt"],
        }
        for key in (
            "selection_score",
            "mean_base_iou",
            "mean_final_iou",
            "mean_iou_gain",
            "mean_base_boundary_iou",
            "mean_final_boundary_iou",
            "mean_boundary_gain",
        ):
            if key in record:
                row[key] = record[key]
        for method, values in record["metrics"].items():
            row[f"{method}_iou"] = values["IoU"]
            row[f"{method}_boundary_iou"] = values["bIoU"]
        rows.append(row)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    if (
        args.sample_count <= 0
        or args.rows_per_page <= 0
        or args.zoom_rows_per_page <= 0
        or args.candidate_pool <= 0
        or args.dpi <= 0
        or not 1 <= args.hard_min_improved_models <= 3
    ):
        raise ValueError(
            "Counts and dpi must be positive; hard-min-improved-models must be 1--3."
        )
    for name in (
        "stamp_rows",
        "text4seg_rows",
        "pixellm_rows",
        "pixellm_manifest",
        "postprocess_rows",
    ):
        path = getattr(args, name).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        setattr(args, name, path)
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    refiner = build_refiner(args)

    main_rows, main_records = main_table_rows(args, refiner)
    main_titles = [
        "Input",
        "GT",
        "STAMP",
        "+FreeRef",
        "Text4Seg",
        "+FreeRef",
        "PixelLM",
        "+FreeRef",
    ]
    main_pages = []
    if args.render_style in ("overlay", "both"):
        main_pages = save_grid_pages(
            main_rows,
            main_titles,
            args.output_dir / "main_table_qualitative",
            args.dpi,
            {3, 5, 7},
            args.rows_per_page,
        )
    main_binary_zoom_pages = []
    if args.render_style in ("binary_zoom", "both"):
        main_binary_zoom_pages = save_binary_zoom_grid_pages(
            main_rows,
            main_titles,
            args.output_dir / "main_table_binary_zoom",
            args.dpi,
            {3, 5, 7},
            args.zoom_rows_per_page,
        )
    flatten_records(main_records, args.output_dir / "main_table_qualitative_rows.csv")

    post_rows, post_records = postprocess_rows(args, refiner)
    post_titles = [
        "Input",
        "GT",
        "Coarse",
        "DenseCRF",
        "Guided Filter",
        "FBS",
        "SLIC Avg.",
        "FreeRef",
    ]
    post_pages = []
    if args.render_style in ("overlay", "both"):
        post_pages = save_grid_pages(
            post_rows,
            post_titles,
            args.output_dir / "postprocess_qualitative",
            args.dpi,
            {7},
            args.rows_per_page,
        )
    post_binary_zoom_pages = []
    if args.render_style in ("binary_zoom", "both"):
        post_binary_zoom_pages = save_binary_zoom_grid_pages(
            post_rows,
            post_titles,
            args.output_dir / "postprocess_binary_zoom",
            args.dpi,
            {7},
            args.zoom_rows_per_page,
        )
    flatten_records(post_records, args.output_dir / "postprocess_qualitative_rows.csv")

    manifest = {
        "main_table": main_records,
        "postprocess": post_records,
        "main_table_pages": main_pages,
        "postprocess_pages": post_pages,
        "main_table_binary_zoom_pages": main_binary_zoom_pages,
        "postprocess_binary_zoom_pages": post_binary_zoom_pages,
        "inputs": {
            "stamp_rows": str(args.stamp_rows),
            "text4seg_rows": str(args.text4seg_rows),
            "pixellm_rows": str(args.pixellm_rows),
            "pixellm_manifest": str(args.pixellm_manifest),
            "postprocess_rows": str(args.postprocess_rows),
        },
    }
    (args.output_dir / "qualitative_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "main_samples": [r["sample_id"] for r in main_records],
                "postprocess_samples": [r["sample_id"] for r in post_records],
                "main_pages": main_pages,
                "postprocess_pages": post_pages,
                "main_binary_zoom_pages": main_binary_zoom_pages,
                "postprocess_binary_zoom_pages": post_binary_zoom_pages,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
