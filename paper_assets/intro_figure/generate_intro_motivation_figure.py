from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.patches import Rectangle
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion, label

from training_free_refine.visualize_comparison import (
    SampleView,
    add_refiner_arguments,
    build_refiner,
    load_stamp_view,
    load_text4seg_view,
)
from training_free_refine.refiner import probability_uncertainty
from universal_freeref.io import load_mask, load_probability, load_rgb
from universal_freeref.schema import ManifestItem


INK = "#17242D"
MUTED = "#5B6870"
BLUE = "#2E78B7"
CYAN = "#27A6B3"
ORANGE = "#E96B3C"
RED = "#D62728"
GREEN = "#2A9D63"
WHITE = "#FFFFFF"
LIGHT = "#F4F6F7"
LINE = "#A8B2B8"


@dataclass
class PairedCandidate:
    instance_id: str
    stamp_row: dict[str, str]
    text4seg_row: dict[str, str]
    pixellm_row: dict[str, str]
    score: float


@dataclass
class LoadedSample:
    candidate: PairedCandidate
    stamp: SampleView
    text4seg: SampleView
    pixellm_item: ManifestItem
    pixellm_query: str
    pixellm_probability: np.ndarray
    pixellm_target: np.ndarray
    bbox_iou: float
    object_fraction: float
    score: float


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"Manifest row is not an object: {value!r}")
            values.append(value)
    if not values:
        raise ValueError(f"Manifest is empty: {path}")
    return values


def parse_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def canonical_instance_id(row: dict[str, str]) -> str:
    value = str(row.get("instance_id", "")).strip()
    if value:
        try:
            return str(int(value))
        except ValueError:
            return value
    for key in ("index", "name"):
        value = str(row.get(key, "")).strip()
        match = re.search(r"(\d+)$", value)
        if match:
            return str(int(match.group(1)))
    return ""


def manifest_index(path: Path) -> dict[str, tuple[ManifestItem, dict[str, Any]]]:
    result: dict[str, tuple[ManifestItem, dict[str, Any]]] = {}
    for line_number, raw in enumerate(read_jsonl(path), start=1):
        item = ManifestItem.from_mapping(raw, path, line_number)
        instance_id = str(item.instance_id or line_number - 1)
        try:
            instance_id = str(int(instance_id))
        except ValueError:
            pass
        result[instance_id] = (item, raw)
    return result


def row_boundary_gain(row: dict[str, str]) -> float:
    return parse_float(row, "refined_boundary_iou") - parse_float(
        row, "coarse_boundary_iou"
    )


def row_iou_gain(row: dict[str, str]) -> float:
    if str(row.get("iou_delta", "")).strip():
        return parse_float(row, "iou_delta")
    return parse_float(row, "refined_iou") - parse_float(row, "coarse_iou")


def pair_candidates(
    stamp_rows: list[dict[str, str]],
    text4seg_rows: list[dict[str, str]],
    pixellm_rows: list[dict[str, str]],
) -> list[PairedCandidate]:
    stamp_by_id = {
        instance_id: row
        for row in stamp_rows
        if (instance_id := canonical_instance_id(row))
    }
    pixellm_by_id = {
        instance_id: row
        for row in pixellm_rows
        if (instance_id := canonical_instance_id(row))
    }
    text4seg_by_id = {
        instance_id: row
        for row in text4seg_rows
        if (instance_id := canonical_instance_id(row))
    }
    candidates: list[PairedCandidate] = []
    for instance_id in sorted(
        stamp_by_id.keys() & text4seg_by_id.keys() & pixellm_by_id.keys(),
        key=lambda value: (
            not value.isdigit(),
            int(value) if value.isdigit() else value,
        ),
    ):
        stamp = stamp_by_id[instance_id]
        text4seg = text4seg_by_id[instance_id]
        pixellm = pixellm_by_id[instance_id]
        stamp_iou = parse_float(stamp, "coarse_iou")
        text4seg_iou = parse_float(text4seg, "coarse_iou")
        pixellm_iou = parse_float(pixellm, "coarse_iou")
        if min(stamp_iou, text4seg_iou, pixellm_iou) < 0.40:
            continue
        stamp_boundary = parse_float(stamp, "coarse_boundary_iou")
        text4seg_boundary = parse_float(text4seg, "coarse_boundary_iou")
        pixellm_boundary = parse_float(pixellm, "coarse_boundary_iou")
        semantic_strength = min(stamp_iou, text4seg_iou, pixellm_iou)
        spatial_gap = (
            max(stamp_iou - stamp_boundary, 0.0)
            + max(text4seg_iou - text4seg_boundary, 0.0)
            + max(pixellm_iou - pixellm_boundary, 0.0)
        ) / 3.0
        boundary_gain = (
            max(row_boundary_gain(stamp), 0.0)
            + max(row_boundary_gain(text4seg), 0.0)
            + max(row_boundary_gain(pixellm), 0.0)
        ) / 3.0
        iou_gain = (
            max(row_iou_gain(stamp), 0.0)
            + max(row_iou_gain(text4seg), 0.0)
            + max(row_iou_gain(pixellm), 0.0)
        ) / 3.0
        score = (
            2.2 * semantic_strength
            + 1.5 * spatial_gap
            + 1.8 * boundary_gain
            + 0.4 * iou_gain
        )
        candidates.append(
            PairedCandidate(instance_id, stamp, text4seg, pixellm, score)
        )
    return sorted(candidates, key=lambda candidate: candidate.score, reverse=True)


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def bbox_iou(
    first: tuple[int, int, int, int] | None,
    second: tuple[int, int, int, int] | None,
) -> float:
    if first is None or second is None:
        return 0.0
    x0 = max(first[0], second[0])
    y0 = max(first[1], second[1])
    x1 = min(first[2], second[2])
    y1 = min(first[3], second[3])
    intersection = max(x1 - x0, 0) * max(y1 - y0, 0)
    first_area = max(first[2] - first[0], 0) * max(first[3] - first[1], 0)
    second_area = max(second[2] - second[0], 0) * max(second[3] - second[1], 0)
    return intersection / max(first_area + second_area - intersection, 1)


def resize_rgb(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if image.shape[:2] == shape:
        return image
    return np.asarray(
        Image.fromarray(image).resize((shape[1], shape[0]), Image.Resampling.BILINEAR)
    )


def image_distance(first: np.ndarray, second: np.ndarray) -> float:
    second = resize_rgb(second, first.shape[:2])
    return float(
        np.mean(np.abs(first.astype(np.float32) - second.astype(np.float32))) / 255.0
    )


def load_candidate(
    candidate: PairedCandidate,
    stamp_rows_path: Path,
    text4seg_rows_path: Path,
    manifest: dict[str, tuple[ManifestItem, dict[str, Any]]],
    refiner: Any,
    stamp_label: str,
    threshold: float,
    minimum_box_iou: float,
    boundary_sigma: float,
) -> LoadedSample:
    if candidate.instance_id not in manifest:
        raise KeyError(f"PixelLM manifest has no instance_id={candidate.instance_id}.")
    stamp = load_stamp_view(candidate.stamp_row, stamp_rows_path, stamp_label, refiner)
    text4seg = load_text4seg_view(
        candidate.text4seg_row,
        text4seg_rows_path,
        "Text4Seg-p24",
        refiner,
        boundary_sigma,
    )
    pixellm_item, pixellm_raw = manifest[candidate.instance_id]
    pixellm_target = load_mask(pixellm_item.gt_mask)
    pixellm_image = load_rgb(pixellm_item.image)
    pixellm_image = resize_rgb(pixellm_image, pixellm_target.shape)
    if image_distance(stamp.image, pixellm_image) > 0.025:
        raise ValueError(
            "STAMP and PixelLM rows share an index but not the same image."
        )
    if image_distance(stamp.image, text4seg.image) > 0.025:
        raise ValueError(
            "STAMP and Text4Seg rows share an index but not the same image."
        )
    pixellm_probability, _ = load_probability(pixellm_item, pixellm_target.shape)
    stamp_coarse = stamp.coarse_probability >= threshold
    localization_iou = bbox_iou(mask_bbox(stamp_coarse), mask_bbox(stamp.target))
    object_fraction = float(stamp.target.mean())
    if not 0.015 <= object_fraction <= 0.70:
        raise ValueError(
            f"Object fraction {object_fraction:.4f} is unsuitable for a compact figure."
        )
    if localization_iou < minimum_box_iou:
        raise ValueError(
            f"Predicted box IoU {localization_iou:.3f} is too low to demonstrate localization."
        )
    query = str(pixellm_raw.get("query") or stamp.query)
    score = candidate.score + 0.8 * localization_iou
    return LoadedSample(
        candidate=candidate,
        stamp=stamp,
        text4seg=text4seg,
        pixellm_item=pixellm_item,
        pixellm_query=query,
        pixellm_probability=pixellm_probability,
        pixellm_target=pixellm_target,
        bbox_iou=localization_iou,
        object_fraction=object_fraction,
        score=score,
    )


def contour(axis: plt.Axes, mask: np.ndarray, color: str, width: float) -> None:
    if mask.any() and not mask.all():
        axis.contour(
            mask.astype(float),
            levels=[0.5],
            colors=[color],
            linewidths=width,
        )


def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: str = ORANGE,
    alpha: float = 0.44,
) -> np.ndarray:
    rgb = np.asarray(colors.to_rgb(color), dtype=np.float32) * 255.0
    result = image.astype(np.float32).copy()
    selected = mask.astype(bool)
    result[selected] = (1.0 - alpha) * result[selected] + alpha * rgb
    return np.clip(result, 0, 255).astype(np.uint8)


def box_overlap(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> float:
    return bbox_iou(first, second)


def error_callouts(
    coarse: np.ndarray,
    target: np.ndarray,
    count: int = 2,
) -> list[dict[str, Any]]:
    height, width = target.shape
    target_boundary = np.logical_xor(target, binary_erosion(target))
    coarse_boundary = np.logical_xor(coarse, binary_erosion(coarse))
    radius = max(2, int(round(min(height, width) * 0.018)))
    boundary_band = binary_dilation(
        target_boundary | coarse_boundary, iterations=radius
    )
    error = np.logical_xor(coarse, target) & boundary_band
    grouped = binary_dilation(error, iterations=max(1, radius // 2))
    labels, components = label(grouped)
    minimum_area = max(6, int(height * width * 0.00008))
    proposals: list[dict[str, Any]] = []
    for component_id in range(1, components + 1):
        component = labels == component_id
        actual = component & error
        area = int(actual.sum())
        if area < minimum_area:
            continue
        ys, xs = np.nonzero(component)
        margin = max(4, int(round(min(height, width) * 0.025)))
        x0 = max(int(xs.min()) - margin, 0)
        y0 = max(int(ys.min()) - margin, 0)
        x1 = min(int(xs.max()) + margin + 1, width)
        y1 = min(int(ys.max()) + margin + 1, height)
        min_width = max(14, int(width * 0.12))
        min_height = max(14, int(height * 0.12))
        if x1 - x0 < min_width:
            padding = min_width - (x1 - x0)
            x0 = max(x0 - padding // 2, 0)
            x1 = min(x0 + min_width, width)
            x0 = max(x1 - min_width, 0)
        if y1 - y0 < min_height:
            padding = min_height - (y1 - y0)
            y0 = max(y0 - padding // 2, 0)
            y1 = min(y0 + min_height, height)
            y0 = max(y1 - min_height, 0)
        false_positive = int((actual & coarse & ~target).sum())
        false_negative = int((actual & ~coarse & target).sum())
        if false_positive > 1.5 * false_negative:
            category = "Leakage"
        elif false_negative > 1.5 * false_positive:
            category = "Missing detail"
        else:
            category = "Boundary mismatch"
        proposals.append(
            {
                "box": (x0, y0, x1, y1),
                "area": area,
                "category": category,
                "false_positive": false_positive,
                "false_negative": false_negative,
            }
        )
    proposals.sort(key=lambda value: value["area"], reverse=True)
    selected: list[dict[str, Any]] = []
    for proposal in proposals:
        if all(
            box_overlap(proposal["box"], existing["box"]) < 0.18
            for existing in selected
        ):
            selected.append(proposal)
        if len(selected) >= count:
            break
    if selected:
        return selected
    fallback = mask_bbox(error)
    if fallback is None:
        fallback = mask_bbox(np.logical_xor(coarse, target))
    if fallback is None:
        fallback = mask_bbox(target)
    return [
        {
            "box": fallback or (0, 0, width, height),
            "area": int(error.sum()),
            "category": "Boundary mismatch",
            "false_positive": 0,
            "false_negative": 0,
        }
    ]


def clean_prompt(query: str, limit: int = 68) -> str:
    value = " ".join(query.replace("\n", " ").split())
    value = re.sub(
        r"^(please\s+)?segment\s+(the\s+)?",
        "",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(r"\s+in\s+this\s+image[.!?]*$", "", value, flags=re.IGNORECASE)
    if len(value) > limit:
        value = value[: limit - 3].rstrip() + "..."
    return value.strip(" .\"'")


def style_image_axis(axis: plt.Axes) -> None:
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_color(LINE)
        spine.set_linewidth(0.65)


def prompt_axis(axis: plt.Axes, prompt: str) -> None:
    axis.set_facecolor(WHITE)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.text(
        0.50,
        0.50,
        f'Prompt\n"{clean_prompt(prompt)}"',
        rotation=90,
        ha="center",
        va="center",
        fontsize=8.0,
        color=INK,
        fontweight="semibold",
        transform=axis.transAxes,
    )


def row_marker_axis(axis: plt.Axes, marker: str) -> None:
    axis.axis("off")
    axis.text(
        0.52,
        0.50,
        marker,
        ha="center",
        va="center",
        fontsize=10.5,
        fontweight="bold",
        color=INK,
        transform=axis.transAxes,
    )


def uncertain_grid_cells(
    uncertainty: np.ndarray,
    coarse: np.ndarray,
    max_cells: int = 32,
) -> list[tuple[int, int, int, int, float]]:
    """Select compact grid cells around the most uncertain mask boundaries."""
    uncertainty = np.clip(np.asarray(uncertainty, dtype=np.float64), 0.0, 1.0)
    coarse = np.asarray(coarse, dtype=bool)
    height, width = coarse.shape
    cell_size = max(8, int(round(min(height, width) / 24.0)))
    boundary = np.logical_xor(binary_dilation(coarse), binary_erosion(coarse))
    boundary_band = binary_dilation(boundary, iterations=max(2, cell_size // 2))
    proposals: list[tuple[int, int, int, int, float]] = []
    for y0 in range(0, height, cell_size):
        y1 = min(y0 + cell_size, height)
        for x0 in range(0, width, cell_size):
            x1 = min(x0 + cell_size, width)
            local_band = boundary_band[y0:y1, x0:x1]
            if float(local_band.mean()) < 0.08:
                continue
            local_uncertainty = uncertainty[y0:y1, x0:x1]
            score = float(local_uncertainty[local_band].mean())
            if score > 0.05:
                proposals.append((x0, y0, x1, y1, score))
    if not proposals:
        return []
    scores = np.asarray([cell[-1] for cell in proposals], dtype=np.float64)
    cutoff = max(0.16, float(np.quantile(scores, 0.58)))
    selected = [cell for cell in proposals if cell[-1] >= cutoff]
    selected.sort(key=lambda cell: cell[-1], reverse=True)
    return sorted(selected[:max_cells], key=lambda cell: (cell[1], cell[0]))


def draw_uncertain_grid(
    axis: plt.Axes,
    cells: list[tuple[int, int, int, int, float]],
) -> None:
    for x0, y0, x1, y1, score in cells:
        axis.add_patch(
            Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                facecolor="#FFD54A",
                edgecolor="#7A251F",
                linewidth=0.75,
                alpha=0.18 + 0.22 * score,
            )
        )


def draw_localization_row(
    figure: plt.Figure,
    grid: Any,
    sample: LoadedSample,
    threshold: float,
) -> None:
    prompt = sample.stamp.query or sample.pixellm_query
    prompt_axis(figure.add_subplot(grid[0, 0]), prompt)
    input_axis = figure.add_subplot(grid[0, 1])
    input_axis.imshow(sample.stamp.image)
    input_axis.set_title("Image", fontsize=9.2, color=INK, pad=4)
    style_image_axis(input_axis)

    heatmap_axis = figure.add_subplot(grid[0, 2])
    heatmap_axis.imshow(sample.stamp.image)
    heatmap_axis.imshow(
        sample.stamp.coarse_probability,
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        alpha=0.68,
    )
    heatmap_axis.set_title("Localization", fontsize=9.2, color=INK, pad=4)
    style_image_axis(heatmap_axis)

    box_axis = figure.add_subplot(grid[0, 3])
    box_axis.imshow(sample.stamp.image)
    predicted = mask_bbox(sample.stamp.coarse_probability >= threshold)
    if predicted is not None:
        x0, y0, x1, y1 = predicted
        box_axis.add_patch(
            Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                fill=False,
                edgecolor=GREEN,
                linewidth=2.2,
            )
        )
    box_axis.set_title("Box", fontsize=9.2, color=INK, pad=4)
    style_image_axis(box_axis)


def draw_output_row(
    figure: plt.Figure,
    grid: Any,
    row: int,
    marker: str,
    image: np.ndarray,
    probability: np.ndarray,
    uncertainty: np.ndarray,
    threshold: float,
) -> list[tuple[int, int, int, int, float]]:
    coarse = probability >= threshold
    cells = uncertain_grid_cells(uncertainty, coarse)
    row_marker_axis(figure.add_subplot(grid[row, 0]), marker)
    input_axis = figure.add_subplot(grid[row, 1])
    input_axis.imshow(image)
    if row == 1:
        input_axis.set_title("Image", fontsize=9.2, color=INK, pad=4)
    style_image_axis(input_axis)

    mask_axis = figure.add_subplot(grid[row, 2])
    mask_axis.imshow(overlay_mask(image, coarse, color=RED, alpha=0.40))
    contour(mask_axis, coarse, "#8E241E", 0.85)
    if row == 1:
        mask_axis.set_title("Coarse mask", fontsize=9.2, color=INK, pad=4)
    style_image_axis(mask_axis)

    uncertainty_axis = figure.add_subplot(grid[row, 3])
    uncertainty_axis.imshow(overlay_mask(image, coarse, color=RED, alpha=0.32))
    contour(uncertainty_axis, coarse, "#8E241E", 0.75)
    draw_uncertain_grid(uncertainty_axis, cells)
    if row == 1:
        uncertainty_axis.set_title("Uncertain patches", fontsize=9.2, color=INK, pad=4)
    style_image_axis(uncertainty_axis)
    return cells


def render_figure(
    sample: LoadedSample,
    output_dir: Path,
    stem: str,
    threshold: float,
    dpi: int,
) -> dict[str, list[tuple[int, int, int, int, float]]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(10.8, 9.0), facecolor=WHITE)
    grid = figure.add_gridspec(
        4,
        4,
        width_ratios=(0.11, 1.0, 1.0, 1.0),
        height_ratios=(1.0, 1.0, 1.0, 1.0),
        left=0.025,
        right=0.985,
        bottom=0.035,
        top=0.975,
        wspace=0.045,
        hspace=0.12,
    )
    draw_localization_row(figure, grid, sample, threshold)

    pixellm_image = resize_rgb(
        load_rgb(sample.pixellm_item.image), sample.pixellm_target.shape
    )
    cell_records: dict[str, list[tuple[int, int, int, int, float]]] = {}
    cell_records["a"] = draw_output_row(
        figure,
        grid,
        1,
        "(a)",
        pixellm_image,
        sample.pixellm_probability,
        probability_uncertainty(sample.pixellm_probability),
        sample.pixellm_item.threshold,
    )
    cell_records["b"] = draw_output_row(
        figure,
        grid,
        2,
        "(b)",
        sample.stamp.image,
        sample.stamp.coarse_probability,
        sample.stamp.uncertainty,
        threshold,
    )
    cell_records["c"] = draw_output_row(
        figure,
        grid,
        3,
        "(c)",
        sample.text4seg.image,
        sample.text4seg.coarse_probability,
        sample.text4seg.uncertainty,
        threshold,
    )

    upper_bottom = grid[0, 0].get_position(figure).y0
    lower_top = grid[1, 0].get_position(figure).y1
    separator_y = 0.5 * (upper_bottom + lower_top)
    separator = plt.Line2D(
        [0.035, 0.985],
        [separator_y, separator_y],
        transform=figure.transFigure,
        color=INK,
        linewidth=0.85,
    )
    figure.add_artist(separator)
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(
            output_dir / f"{stem}.{suffix}",
            dpi=dpi,
            facecolor=WHITE,
            bbox_inches="tight",
            pad_inches=0.05,
        )
    plt.close(figure)
    return cell_records


def save_contact_sheet(
    loaded: list[LoadedSample],
    output_path: Path,
    threshold: float,
    count: int,
    dpi: int,
) -> None:
    shown = loaded[:count]
    if not shown:
        return
    figure, axes = plt.subplots(
        len(shown),
        5,
        figsize=(13.5, max(2.25 * len(shown), 3.0)),
        squeeze=False,
        constrained_layout=True,
    )
    for row_index, sample in enumerate(shown):
        stamp_coarse = sample.stamp.coarse_probability >= threshold
        pixel_coarse = sample.pixellm_probability >= sample.pixellm_item.threshold
        images = (
            sample.stamp.image,
            sample.stamp.coarse_probability,
            overlay_mask(
                resize_rgb(
                    load_rgb(sample.pixellm_item.image),
                    sample.pixellm_target.shape,
                ),
                pixel_coarse,
            ),
            overlay_mask(sample.stamp.image, stamp_coarse),
            overlay_mask(
                sample.text4seg.image,
                sample.text4seg.coarse_probability >= threshold,
            ),
        )
        titles = (
            f"#{row_index + 1} ID {sample.candidate.instance_id}",
            f"Heatmap | box IoU {sample.bbox_iou:.2f}",
            "PixelLM coarse mask",
            "STAMP coarse mask",
            "Text4Seg coarse mask",
        )
        for column, (image, title) in enumerate(zip(images, titles)):
            axes[row_index, column].imshow(
                image,
                cmap="viridis" if column == 1 else None,
                vmin=0.0 if column == 1 else None,
                vmax=1.0 if column == 1 else None,
            )
            axes[row_index, column].set_title(title, fontsize=8)
            axes[row_index, column].axis("off")
    figure.suptitle(
        "Deterministically ranked paired candidates",
        fontsize=11,
        fontweight="bold",
    )
    figure.savefig(output_path, dpi=dpi, facecolor=WHITE, bbox_inches="tight")
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Select a paired STAMP/Text4Seg/PixelLM sample and render the FreeRef "
            "introduction motivation figure."
        )
    )
    parser.add_argument("--stamp-rows", type=Path, required=True)
    parser.add_argument("--text4seg-rows", type=Path, required=True)
    parser.add_argument("--pixellm-rows", type=Path, required=True)
    parser.add_argument("--pixellm-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stem", default="freeref_intro_motivation")
    parser.add_argument("--stamp-label", default="STAMP-7B")
    parser.add_argument("--sample-id", default="")
    parser.add_argument("--candidate-pool", type=int, default=48)
    parser.add_argument("--contact-sheet-count", type=int, default=8)
    parser.add_argument("--minimum-box-iou", type=float, default=0.50)
    parser.add_argument("--boundary-sigma", type=float, default=8.0)
    parser.add_argument("--dpi", type=int, default=300)
    add_refiner_arguments(parser)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.candidate_pool <= 0 or args.contact_sheet_count <= 0 or args.dpi <= 0:
        raise ValueError(
            "candidate-pool, contact-sheet-count, and dpi must be positive."
        )
    if not 0.0 <= args.minimum_box_iou <= 1.0:
        raise ValueError("minimum-box-iou must lie in [0, 1].")
    if args.boundary_sigma <= 0:
        raise ValueError("boundary-sigma must be positive.")
    args.stamp_rows = args.stamp_rows.expanduser().resolve()
    args.text4seg_rows = args.text4seg_rows.expanduser().resolve()
    args.pixellm_rows = args.pixellm_rows.expanduser().resolve()
    args.pixellm_manifest = args.pixellm_manifest.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    for path in (
        args.stamp_rows,
        args.text4seg_rows,
        args.pixellm_rows,
        args.pixellm_manifest,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    stamp_rows = read_csv(args.stamp_rows)
    text4seg_rows = read_csv(args.text4seg_rows)
    pixellm_rows = read_csv(args.pixellm_rows)
    paired = pair_candidates(stamp_rows, text4seg_rows, pixellm_rows)
    if args.sample_id:
        requested = (
            str(int(args.sample_id)) if args.sample_id.isdigit() else args.sample_id
        )
        paired = [
            candidate for candidate in paired if candidate.instance_id == requested
        ]
        if not paired:
            raise ValueError(f"No eligible paired sample has ID {requested!r}.")
    if not paired:
        raise RuntimeError(
            "No paired STAMP/Text4Seg/PixelLM rows passed the semantic-localization filter."
        )

    manifest = manifest_index(args.pixellm_manifest)
    refiner = build_refiner(args)
    loaded: list[LoadedSample] = []
    failures: list[dict[str, str]] = []
    for candidate in paired[: args.candidate_pool]:
        try:
            sample = load_candidate(
                candidate,
                args.stamp_rows,
                args.text4seg_rows,
                manifest,
                refiner,
                args.stamp_label,
                args.threshold,
                args.minimum_box_iou,
                args.boundary_sigma,
            )
            if sample.bbox_iou < args.minimum_box_iou:
                continue
            loaded.append(sample)
        except Exception as error:
            failures.append(
                {
                    "instance_id": candidate.instance_id,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
    if not loaded:
        raise RuntimeError(
            "Every paired candidate failed to load or pass the localization filter. "
            + json.dumps(failures[:5], ensure_ascii=False)
        )
    loaded.sort(key=lambda sample: sample.score, reverse=True)
    selected = loaded[0]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_contact_sheet(
        loaded,
        args.output_dir / "intro_candidate_contact_sheet.png",
        args.threshold,
        args.contact_sheet_count,
        min(args.dpi, 180),
    )
    uncertainty_cells = render_figure(
        selected,
        args.output_dir,
        args.stem,
        args.threshold,
        args.dpi,
    )
    candidate_rows = [
        {
            "rank": rank,
            "instance_id": sample.candidate.instance_id,
            "score": sample.score,
            "bbox_iou": sample.bbox_iou,
            "object_fraction": sample.object_fraction,
            "stamp_coarse_iou": parse_float(sample.candidate.stamp_row, "coarse_iou"),
            "stamp_coarse_boundary_iou": parse_float(
                sample.candidate.stamp_row, "coarse_boundary_iou"
            ),
            "text4seg_coarse_iou": parse_float(
                sample.candidate.text4seg_row, "coarse_iou"
            ),
            "text4seg_coarse_boundary_iou": parse_float(
                sample.candidate.text4seg_row, "coarse_boundary_iou"
            ),
            "pixellm_coarse_iou": parse_float(
                sample.candidate.pixellm_row, "coarse_iou"
            ),
            "pixellm_coarse_boundary_iou": parse_float(
                sample.candidate.pixellm_row, "coarse_boundary_iou"
            ),
        }
        for rank, sample in enumerate(loaded, start=1)
    ]
    with (args.output_dir / "intro_candidates.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(candidate_rows[0]))
        writer.writeheader()
        writer.writerows(candidate_rows)
    if failures:
        (args.output_dir / "candidate_load_failures.json").write_text(
            json.dumps(failures, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    manifest_value = {
        "real_experiment_data": True,
        "selected_instance_id": selected.candidate.instance_id,
        "query": selected.stamp.query or selected.pixellm_query,
        "stamp_rows": str(args.stamp_rows),
        "text4seg_rows": str(args.text4seg_rows),
        "pixellm_rows": str(args.pixellm_rows),
        "pixellm_manifest": str(args.pixellm_manifest),
        "selection_score": selected.score,
        "predicted_box_iou": selected.bbox_iou,
        "object_fraction": selected.object_fraction,
        "stamp_metrics": selected.candidate.stamp_row,
        "text4seg_metrics": selected.candidate.text4seg_row,
        "pixellm_metrics": selected.candidate.pixellm_row,
        "uncertainty_cells": uncertainty_cells,
        "note": (
            "Ground truth is used only for deterministic candidate ranking, "
            "and box-IoU verification. The heatmap, coarse masks, and "
            "uncertainty grids are computed only from model outputs. "
            "Rows (a), (b), and (c) correspond to PixelLM, STAMP, and Text4Seg."
        ),
    }
    (args.output_dir / "intro_figure_manifest.json").write_text(
        json.dumps(manifest_value, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "selected_instance_id": selected.candidate.instance_id,
                "query": selected.stamp.query or selected.pixellm_query,
                "bbox_iou": selected.bbox_iou,
                "figure_png": str(args.output_dir / f"{args.stem}.png"),
                "figure_pdf": str(args.output_dir / f"{args.stem}.pdf"),
                "figure_svg": str(args.output_dir / f"{args.stem}.svg"),
                "contact_sheet": str(
                    args.output_dir / "intro_candidate_contact_sheet.png"
                ),
                "manifest": str(args.output_dir / "intro_figure_manifest.json"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
