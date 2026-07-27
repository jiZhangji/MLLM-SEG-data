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
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion
from tqdm import tqdm

from paper_assets.intro_figure.generate_intro_motivation_figure import (
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
    parser.add_argument("--candidate-pool", type=int, default=96)
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


def save_panels(root: Path, sample_id: str, names: list[str], panels: list[np.ndarray]) -> None:
    output = root / f"sample_{int(sample_id):06d}"
    output.mkdir(parents=True, exist_ok=True)
    for name, panel in zip(names, panels):
        Image.fromarray(np.asarray(panel, dtype=np.uint8)).save(output / f"{name}.png")


def main_table_rows(args: argparse.Namespace, refiner: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    stamp_rows = read_csv(args.stamp_rows)
    text_rows = read_csv(args.text4seg_rows)
    pixel_rows = read_csv(args.pixellm_rows)
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
        rows.append({"sample_id": sample.candidate.instance_id, "prompt": query, "panels": panels})
        record = {"sample_id": sample.candidate.instance_id, "prompt": query, "metrics": row_metrics}
        records.append(record)
        save_panels(args.output_dir / "main_table_panels", sample.candidate.instance_id, panel_names, panels)
    return rows, records


def postprocess_score(row: dict[str, str]) -> float:
    competitors = ("densecrf", "guided_filter", "fast_bilateral_solver", "slic_average")
    free_iou = float(row.get("freeref_iou", 0.0))
    free_boundary = float(row.get("freeref_boundary_iou", 0.0))
    best_iou = max(float(row.get(f"{name}_iou", 0.0)) for name in competitors)
    best_boundary = max(float(row.get(f"{name}_boundary_iou", 0.0)) for name in competitors)
    base_iou = float(row.get("base_iou", 0.0))
    base_boundary = float(row.get("base_boundary_iou", 0.0))
    if not 0.01 <= base_iou <= 0.98:
        return -1e9
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
        candidate_ids = sorted(
            stamp_by_id.keys() & metric_by_id.keys(),
            key=lambda value: postprocess_score(metric_by_id[value]),
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
        rows.append({"sample_id": sample_id, "prompt": view.query, "panels": panels})
        records.append(
            {
                "sample_id": sample_id,
                "prompt": view.query,
                "selection_score": postprocess_score(metric_by_id.get(sample_id, {})),
                "metrics": row_metrics,
            }
        )
        save_panels(args.output_dir / "postprocess_panels", sample_id, titles, panels)
    return rows, records


def flatten_records(records: list[dict[str, Any]], path: Path) -> None:
    rows = []
    for record in records:
        row: dict[str, Any] = {
            "sample_id": record["sample_id"],
            "prompt": record["prompt"],
        }
        if "selection_score" in record:
            row["selection_score"] = record["selection_score"]
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
        or args.candidate_pool <= 0
        or args.dpi <= 0
    ):
        raise ValueError(
            "sample-count, rows-per-page, candidate-pool, and dpi must be positive."
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
    main_pages = save_grid_pages(
        main_rows,
        main_titles,
        args.output_dir / "main_table_qualitative",
        args.dpi,
        {3, 5, 7},
        args.rows_per_page,
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
    post_pages = save_grid_pages(
        post_rows,
        post_titles,
        args.output_dir / "postprocess_qualitative",
        args.dpi,
        {7},
        args.rows_per_page,
    )
    flatten_records(post_records, args.output_dir / "postprocess_qualitative_rows.csv")

    manifest = {
        "main_table": main_records,
        "postprocess": post_records,
        "main_table_pages": main_pages,
        "postprocess_pages": post_pages,
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
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
