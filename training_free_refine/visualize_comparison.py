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
import torch
from matplotlib.patches import Patch
from PIL import Image
from skimage.segmentation import mark_boundaries
from tqdm import tqdm

from .eval_stamp_dumps import boundary_iou, mask_iou, resolve_asset
from .eval_text4seg_outputs import load_mask, load_rgb, resize_float, resize_mask
from .refiner import TrainingFreeRefineConfig, TrainingFreeUncertaintyRefiner


@dataclass
class SampleView:
    name: str
    label: str
    query: str
    image: np.ndarray
    target: np.ndarray
    coarse_probability: np.ndarray
    refined_probability: np.ndarray
    graph_probability: np.ndarray
    uncertainty: np.ndarray
    superpixels: np.ndarray
    diagnostics: dict[str, float]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def resolve_row_path(value: str, rows_path: Path) -> Path:
    path = Path(value)
    candidates = [path, rows_path.parent / path, Path.cwd() / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not resolve {value!r} from {rows_path}.")


def resize_rgb(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if image.shape[:2] == shape:
        return image
    return np.asarray(Image.fromarray(image).resize((shape[1], shape[0]), Image.Resampling.BILINEAR))


def resize_labels(labels: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if labels.shape == shape:
        return labels.astype(np.int64)
    image = Image.fromarray(labels.astype(np.int32), mode="I")
    return np.asarray(image.resize((shape[1], shape[0]), Image.Resampling.NEAREST), dtype=np.int64)


def build_refiner(args: argparse.Namespace) -> TrainingFreeUncertaintyRefiner:
    return TrainingFreeUncertaintyRefiner(
        TrainingFreeRefineConfig(
            n_segments=args.n_segments,
            compactness=args.compactness,
            slic_sigma=args.slic_sigma,
            graph_lambda=args.graph_lambda,
            confidence_power=args.confidence_power,
            fusion_power=args.fusion_power,
            foreground_seed=args.foreground_seed,
            background_seed=args.background_seed,
            seed_strength=args.seed_strength,
            threshold=args.threshold,
        )
    )


def load_stamp_view(
    row: dict[str, str], rows_path: Path, label: str, refiner: TrainingFreeUncertaintyRefiner
) -> SampleView:
    dump_path = resolve_row_path(row["dump"], rows_path)
    payload = torch.load(dump_path, map_location="cpu", weights_only=False)
    image_path = resolve_asset(payload["image_path"], dump_path)
    mask_path = resolve_asset(payload["mask_path"], dump_path)
    image = np.asarray(Image.open(image_path).convert("RGB"))
    target = np.asarray(Image.open(mask_path).convert("L")) > 0
    image = resize_rgb(image, target.shape)
    output = refiner.refine(image, payload["mask_logits"], tuple(payload["grid_hw"]))
    return SampleView(
        name=str(payload.get("name", dump_path.stem)),
        label=label,
        query=str(payload.get("query", "")),
        image=image,
        target=target,
        coarse_probability=output["coarse_probability"].numpy(),
        refined_probability=output["refined_probability"].numpy(),
        graph_probability=output["graph_probability"].numpy(),
        uncertainty=output["uncertainty_map"].numpy(),
        superpixels=output["superpixels"].numpy(),
        diagnostics=dict(output["diagnostics"]),
    )


def load_text4seg_view(
    row: dict[str, str],
    rows_path: Path,
    label: str,
    refiner: TrainingFreeUncertaintyRefiner,
    boundary_sigma: float,
) -> SampleView:
    image_path = resolve_row_path(row["image"], rows_path)
    target_path = resolve_row_path(row["gt_mask"], rows_path)
    pred_path = resolve_row_path(row["pred_mask"], rows_path)
    image = load_rgb(image_path)
    target = load_mask(target_path)
    coarse_at_image = resize_mask(load_mask(pred_path), image.shape[:2])
    output = refiner.refine_hard_mask(image, coarse_at_image, boundary_sigma=boundary_sigma)
    target_shape = target.shape
    return SampleView(
        name=str(row.get("name") or pred_path.stem),
        label=label,
        query=str(row.get("query") or ""),
        image=resize_rgb(image, target_shape),
        target=target,
        coarse_probability=resize_float(output["coarse_probability"].numpy(), target_shape),
        refined_probability=resize_float(output["refined_probability"].numpy(), target_shape),
        graph_probability=resize_float(output["graph_probability"].numpy(), target_shape),
        uncertainty=resize_float(output["uncertainty_map"].numpy(), target_shape),
        superpixels=resize_labels(output["superpixels"].numpy(), target_shape),
        diagnostics=dict(output["diagnostics"]),
    )


def load_view(
    kind: str,
    row: dict[str, str],
    rows_path: Path,
    label: str,
    refiner: TrainingFreeUncertaintyRefiner,
    boundary_sigma: float,
) -> SampleView:
    if kind == "stamp":
        return load_stamp_view(row, rows_path, label, refiner)
    return load_text4seg_view(row, rows_path, label, refiner, boundary_sigma)


def view_metrics(view: SampleView, threshold: float, boundary_tolerance: int) -> dict[str, float | int]:
    coarse = view.coarse_probability >= threshold
    refined = view.refined_probability >= threshold
    target = view.target.astype(bool)
    coarse_iou, coarse_inter, coarse_union = mask_iou(coarse, target)
    refined_iou, refined_inter, refined_union = mask_iou(refined, target)
    coarse_correct = coarse == target
    refined_correct = refined == target
    changed = coarse != refined
    corrected = (~coarse_correct) & refined_correct
    regressed = coarse_correct & (~refined_correct)
    persistent_error = (~coarse_correct) & (~refined_correct)
    boundary_before = boundary_iou(coarse, target, boundary_tolerance)
    boundary_after = boundary_iou(refined, target, boundary_tolerance)
    return {
        "coarse_iou": coarse_iou,
        "refined_iou": refined_iou,
        "iou_delta": refined_iou - coarse_iou,
        "coarse_intersection": coarse_inter,
        "coarse_union": coarse_union,
        "refined_intersection": refined_inter,
        "refined_union": refined_union,
        "coarse_boundary_iou": boundary_before,
        "refined_boundary_iou": boundary_after,
        "boundary_iou_delta": boundary_after - boundary_before,
        "object_fraction": float(target.mean()),
        "uncertainty_mean": float(view.uncertainty.mean()),
        "uncertainty_q90": float(np.quantile(view.uncertainty, 0.9)),
        "uncertain_fraction": float((view.uncertainty >= 0.5).mean()),
        "changed_fraction": float(changed.mean()),
        "corrected_fraction": float(corrected.mean()),
        "regressed_fraction": float(regressed.mean()),
        "persistent_error_fraction": float(persistent_error.mean()),
        "uncertainty_on_changed": float(view.uncertainty[changed].mean()) if changed.any() else 0.0,
        "uncertainty_on_corrected": float(view.uncertainty[corrected].mean()) if corrected.any() else 0.0,
    }


def overlay(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.46) -> np.ndarray:
    result = image.astype(np.float32).copy()
    selected = mask.astype(bool)
    result[selected] = result[selected] * (1.0 - alpha) + np.asarray(color) * alpha
    return np.clip(result, 0, 255).astype(np.uint8)


def transition_map(view: SampleView, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    coarse = view.coarse_probability >= threshold
    refined = view.refined_probability >= threshold
    target = view.target.astype(bool)
    coarse_correct = coarse == target
    refined_correct = refined == target
    state = np.zeros(target.shape, dtype=np.uint8)
    state[(~coarse_correct) & refined_correct] = 1
    state[coarse_correct & (~refined_correct)] = 2
    state[(~coarse_correct) & (~refined_correct)] = 3
    image = (view.image.astype(np.float32) * 0.28).astype(np.uint8)
    colors = np.asarray([[0, 0, 0], [35, 176, 92], [220, 63, 67], [238, 164, 54]], dtype=np.uint8)
    active = state > 0
    image[active] = colors[state[active]]
    return image, state


def safe_name(value: str) -> str:
    cleaned = "".join(character if character.isalnum() or character in "-_" else "_" for character in value)
    return cleaned[:100] or "sample"


def save_sample_panel(
    view: SampleView,
    metrics: dict[str, float | int],
    output_path: Path,
    threshold: float,
    category: str,
) -> None:
    coarse = view.coarse_probability >= threshold
    refined = view.refined_probability >= threshold
    transition, _ = transition_map(view, threshold)
    slic_overlay = mark_boundaries(view.image / 255.0, view.superpixels, color=(1.0, 1.0, 1.0), mode="thick")
    figure, axes = plt.subplots(2, 4, figsize=(17.2, 9.2), constrained_layout=True)
    panels = [
        (view.image, "Input image", None, None),
        (overlay(view.image, view.target, (34, 184, 111)), "Ground truth", None, None),
        (
            overlay(view.image, coarse, (224, 78, 65)),
            f"Original {view.label}\nIoU {float(metrics['coarse_iou']):.3f}",
            None,
            None,
        ),
        (
            overlay(view.image, refined, (42, 112, 213)),
            f"+ Training-Free\nIoU {float(metrics['refined_iou']):.3f}",
            None,
            None,
        ),
        (view.coarse_probability, "Original foreground probability", "viridis", (0.0, 1.0)),
        (view.uncertainty, "Uncertainty: where refinement acts", "magma", (0.0, 1.0)),
        (slic_overlay, "Image-aware superpixel graph", None, None),
        (transition, "Pixel outcome after refinement", None, None),
    ]
    for axis, (values, title, cmap, limits) in zip(axes.flat, panels):
        image_artist = axis.imshow(values, cmap=cmap, vmin=limits[0] if limits else None, vmax=limits[1] if limits else None)
        axis.set_title(title, fontsize=12, pad=8)
        axis.axis("off")
        if cmap is not None:
            figure.colorbar(image_artist, ax=axis, fraction=0.046, pad=0.025)
    axes[1, 3].legend(
        handles=[
            Patch(facecolor="#23b05c", label="Corrected"),
            Patch(facecolor="#dc3f43", label="New error"),
            Patch(facecolor="#eea436", label="Persistent error"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=3,
        frameon=False,
        fontsize=9,
    )
    query_text = " ".join(view.query.split())
    if len(query_text) > 96:
        query_text = query_text[:93] + "..."
    query = f" | Query: {query_text}" if query_text else ""
    figure.suptitle(
        f"{view.label} vs Training-Free | {category} | {view.name}{query}\n"
        f"IoU delta {float(metrics['iou_delta']):+.3f} | Boundary delta "
        f"{float(metrics['boundary_iou_delta']):+.3f} | Uncertain area "
        f"{100.0 * float(metrics['uncertain_fraction']):.1f}% | Changed area "
        f"{100.0 * float(metrics['changed_fraction']):.1f}%",
        fontsize=14,
        fontweight="bold",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def numeric(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=np.float64)


def correlation(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def save_run_overview(rows: list[dict[str, Any]], label: str, output_dir: Path) -> None:
    delta = numeric(rows, "iou_delta")
    coarse = numeric(rows, "coarse_iou")
    uncertainty = numeric(rows, "uncertain_fraction")
    object_fraction = numeric(rows, "object_fraction")
    boundary_delta = numeric(rows, "boundary_iou_delta")
    figure, axes = plt.subplots(2, 3, figsize=(16.5, 9.4), constrained_layout=True)
    axes[0, 0].hist(delta, bins=45, color="#3274a1", alpha=0.88)
    axes[0, 0].axvline(0.0, color="#bf3f3f", linewidth=1.5)
    axes[0, 0].set(title="Per-sample IoU change", xlabel="Refined IoU - original IoU", ylabel="Samples")
    scatter = axes[0, 1].scatter(coarse, delta, c=uncertainty, cmap="viridis", s=11, alpha=0.58)
    axes[0, 1].axhline(0.0, color="#777777", linewidth=1)
    axes[0, 1].set(title="Baseline quality vs gain", xlabel="Original IoU", ylabel="IoU change")
    figure.colorbar(scatter, ax=axes[0, 1], label="Uncertain fraction")
    axes[0, 2].scatter(uncertainty, delta, color="#8d5aa7", s=11, alpha=0.55)
    axes[0, 2].axhline(0.0, color="#777777", linewidth=1)
    axes[0, 2].set(title="Does uncertainty predict gain?", xlabel="Uncertain fraction", ylabel="IoU change")
    axes[1, 0].scatter(object_fraction, delta, color="#3a9d72", s=11, alpha=0.55)
    axes[1, 0].axhline(0.0, color="#777777", linewidth=1)
    axes[1, 0].set(title="Object size vs gain", xlabel="GT foreground fraction", ylabel="IoU change")
    axes[1, 1].hist(boundary_delta, bins=45, color="#d28b32", alpha=0.88)
    axes[1, 1].axvline(0.0, color="#bf3f3f", linewidth=1.5)
    axes[1, 1].set(title="Boundary IoU change", xlabel="Boundary IoU change", ylabel="Samples")
    counts = [int((delta > 1e-12).sum()), int((delta < -1e-12).sum()), int((np.abs(delta) <= 1e-12).sum())]
    bars = axes[1, 2].bar(["Improved", "Degraded", "Unchanged"], counts, color=["#2f9e64", "#cf4d4d", "#999999"])
    axes[1, 2].bar_label(bars, labels=[f"{value}\n({100 * value / len(rows):.1f}%)" for value in counts], padding=3)
    axes[1, 2].set(title="Outcome stability", ylabel="Samples")
    figure.suptitle(
        f"{label}: Training-Free refinement analysis | N={len(rows)} | mean IoU delta {delta.mean():+.4f}",
        fontsize=15,
        fontweight="bold",
    )
    for axis in axes.flat:
        axis.grid(alpha=0.2, linewidth=0.6)
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"run_overview.{suffix}", dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def select_rows(rows: list[dict[str, Any]], count: int) -> dict[str, list[dict[str, Any]]]:
    if count <= 0:
        return {}
    delta = numeric(rows, "iou_delta")
    median = float(np.median(delta))
    selectors = {
        "strongest_improvements": sorted(
            (row for row in rows if float(row["iou_delta"]) > 1e-12),
            key=lambda row: float(row["iou_delta"]),
            reverse=True,
        ),
        "strongest_degradations": sorted(
            (row for row in rows if float(row["iou_delta"]) < -1e-12),
            key=lambda row: float(row["iou_delta"]),
        ),
        "highest_uncertainty": sorted(rows, key=lambda row: float(row["uncertain_fraction"]), reverse=True),
        "representative": sorted(rows, key=lambda row: abs(float(row["iou_delta"]) - median)),
    }
    selected: dict[str, list[dict[str, Any]]] = {}
    used: set[int] = set()
    for category, candidates in selectors.items():
        values = []
        for candidate in candidates:
            index = int(candidate["source_row_index"])
            if index in used:
                continue
            values.append(candidate)
            used.add(index)
            if len(values) == count:
                break
        selected[category] = values
    return selected


def summarize_analysis(rows: list[dict[str, Any]], label: str, kind: str) -> dict[str, Any]:
    delta = numeric(rows, "iou_delta")
    coarse = numeric(rows, "coarse_iou")
    refined = numeric(rows, "refined_iou")
    uncertain = numeric(rows, "uncertain_fraction")
    boundary_delta = numeric(rows, "boundary_iou_delta")
    corrected = numeric(rows, "corrected_fraction")
    regressed = numeric(rows, "regressed_fraction")
    coarse_ciou = sum(int(row["coarse_intersection"]) for row in rows) / max(
        sum(int(row["coarse_union"]) for row in rows), 1
    )
    refined_ciou = sum(int(row["refined_intersection"]) for row in rows) / max(
        sum(int(row["refined_union"]) for row in rows), 1
    )
    return {
        "label": label,
        "kind": kind,
        "samples": len(rows),
        "coarse_mean_iou": float(coarse.mean()),
        "refined_mean_iou": float(refined.mean()),
        "mean_iou_delta": float(delta.mean()),
        "coarse_ciou": coarse_ciou,
        "refined_ciou": refined_ciou,
        "ciou_delta": refined_ciou - coarse_ciou,
        "mean_boundary_iou_delta": float(boundary_delta.mean()),
        "improved_samples": int((delta > 1e-12).sum()),
        "degraded_samples": int((delta < -1e-12).sum()),
        "unchanged_samples": int((np.abs(delta) <= 1e-12).sum()),
        "mean_uncertain_fraction": float(uncertain.mean()),
        "mean_corrected_fraction": float(corrected.mean()),
        "mean_regressed_fraction": float(regressed.mean()),
        "correlation_coarse_iou_vs_delta": correlation(coarse, delta),
        "correlation_uncertainty_vs_delta": correlation(uncertain, delta),
        "correlation_object_fraction_vs_delta": correlation(numeric(rows, "object_fraction"), delta),
    }


def write_run_markdown(summary: dict[str, Any], output_path: Path) -> None:
    samples = int(summary["samples"])
    improved = int(summary["improved_samples"])
    degraded = int(summary["degraded_samples"])
    output_path.write_text(
        "\n".join(
            [
                f"# {summary['label']} Training-Free Visualization Analysis",
                "",
                f"- Samples: {samples}",
                f"- mIoU: {summary['coarse_mean_iou']:.4f} -> {summary['refined_mean_iou']:.4f} "
                f"({summary['mean_iou_delta']:+.4f})",
                f"- cIoU: {summary['coarse_ciou']:.4f} -> {summary['refined_ciou']:.4f} "
                f"({summary['ciou_delta']:+.4f})",
                f"- Mean Boundary IoU delta: {summary['mean_boundary_iou_delta']:+.4f}",
                f"- Improved: {improved}/{samples} ({100 * improved / samples:.2f}%)",
                f"- Degraded: {degraded}/{samples} ({100 * degraded / samples:.2f}%)",
                f"- Mean uncertain area: {100 * summary['mean_uncertain_fraction']:.2f}%",
                f"- Mean corrected/regressed pixels: {100 * summary['mean_corrected_fraction']:.3f}% / "
                f"{100 * summary['mean_regressed_fraction']:.3f}%",
                f"- Corr(original IoU, delta): {summary['correlation_coarse_iou_vs_delta']:+.3f}",
                f"- Corr(uncertain fraction, delta): {summary['correlation_uncertainty_vs_delta']:+.3f}",
                f"- Corr(object fraction, delta): {summary['correlation_object_fraction_vs_delta']:+.3f}",
                "",
                "Sample selection uses GT-derived IoU only for post-hoc visualization analysis; GT is never used by the refiner.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def run_analysis(args: argparse.Namespace) -> int:
    rows_path = args.rows.resolve()
    source_rows = read_csv(rows_path)
    if args.limit:
        source_rows = source_rows[: args.limit]
    if not source_rows:
        raise ValueError(f"No rows found in {rows_path}.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    refiner = build_refiner(args)
    analysis_rows: list[dict[str, Any]] = []
    for index, row in enumerate(tqdm(source_rows, desc=f"analyze {args.label}", dynamic_ncols=True)):
        view = load_view(args.kind, row, rows_path, args.label, refiner, args.boundary_sigma)
        metrics = view_metrics(view, args.threshold, args.boundary_tolerance)
        analysis_rows.append(
            {
                "source_row_index": index,
                "name": view.name,
                **{key: value for key, value in row.items() if key not in metrics},
                **metrics,
            }
        )
    analysis_path = args.output_dir / "visual_analysis_rows.csv"
    write_csv(analysis_path, analysis_rows)
    summary = summarize_analysis(analysis_rows, args.label, args.kind)
    (args.output_dir / "visual_analysis_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    write_run_markdown(summary, args.output_dir / "visual_analysis_summary.md")
    save_run_overview(analysis_rows, args.label, args.output_dir)
    for category, selected in select_rows(analysis_rows, args.panels_per_group).items():
        for rank, analysis_row in enumerate(selected, start=1):
            source_row = source_rows[int(analysis_row["source_row_index"])]
            view = load_view(args.kind, source_row, rows_path, args.label, refiner, args.boundary_sigma)
            output_path = args.output_dir / "sample_panels" / category / (
                f"{rank:02d}_{safe_name(view.name)}.png"
            )
            save_sample_panel(view, analysis_row, output_path, args.threshold, category.replace("_", " "))
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


def parse_run_spec(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--run must use LABEL=/path/to/visual_analysis_rows.csv")
    label, path = value.split("=", 1)
    return label.strip(), Path(path).expanduser()


def save_cross_model_figure(runs: list[tuple[str, list[dict[str, str]]]], output_dir: Path) -> list[dict[str, Any]]:
    summaries = [summarize_analysis(rows, label, "comparison") for label, rows in runs]
    labels = [item["label"] for item in summaries]
    x = np.arange(len(labels))
    width = 0.34
    figure, axes = plt.subplots(2, 2, figsize=(13.8, 9.5), constrained_layout=True)
    for axis, coarse_key, refined_key, title in [
        (axes[0, 0], "coarse_mean_iou", "refined_mean_iou", "Mean IoU"),
        (axes[0, 1], "coarse_ciou", "refined_ciou", "Cumulative IoU"),
    ]:
        coarse_values = [100 * float(item[coarse_key]) for item in summaries]
        refined_values = [100 * float(item[refined_key]) for item in summaries]
        first = axis.bar(x - width / 2, coarse_values, width, label="Original", color="#d36a5a")
        second = axis.bar(x + width / 2, refined_values, width, label="+ Training-Free", color="#3978b8")
        axis.bar_label(first, fmt="%.2f", padding=2, fontsize=9)
        axis.bar_label(second, fmt="%.2f", padding=2, fontsize=9)
        axis.set_xticks(x, labels)
        axis.set_ylabel("Score (%)")
        axis.set_title(title)
        axis.legend(frameon=False)
        axis.grid(axis="y", alpha=0.2)
    distributions = [numeric(rows, "iou_delta") for _, rows in runs]
    violin = axes[1, 0].violinplot(distributions, positions=x, showmeans=True, showextrema=False)
    for body in violin["bodies"]:
        body.set_facecolor("#6f63a7")
        body.set_alpha(0.65)
    axes[1, 0].axhline(0.0, color="#777777", linewidth=1)
    axes[1, 0].set_xticks(x, labels)
    axes[1, 0].set(title="Per-sample IoU change distribution", ylabel="IoU change")
    corrected = [100 * float(item["mean_corrected_fraction"]) for item in summaries]
    regressed = [100 * float(item["mean_regressed_fraction"]) for item in summaries]
    first = axes[1, 1].bar(x - width / 2, corrected, width, label="Corrected pixels", color="#2f9e64")
    second = axes[1, 1].bar(x + width / 2, regressed, width, label="New errors", color="#cf4d4d")
    axes[1, 1].bar_label(first, fmt="%.2f%%", padding=2, fontsize=9)
    axes[1, 1].bar_label(second, fmt="%.2f%%", padding=2, fontsize=9)
    axes[1, 1].set_xticks(x, labels)
    axes[1, 1].set(title="Where pixels changed", ylabel="Image pixels (%)")
    axes[1, 1].legend(frameon=False)
    axes[1, 1].grid(axis="y", alpha=0.2)
    figure.suptitle("Original methods vs Training-Free refinement", fontsize=15, fontweight="bold")
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"cross_model_comparison.{suffix}", dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return summaries


def compare_runs(args: argparse.Namespace) -> int:
    runs = [(label, read_csv(path.resolve())) for label, path in args.run]
    if len(runs) < 2:
        raise ValueError("At least two --run inputs are required.")
    summaries = save_cross_model_figure(runs, args.output_dir)
    (args.output_dir / "cross_model_summary.json").write_text(
        json.dumps({"runs": summaries}, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    lines = [
        "# Cross-Model Training-Free Comparison",
        "",
        "| Model | Samples | Original mIoU | Refined mIoU | Delta | Original cIoU | Refined cIoU | Delta |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        lines.append(
            f"| {item['label']} | {item['samples']} | {item['coarse_mean_iou']:.4f} | "
            f"{item['refined_mean_iou']:.4f} | {item['mean_iou_delta']:+.4f} | "
            f"{item['coarse_ciou']:.4f} | {item['refined_ciou']:.4f} | {item['ciou_delta']:+.4f} |"
        )
    (args.output_dir / "cross_model_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"runs": summaries}, indent=2, ensure_ascii=True))
    return 0


def add_refiner_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--n-segments", type=int, default=1024)
    parser.add_argument("--compactness", type=float, default=10.0)
    parser.add_argument("--slic-sigma", type=float, default=1.0)
    parser.add_argument("--graph-lambda", type=float, default=1.0)
    parser.add_argument("--confidence-power", type=float, default=2.0)
    parser.add_argument("--fusion-power", type=float, default=1.0)
    parser.add_argument("--foreground-seed", type=float, default=0.9)
    parser.add_argument("--background-seed", type=float, default=0.1)
    parser.add_argument("--seed-strength", type=float, default=50.0)
    parser.add_argument("--threshold", type=float, default=0.5)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Publication-quality Training-Free comparison visualizations.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="Analyze and visualize one original-method run.")
    run.add_argument("--kind", choices=["stamp", "text4seg"], required=True)
    run.add_argument("--rows", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--label", required=True)
    run.add_argument("--limit", type=int, default=0)
    run.add_argument("--panels-per-group", type=int, default=3)
    run.add_argument("--boundary-sigma", type=float, default=8.0)
    run.add_argument("--boundary-tolerance", type=int, default=2)
    add_refiner_arguments(run)
    compare = subparsers.add_parser("compare", help="Compare two or more analyzed runs.")
    compare.add_argument("--run", type=parse_run_spec, action="append", required=True)
    compare.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "run":
        if args.limit < 0 or args.panels_per_group < 0:
            raise ValueError("limit and panels-per-group must be non-negative.")
        return run_analysis(args)
    return compare_runs(args)


if __name__ == "__main__":
    raise SystemExit(main())
