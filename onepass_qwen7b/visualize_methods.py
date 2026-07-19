from __future__ import annotations

import argparse
import csv
import json
import re
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.patches import Patch
from PIL import Image

from .checkpoint import load_checkpoint, read_checkpoint_config
from .data import OnePass7BDataset, OnePass7BSample, prepare_batch
from .runtime import configure_cudnn, load_base_onepass_model


@dataclass(frozen=True)
class RunSpec:
    label: str
    checkpoint: Path
    rows_csv: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare three OnePass-7B checkpoints with IoU plots and mask overlays."
    )
    parser.add_argument(
        "--run",
        action="append",
        nargs=3,
        metavar=("LABEL", "CHECKPOINT", "ROWS_CSV"),
        required=True,
        help="Repeat exactly three times in training order.",
    )
    parser.add_argument("--stamp-code-dir", required=True, type=Path)
    parser.add_argument("--base-model", type=Path)
    parser.add_argument("--eval-json", required=True, type=Path, nargs="+")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--indices", type=int, nargs="*")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"))
    parser.add_argument("--attn-implementation")
    parser.add_argument("--min-pixels", type=int)
    parser.add_argument("--max-pixels", type=int)
    parser.add_argument("--disable-cudnn", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--allow-existing-segmentation-tokens", action="store_true")
    return parser.parse_args()


def slug(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("_")
    return result or "method"


def load_iou_rows(path: Path) -> dict[int, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: dict[int, dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            index = int(row["index"])
            rows[index] = {
                **row,
                "index": index,
                "iou": float(row["iou"]),
            }
    if not rows:
        raise ValueError(f"No evaluation rows found in {path}.")
    return rows


def common_indices(rows_by_run: list[dict[int, dict[str, Any]]]) -> list[int]:
    values = set(rows_by_run[0])
    for rows in rows_by_run[1:]:
        values &= set(rows)
    if not values:
        raise ValueError("The evaluation CSV files have no common sample indices.")
    return sorted(values)


def select_cases(
    rows_by_run: list[dict[int, dict[str, Any]]],
    count: int,
) -> list[dict[str, Any]]:
    if len(rows_by_run) != 3:
        raise ValueError("Case selection expects exactly three runs.")
    indices = common_indices(rows_by_run)
    values = {
        index: np.asarray([rows[index]["iou"] for rows in rows_by_run], dtype=np.float64)
        for index in indices
    }
    quota = max(1, count // 6)
    rankings = [
        ("all_methods_fail", sorted(indices, key=lambda i: float(values[i].mean()))),
        (
            "stamp_lora_gain",
            sorted(indices, key=lambda i: float(values[i][1] - values[i][0]), reverse=True),
        ),
        (
            "seg_grounding_gain",
            sorted(indices, key=lambda i: float(values[i][2] - values[i][1]), reverse=True),
        ),
        (
            "seg_grounding_regression",
            sorted(indices, key=lambda i: float(values[i][2] - values[i][1])),
        ),
        (
            "largest_method_disagreement",
            sorted(indices, key=lambda i: float(values[i].max() - values[i].min()), reverse=True),
        ),
        ("all_methods_succeed", sorted(indices, key=lambda i: float(values[i].mean()), reverse=True)),
    ]
    selected: list[dict[str, Any]] = []
    used: set[int] = set()
    for category, ranking in rankings:
        added = 0
        for index in ranking:
            if index in used:
                continue
            selected.append(
                {
                    "index": index,
                    "category": category,
                    "csv_ious": values[index].tolist(),
                }
            )
            used.add(index)
            added += 1
            if added >= quota or len(selected) >= count:
                break
        if len(selected) >= count:
            break
    if len(selected) < count:
        for index in indices:
            if index not in used:
                selected.append(
                    {
                        "index": index,
                        "category": "additional",
                        "csv_ious": values[index].tolist(),
                    }
                )
                if len(selected) >= count:
                    break
    return selected


def mask_from_logits(
    logits: torch.Tensor,
    grid_shape: tuple[int, int],
    target_shape: tuple[int, int],
    threshold: float,
) -> np.ndarray:
    probability = torch.sigmoid(logits.float()).reshape(1, 1, *grid_shape)
    probability = F.interpolate(
        probability,
        size=target_shape,
        mode="bilinear",
        align_corners=False,
    ).squeeze()
    return probability.ge(threshold).cpu().numpy().astype(bool)


def mask_iou(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    union = np.logical_or(prediction, target).sum()
    return float(np.logical_and(prediction, target).sum() / max(int(union), 1))


def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    alpha: float = 0.52,
) -> np.ndarray:
    output = image.astype(np.float32).copy()
    values = np.asarray(color, dtype=np.float32)
    output[mask] = (1.0 - alpha) * output[mask] + alpha * values
    return np.clip(output, 0, 255).astype(np.uint8)


def error_overlay(image: np.ndarray, prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    target = target.astype(bool)
    prediction = prediction.astype(bool)
    true_positive = prediction & target
    false_positive = prediction & ~target
    false_negative = ~prediction & target
    output = overlay_mask(image, true_positive, (35, 190, 95))
    output = overlay_mask(output, false_positive, (235, 65, 55))
    return overlay_mask(output, false_negative, (45, 115, 235))


def chunks(values: list[Any], size: int) -> list[list[Any]]:
    return [values[start : start + size] for start in range(0, len(values), size)]


def run_selected_inference(
    spec: RunSpec,
    samples: list[OnePass7BSample],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    config = read_checkpoint_config(spec.checkpoint)
    base_model = str(args.base_model) if args.base_model else config.get("base_model")
    if not base_model:
        raise ValueError(f"No base model found for {spec.label}.")
    dtype = args.dtype or str(config.get("dtype", "bf16"))
    attn_implementation = args.attn_implementation or str(
        config.get("attn_implementation", "sdpa")
    )
    min_pixels = args.min_pixels or int(config.get("min_pixels", 802816))
    max_pixels = args.max_pixels or int(config.get("max_pixels", 1003520))
    model, processor = load_base_onepass_model(
        stamp_code_dir=args.stamp_code_dir,
        base_model=base_model,
        device=device,
        dtype=dtype,
        attn_implementation=attn_implementation,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        max_grid_height=int(config.get("max_grid_height", 512)),
        max_grid_width=int(config.get("max_grid_width", 512)),
        use_lora=bool(config.get("use_lora", True)),
        lora_rank=int(config.get("lora_rank", 16)),
        lora_alpha=float(config.get("lora_alpha", 32.0)),
        lora_dropout=float(config.get("lora_dropout", 0.05)),
        use_rslora=bool(config.get("use_rslora", False)),
        train_visual_projection=bool(config.get("train_visual_projection", True)),
        train_position_embeddings=bool(config.get("train_position_embeddings", True)),
        train_classifier=bool(config.get("train_classifier", True)),
        use_seg_grounding=bool(config.get("use_seg_grounding", False)),
        seg_grounding_size=int(config.get("seg_grounding_size", 256)),
        seg_grounding_temperature=float(config.get("seg_grounding_temperature", 0.1)),
        use_seg_fusion=bool(config.get("use_seg_fusion", True)),
        allow_existing_segmentation_tokens=args.allow_existing_segmentation_tokens,
        gradient_checkpointing=False,
    )
    load_checkpoint(model, spec.checkpoint)
    model.eval()
    autocast_enabled = device.type == "cuda" and dtype != "fp32"
    autocast_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16
    predictions: dict[int, dict[str, Any]] = {}
    started = time.perf_counter()
    with torch.inference_mode():
        for batch in chunks(samples, args.batch_size):
            prepared = prepare_batch(batch, processor, device)
            with torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=autocast_enabled,
            ):
                outputs = model(
                    input_ids=prepared["input_ids"],
                    attention_mask=prepared["attention_mask"],
                    pixel_values=prepared["pixel_values"],
                    image_grid_thw=prepared["image_grid_thw"],
                )
            seg_groups = outputs.seg_logits or [None] * len(batch)
            for sample, final_logits, raw_logits, seg_logits, grid_shape in zip(
                batch,
                outputs.mask_logits,
                outputs.raw_mask_logits,
                seg_groups,
                outputs.grid_shapes,
            ):
                target = sample.mask.numpy().astype(bool)
                final = mask_from_logits(
                    final_logits, grid_shape, tuple(target.shape), args.threshold
                )
                raw = mask_from_logits(raw_logits, grid_shape, tuple(target.shape), args.threshold)
                result: dict[str, Any] = {
                    "prediction": final,
                    "raw_prediction": raw,
                    "iou": mask_iou(final, target),
                    "raw_iou": mask_iou(raw, target),
                    "final_raw_disagreement_rate": float(np.not_equal(final, raw).mean()),
                }
                if seg_logits is not None:
                    seg = mask_from_logits(
                        seg_logits, grid_shape, tuple(target.shape), args.threshold
                    )
                    result.update(
                        {
                            "seg_prediction": seg,
                            "seg_iou": mask_iou(seg, target),
                        }
                    )
                predictions[sample.record.index] = result
    elapsed = time.perf_counter() - started
    metadata = {
        "label": spec.label,
        "checkpoint": str(spec.checkpoint),
        "rows_csv": str(spec.rows_csv),
        "initialization": config.get("initialization"),
        "use_seg_grounding": bool(config.get("use_seg_grounding", False)),
        "seg_fusion_alpha": model.seg_fusion_alpha(),
        "selected_inference_seconds": elapsed,
    }
    del model, processor
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return predictions, metadata


def save_prediction_masks(
    output_dir: Path,
    specs: list[RunSpec],
    samples: list[OnePass7BSample],
    predictions: list[dict[int, dict[str, Any]]],
) -> None:
    for spec, method_predictions in zip(specs, predictions):
        method_dir = output_dir / "pred_masks" / slug(spec.label)
        method_dir.mkdir(parents=True, exist_ok=True)
        for sample in samples:
            index = sample.record.index
            mask = method_predictions[index]["prediction"].astype(np.uint8) * 255
            Image.fromarray(mask, mode="L").save(method_dir / f"sample_{index:06d}.png")


def save_case_panel(
    path: Path,
    sample: OnePass7BSample,
    specs: list[RunSpec],
    predictions: list[dict[int, dict[str, Any]]],
    category: str,
) -> None:
    image = np.asarray(sample.image.convert("RGB"), dtype=np.uint8)
    target = sample.mask.numpy().astype(bool)
    panels = [
        (image, "Image"),
        (overlay_mask(image, target, (35, 190, 95)), "Ground truth"),
    ]
    index = sample.record.index
    for spec, method_predictions in zip(specs, predictions):
        result = method_predictions[index]
        panels.append(
            (
                error_overlay(image, result["prediction"], target),
                f"{spec.label}\nIoU {result['iou']:.3f}",
            )
        )
    figure, axes = plt.subplots(1, len(panels), figsize=(4.0 * len(panels), 4.4))
    for axis, (panel, title) in zip(axes, panels):
        axis.imshow(panel)
        axis.set_title(title, fontsize=10)
        axis.axis("off")
    prompt = textwrap.fill(sample.record.instruction, width=120)
    figure.suptitle(f"{category} | sample {index}\n{prompt}", fontsize=11)
    figure.legend(
        handles=[
            Patch(color="#23be5f", label="TP"),
            Patch(color="#eb4137", label="FP"),
            Patch(color="#2d73eb", label="FN"),
        ],
        loc="lower center",
        ncol=3,
        frameon=False,
    )
    figure.tight_layout(rect=(0, 0.06, 1, 0.91))
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_seg_branch_panel(
    path: Path,
    sample: OnePass7BSample,
    spec: RunSpec,
    result: dict[str, Any],
) -> None:
    if "seg_prediction" not in result:
        return
    image = np.asarray(sample.image.convert("RGB"), dtype=np.uint8)
    target = sample.mask.numpy().astype(bool)
    values = [
        (image, "Image"),
        (overlay_mask(image, target, (35, 190, 95)), "Ground truth"),
        (
            error_overlay(image, result["raw_prediction"], target),
            f"Raw branch\nIoU {result['raw_iou']:.3f}",
        ),
        (
            error_overlay(image, result["seg_prediction"], target),
            f"SEG-only branch\nIoU {result['seg_iou']:.3f}",
        ),
        (
            error_overlay(image, result["prediction"], target),
            f"Final ({spec.label})\nIoU {result['iou']:.3f}",
        ),
    ]
    figure, axes = plt.subplots(1, len(values), figsize=(20, 4.4))
    for axis, (panel, title) in zip(axes, values):
        axis.imshow(panel)
        axis.set_title(title, fontsize=10)
        axis.axis("off")
    figure.suptitle(
        f"SEG fusion diagnostic | sample {sample.record.index} | "
        f"final/raw pixel disagreement {100.0 * result['final_raw_disagreement_rate']:.4f}%",
        fontsize=11,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_distribution_plot(
    path: Path,
    specs: list[RunSpec],
    rows_by_run: list[dict[int, dict[str, Any]]],
) -> dict[str, Any]:
    indices = common_indices(rows_by_run)
    arrays = [np.asarray([rows[index]["iou"] for index in indices]) for rows in rows_by_run]
    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    bins = np.linspace(0.0, 1.0, 41)
    colors = ("#2f6fdd", "#e28b23", "#bf4dc1")
    for spec, values, color in zip(specs, arrays, colors):
        axes[0, 0].hist(values, bins=bins, alpha=0.35, density=True, label=spec.label, color=color)
        ordered = np.sort(values)
        axes[0, 1].plot(ordered, np.linspace(0, 1, len(ordered)), label=spec.label, color=color)
    axes[0, 0].set_title("Per-sample IoU distribution")
    axes[0, 0].set_xlabel("IoU")
    axes[0, 0].set_ylabel("Density")
    axes[0, 0].legend()
    axes[0, 1].set_title("Empirical cumulative distribution")
    axes[0, 1].set_xlabel("IoU")
    axes[0, 1].set_ylabel("Fraction of samples")
    axes[0, 1].legend()

    deltas = (arrays[1] - arrays[0], arrays[2] - arrays[1])
    delta_labels = (
        f"{specs[1].label} - {specs[0].label}",
        f"{specs[2].label} - {specs[1].label}",
    )
    for values, label, color in zip(deltas, delta_labels, colors[1:]):
        bound = max(float(np.percentile(np.abs(values), 99)), 0.01)
        axes[1, 0].hist(values, bins=np.linspace(-bound, bound, 51), alpha=0.45, label=label, color=color)
    axes[1, 0].axvline(0, color="black", linewidth=1)
    axes[1, 0].set_title("Per-sample IoU changes")
    axes[1, 0].set_xlabel("Delta IoU")
    axes[1, 0].set_ylabel("Samples")
    axes[1, 0].legend(fontsize=8)

    bucket_labels = ("<0.25", "0.25-0.50", "0.50-0.75", ">=0.75")
    x = np.arange(len(bucket_labels))
    width = 0.24
    for run_index, (spec, values, color) in enumerate(zip(specs, arrays, colors)):
        counts = np.asarray(
            [
                np.mean(values < 0.25),
                np.mean((values >= 0.25) & (values < 0.50)),
                np.mean((values >= 0.50) & (values < 0.75)),
                np.mean(values >= 0.75),
            ]
        )
        axes[1, 1].bar(x + (run_index - 1) * width, counts, width, label=spec.label, color=color)
    axes[1, 1].set_title("Quality buckets")
    axes[1, 1].set_xticks(x, bucket_labels)
    axes[1, 1].set_ylabel("Fraction of samples")
    axes[1, 1].legend(fontsize=8)
    for axis in axes.flat:
        axis.grid(alpha=0.2)
    figure.suptitle("OnePass-7B method comparison", fontsize=15)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    pairwise = []
    for previous, current, values in zip(specs, specs[1:], deltas):
        pairwise.append(
            {
                "comparison": f"{current.label} - {previous.label}",
                "mean_delta_iou": float(values.mean()),
                "improved_by_more_than_0.01_fraction": float(np.mean(values > 0.01)),
                "regressed_by_more_than_0.01_fraction": float(np.mean(values < -0.01)),
                "nearly_unchanged_fraction": float(np.mean(np.abs(values) <= 0.01)),
            }
        )
    return {
        "common_samples": len(indices),
        "runs": [
            {
                "label": spec.label,
                "mean_sample_iou": float(values.mean()),
                "iou_below_0.5_fraction": float(np.mean(values < 0.5)),
                "iou_above_0.75_fraction": float(np.mean(values >= 0.75)),
            }
            for spec, values in zip(specs, arrays)
        ],
        "pairwise": pairwise,
    }


def write_summary(
    output_dir: Path,
    specs: list[RunSpec],
    selected: list[dict[str, Any]],
    samples_by_index: dict[int, OnePass7BSample],
    predictions: list[dict[int, dict[str, Any]]],
    metadata: list[dict[str, Any]],
    distribution: dict[str, Any],
) -> None:
    sample_rows = []
    for case in selected:
        index = case["index"]
        sample = samples_by_index[index]
        sample_rows.append(
            {
                "index": index,
                "category": case["category"],
                "instruction": sample.record.instruction,
                "ious": {
                    spec.label: predictions[run_index][index]["iou"]
                    for run_index, spec in enumerate(specs)
                },
            }
        )
    summary = {
        "methods": metadata,
        "distribution": distribution,
        "selected_cases": sample_rows,
        "legend": {
            "green": "true positive",
            "red": "false positive / over-segmentation",
            "blue": "false negative / under-segmentation",
        },
    }
    (output_dir / "visualization_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    lines = [
        "# OnePass-7B visual comparison",
        "",
        "## Dataset-level diagnostics",
        "",
        "| Method | Mean sample IoU | IoU < 0.5 | IoU >= 0.75 |",
        "|---|---:|---:|---:|",
    ]
    for row in distribution["runs"]:
        lines.append(
            f"| {row['label']} | {row['mean_sample_iou']:.6f} | "
            f"{row['iou_below_0.5_fraction']:.2%} | {row['iou_above_0.75_fraction']:.2%} |"
        )
    lines.extend(
        [
            "",
            "## Pairwise changes",
            "",
            "| Comparison | Mean delta | Improved >0.01 | Regressed >0.01 | Nearly unchanged |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in distribution["pairwise"]:
        lines.append(
            f"| {row['comparison']} | {row['mean_delta_iou']:+.6f} | "
            f"{row['improved_by_more_than_0.01_fraction']:.2%} | "
            f"{row['regressed_by_more_than_0.01_fraction']:.2%} | "
            f"{row['nearly_unchanged_fraction']:.2%} |"
        )
    lines.extend(
        [
            "",
            "## Selected qualitative cases",
            "",
            "| Index | Category | " + " | ".join(spec.label for spec in specs) + " |",
            "|---:|---|" + "---:|" * len(specs),
        ]
    )
    for row in sample_rows:
        lines.append(
            f"| {row['index']} | {row['category']} | "
            + " | ".join(f"{row['ious'][spec.label]:.4f}" for spec in specs)
            + " |"
        )
    lines.extend(
        [
            "",
            "Colors in prediction panels: green=TP, red=FP, blue=FN.",
            "",
            "The selected cases are post-hoc qualitative diagnostics based on GT IoU; "
            "selection is not part of model inference.",
        ]
    )
    (output_dir / "visualization_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> int:
    args = parse_args()
    if len(args.run) != 3:
        raise ValueError("Pass exactly three --run entries in training order.")
    if args.count <= 0 or args.batch_size <= 0:
        raise ValueError("count and batch-size must be positive.")
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("threshold must be in (0, 1).")
    specs = [
        RunSpec(label=value[0], checkpoint=Path(value[1]), rows_csv=Path(value[2]))
        for value in args.run
    ]
    for spec in specs:
        if not spec.checkpoint.is_file():
            raise FileNotFoundError(spec.checkpoint)
    rows_by_run = [load_iou_rows(spec.rows_csv) for spec in specs]
    if args.indices:
        selected = []
        valid = set(common_indices(rows_by_run))
        for index in args.indices:
            if index not in valid:
                raise ValueError(f"Sample index {index} is not common to all CSV files.")
            selected.append(
                {
                    "index": index,
                    "category": "user_selected",
                    "csv_ious": [rows[index]["iou"] for rows in rows_by_run],
                }
            )
    else:
        selected = select_cases(rows_by_run, args.count)

    dataset = OnePass7BDataset(args.eval_json, args.data_root)
    positions = {record.index: position for position, record in enumerate(dataset.records)}
    missing = [case["index"] for case in selected if case["index"] not in positions]
    if missing:
        raise ValueError(f"Selected indices are absent from eval JSON: {missing}")
    samples = [dataset[positions[case["index"]]] for case in selected]
    samples_by_index = {sample.record.index: sample for sample in samples}
    device = torch.device(args.device)
    configure_cudnn(args.disable_cudnn, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    distribution = save_distribution_plot(
        args.output_dir / "method_iou_diagnostics.png", specs, rows_by_run
    )
    all_predictions = []
    all_metadata = []
    for spec in specs:
        print(f"Running selected samples: {spec.label}", flush=True)
        predictions, metadata = run_selected_inference(spec, samples, args, device)
        all_predictions.append(predictions)
        all_metadata.append(metadata)

    save_prediction_masks(args.output_dir, specs, samples, all_predictions)
    case_dir = args.output_dir / "cases"
    branch_dir = args.output_dir / "seg_branch"
    case_dir.mkdir(parents=True, exist_ok=True)
    branch_dir.mkdir(parents=True, exist_ok=True)
    for case, sample in zip(selected, samples):
        index = sample.record.index
        save_case_panel(
            case_dir / f"sample_{index:06d}_{case['category']}.png",
            sample,
            specs,
            all_predictions,
            case["category"],
        )
        for spec, method_predictions in zip(specs, all_predictions):
            result = method_predictions[index]
            if "seg_prediction" in result:
                save_seg_branch_panel(
                    branch_dir / f"sample_{index:06d}_{slug(spec.label)}.png",
                    sample,
                    spec,
                    result,
                )

    write_summary(
        args.output_dir,
        specs,
        selected,
        samples_by_index,
        all_predictions,
        all_metadata,
        distribution,
    )
    print(f"Visualization summary: {args.output_dir / 'visualization_summary.md'}", flush=True)
    print(f"Case panels: {case_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
