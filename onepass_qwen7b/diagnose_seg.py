from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .checkpoint import load_checkpoint, read_checkpoint_config
from .data import OnePass7BDataset, collate_samples, prepare_batch
from .eval import logits_metrics, synchronize
from .runtime import configure_cudnn, load_base_onepass_model


VARIANTS = (
    "baseline",
    "seg_shuffle",
    "no_seg",
    "gt_foreground_oracle",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SEG shuffle, no-SEG, and GT-foreground-oracle diagnostics."
    )
    parser.add_argument("--stamp-code-dir", required=True, type=Path)
    parser.add_argument("--base-model")
    parser.add_argument("--eval-json", required=True, type=Path, nargs="+")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"))
    parser.add_argument("--attn-implementation")
    parser.add_argument("--min-pixels", type=int)
    parser.add_argument("--max-pixels", type=int)
    parser.add_argument("--disable-cudnn", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--allow-existing-segmentation-tokens", action="store_true")
    return parser.parse_args()


def foreground_prototype_targets(
    samples: list[Any],
    grid_shapes: list[tuple[int, int]],
    device: torch.device,
) -> list[torch.Tensor]:
    """Downsample GT masks with area weights so tiny foregrounds remain non-empty."""
    targets = []
    for sample, (height, width) in zip(samples, grid_shapes):
        mask = sample.mask.to(device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        target = F.interpolate(mask, size=(height, width), mode="area").reshape(-1)
        if target.sum() <= 0:
            raise ValueError(f"Sample {sample.record.name} has an empty GT foreground prototype.")
        targets.append(target)
    return targets


def empty_accumulator() -> dict[str, float | int]:
    return {
        "iou_sum": 0.0,
        "total_intersection": 0,
        "total_union": 0,
        "model_seconds": 0.0,
        "model_calls": 0,
    }


def add_metrics(
    accumulator: dict[str, float | int],
    logits: torch.Tensor,
    target: torch.Tensor,
    grid_shape: tuple[int, int],
    threshold: float,
) -> tuple[float, int, int]:
    iou, intersection, union = logits_metrics(logits, target, grid_shape, threshold)
    accumulator["iou_sum"] = float(accumulator["iou_sum"]) + iou
    accumulator["total_intersection"] = int(accumulator["total_intersection"]) + intersection
    accumulator["total_union"] = int(accumulator["total_union"]) + union
    return iou, intersection, union


def variant_summary(
    accumulator: dict[str, float | int],
    samples: int,
    baseline_mean_iou: float,
    baseline_ciou: float,
) -> dict[str, float | int]:
    mean_iou = float(accumulator["iou_sum"]) / max(samples, 1)
    ciou = int(accumulator["total_intersection"]) / max(int(accumulator["total_union"]), 1)
    return {
        "samples": samples,
        "mean_iou": mean_iou,
        "gIoU": mean_iou,
        "cIoU": ciou,
        "mean_iou_delta_vs_baseline": mean_iou - baseline_mean_iou,
        "cIoU_delta_vs_baseline": ciou - baseline_ciou,
        "total_intersection": int(accumulator["total_intersection"]),
        "total_union": int(accumulator["total_union"]),
        "model_seconds": float(accumulator["model_seconds"]),
        "model_calls": int(accumulator["model_calls"]),
    }


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# OnePass 7B SEG diagnostics",
        "",
        "| Variant | gIoU | cIoU | Delta gIoU | Delta cIoU |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in VARIANTS:
        values = summary["variants"][name]
        lines.append(
            f"| {name} | {values['gIoU']:.6f} | {values['cIoU']:.6f} | "
            f"{values['mean_iou_delta_vs_baseline']:+.6f} | "
            f"{values['cIoU_delta_vs_baseline']:+.6f} |"
        )
    lines.extend(
        [
            "",
            "## Definitions",
            "",
            "- `baseline`: unchanged checkpoint output.",
            "- `seg_shuffle`: rotate contextual `h_SEG` across samples only in the explicit grounding head.",
            "- `no_seg`: zero and attention-mask the SEG input token, then use the raw MASK branch.",
            "- `gt_foreground_oracle`: replace the SEG input embedding with the GT-weighted visual-token prototype.",
            "",
            f"Actually shuffled samples: {summary['actually_shuffled_samples']} / {summary['samples']}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.batch_size < 2:
        raise ValueError("SEG shuffle requires --batch-size >= 2.")
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("threshold must be in (0, 1).")

    config = read_checkpoint_config(args.checkpoint)
    if not bool(config.get("use_seg_grounding", False)):
        raise ValueError("The checkpoint has no SEG-grounding head; use the SEG-grounding checkpoint.")
    base_model = args.base_model or config.get("base_model")
    if not base_model:
        raise ValueError("base-model is required and was not found in the checkpoint config.")
    dtype = args.dtype or str(config.get("dtype", "bf16"))
    attn_implementation = args.attn_implementation or str(config.get("attn_implementation", "sdpa"))
    min_pixels = args.min_pixels or int(config.get("min_pixels", 802816))
    max_pixels = args.max_pixels or int(config.get("max_pixels", 1003520))
    device = torch.device(args.device)
    configure_cudnn(args.disable_cudnn, device)

    dataset = OnePass7BDataset(args.eval_json, args.data_root, limit=args.limit, offset=args.offset)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        collate_fn=collate_samples,
    )
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
        use_seg_grounding=True,
        seg_grounding_size=int(config.get("seg_grounding_size", 256)),
        seg_grounding_temperature=float(config.get("seg_grounding_temperature", 0.1)),
        use_seg_fusion=bool(config.get("use_seg_fusion", True)),
        allow_existing_segmentation_tokens=args.allow_existing_segmentation_tokens,
        gradient_checkpointing=False,
    )
    load_checkpoint(model, args.checkpoint)
    model.eval()
    if model.seg_grounding_head is None:
        raise RuntimeError("SEG grounding head was not constructed.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    accumulators = {name: empty_accumulator() for name in VARIANTS}
    rows: list[dict[str, Any]] = []
    actually_shuffled_samples = 0
    autocast_enabled = device.type == "cuda" and dtype != "fp32"
    autocast_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16

    with torch.inference_mode():
        for samples in tqdm(loader, desc="onepass7b SEG diagnostics", dynamic_ncols=True):
            prepared = prepare_batch(samples, processor, device)
            call_args = {
                "input_ids": prepared["input_ids"],
                "attention_mask": prepared["attention_mask"],
                "pixel_values": prepared["pixel_values"],
                "image_grid_thw": prepared["image_grid_thw"],
            }
            with torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=autocast_enabled,
            ):
                synchronize(device)
                start = time.perf_counter()
                baseline = model(**call_args)
                synchronize(device)
                accumulators["baseline"]["model_seconds"] += time.perf_counter() - start
                accumulators["baseline"]["model_calls"] += 1

                if len(samples) > 1:
                    shuffled_hidden = baseline.seg_hidden.roll(shifts=1, dims=0)
                    actually_shuffled_samples += len(samples)
                else:
                    shuffled_hidden = baseline.seg_hidden
                start = time.perf_counter()
                shuffled_logits, _ = model.logits_for_seg_hidden(
                    baseline.raw_mask_logits,
                    baseline.mask_hidden,
                    shuffled_hidden,
                )
                synchronize(device)
                accumulators["seg_shuffle"]["model_seconds"] += time.perf_counter() - start

                synchronize(device)
                start = time.perf_counter()
                no_seg = model(**call_args, disable_seg_query=True)
                synchronize(device)
                accumulators["no_seg"]["model_seconds"] += time.perf_counter() - start
                accumulators["no_seg"]["model_calls"] += 1

                prototype_targets = foreground_prototype_targets(
                    samples, baseline.grid_shapes, device
                )
                synchronize(device)
                start = time.perf_counter()
                oracle = model(**call_args, seg_query_targets=prototype_targets)
                synchronize(device)
                accumulators["gt_foreground_oracle"]["model_seconds"] += (
                    time.perf_counter() - start
                )
                accumulators["gt_foreground_oracle"]["model_calls"] += 1

            for index, (sample, grid_shape) in enumerate(zip(samples, baseline.grid_shapes)):
                target = sample.mask.to(device=device, dtype=torch.bool)
                logits_by_variant = {
                    "baseline": baseline.mask_logits[index],
                    "seg_shuffle": shuffled_logits[index],
                    "no_seg": no_seg.mask_logits[index],
                    "gt_foreground_oracle": oracle.mask_logits[index],
                }
                row: dict[str, Any] = {
                    "name": sample.record.name,
                    "index": sample.record.index,
                    "image_path": str(sample.record.image_path),
                }
                for name, logits in logits_by_variant.items():
                    iou, intersection, union = add_metrics(
                        accumulators[name], logits, target, grid_shape, args.threshold
                    )
                    row[f"{name}_iou"] = iou
                    row[f"{name}_intersection"] = intersection
                    row[f"{name}_union"] = union
                    if name != "baseline":
                        row[f"{name}_iou_delta"] = iou - row["baseline_iou"]
                rows.append(row)

    if not rows:
        raise ValueError("No evaluation samples were processed.")
    rows_path = args.output_dir / "seg_diagnostic_rows.csv"
    with rows_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    baseline_mean_iou = float(accumulators["baseline"]["iou_sum"]) / len(rows)
    baseline_ciou = int(accumulators["baseline"]["total_intersection"]) / max(
        int(accumulators["baseline"]["total_union"]), 1
    )
    summary = {
        "checkpoint": str(args.checkpoint),
        "eval_json": [str(path) for path in args.eval_json],
        "samples": len(rows),
        "batch_size": args.batch_size,
        "threshold": args.threshold,
        "seg_fusion_alpha": model.seg_fusion_alpha(),
        "actually_shuffled_samples": actually_shuffled_samples,
        "definitions": {
            "baseline": "Unchanged checkpoint output.",
            "seg_shuffle": "Contextual h_SEG rotated across the batch only for the explicit grounding head.",
            "no_seg": "SEG input zeroed and attention-masked; output uses the raw MASK branch.",
            "gt_foreground_oracle": "SEG input replaced by a GT-area-weighted visual-token prototype.",
        },
        "variants": {
            name: variant_summary(
                accumulators[name], len(rows), baseline_mean_iou, baseline_ciou
            )
            for name in VARIANTS
        },
        "rows_csv": str(rows_path),
    }
    summary_path = args.output_dir / "seg_diagnostic_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    markdown_path = args.output_dir / "seg_diagnostic_summary.md"
    write_markdown(summary, markdown_path)
    print(json.dumps(summary, indent=2, ensure_ascii=True), flush=True)
    print(f"Markdown: {markdown_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
