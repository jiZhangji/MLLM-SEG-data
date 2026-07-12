from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .checkpoint import load_checkpoint, read_checkpoint_config
from .data import OnePass7BDataset, collate_samples, prepare_batch
from .runtime import configure_cudnn, load_base_onepass_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate OnePass Qwen2-VL-7B without autoregressive generation.")
    parser.add_argument("--stamp-code-dir", required=True, type=Path)
    parser.add_argument("--base-model")
    parser.add_argument("--eval-json", required=True, type=Path, nargs="+")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"))
    parser.add_argument("--attn-implementation")
    parser.add_argument("--min-pixels", type=int)
    parser.add_argument("--max-pixels", type=int)
    parser.add_argument("--disable-cudnn", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--allow-existing-segmentation-tokens", action="store_true")
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def intersection_union(prediction: torch.Tensor, target: torch.Tensor) -> tuple[int, int]:
    prediction = prediction.bool()
    target = target.bool()
    return int((prediction & target).sum().item()), int((prediction | target).sum().item())


def main() -> int:
    args = parse_args()
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("threshold must be in (0, 1).")
    config = read_checkpoint_config(args.checkpoint)
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
        max_grid_height=int(config.get("max_grid_height", 64)),
        max_grid_width=int(config.get("max_grid_width", 64)),
        use_lora=bool(config.get("use_lora", True)),
        lora_rank=int(config.get("lora_rank", 16)),
        lora_alpha=float(config.get("lora_alpha", 32.0)),
        lora_dropout=float(config.get("lora_dropout", 0.05)),
        use_rslora=bool(config.get("use_rslora", False)),
        train_visual_projection=bool(config.get("train_visual_projection", True)),
        train_position_embeddings=bool(config.get("train_position_embeddings", True)),
        train_classifier=bool(config.get("train_classifier", True)),
        allow_existing_segmentation_tokens=args.allow_existing_segmentation_tokens,
        gradient_checkpointing=False,
    )
    load_checkpoint(model, args.checkpoint)
    model.eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    total_intersection = 0
    total_union = 0
    model_seconds = 0.0
    end_to_end_seconds = 0.0
    autocast_enabled = device.type == "cuda" and dtype != "fp32"
    autocast_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16

    with torch.inference_mode():
        for samples in tqdm(loader, desc="onepass7b eval", dynamic_ncols=True):
            synchronize(device)
            end_start = time.perf_counter()
            prepared = prepare_batch(samples, processor, device)
            synchronize(device)
            model_start = time.perf_counter()
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_enabled):
                outputs = model(
                    input_ids=prepared["input_ids"],
                    attention_mask=prepared["attention_mask"],
                    pixel_values=prepared["pixel_values"],
                    image_grid_thw=prepared["image_grid_thw"],
                )
            synchronize(device)
            model_seconds += time.perf_counter() - model_start
            for sample, logits, grid_shape in zip(samples, outputs.mask_logits, outputs.grid_shapes):
                target = sample.mask.to(device=device, dtype=torch.bool)
                probability = torch.sigmoid(logits.float()).reshape(1, 1, *grid_shape)
                probability = F.interpolate(
                    probability,
                    size=tuple(int(value) for value in target.shape),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze()
                prediction = probability >= args.threshold
                intersection, union = intersection_union(prediction, target)
                iou = intersection / max(union, 1)
                total_intersection += intersection
                total_union += union
                rows.append(
                    {
                        "name": sample.record.name,
                        "index": sample.record.index,
                        "image_path": str(sample.record.image_path),
                        "iou": iou,
                        "intersection": intersection,
                        "union": union,
                    }
                )
            synchronize(device)
            end_to_end_seconds += time.perf_counter() - end_start

    rows_path = args.output_dir / "onepass7b_eval_rows.csv"
    with rows_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    mean_iou = sum(row["iou"] for row in rows) / max(len(rows), 1)
    ciou = total_intersection / max(total_union, 1)
    summary = {
        "samples": len(rows),
        "mean_iou": mean_iou,
        "gIoU": mean_iou,
        "cIoU": ciou,
        "total_intersection": total_intersection,
        "total_union": total_union,
        "model_seconds_per_sample": model_seconds / max(len(rows), 1),
        "end_to_end_seconds_per_sample": end_to_end_seconds / max(len(rows), 1),
        "model_calls": len(loader),
        "autoregressive_generate_calls": 0,
        "initialization": "base_qwen2_vl_without_stamp_weights",
        "prompt_mode": "strict_query",
        "rows_csv": str(rows_path),
    }
    (args.output_dir / "eval_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
