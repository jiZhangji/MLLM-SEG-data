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
from .data import OnePassDataset, collate_samples, prepare_onepass_batch
from .runtime import configure_cudnn, load_onepass_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate true single-forward OnePass-STAMP.")
    parser.add_argument("--stamp-code-dir", type=Path, required=True)
    parser.add_argument("--model-name", type=Path, required=True)
    parser.add_argument("--eval-json", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--attn-implementation")
    parser.add_argument("--min-pixels", type=int)
    parser.add_argument("--max-pixels", type=int)
    parser.add_argument("--use-task-token", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--disable-cudnn", action=argparse.BooleanOptionalAction, default=None)
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    cudnn_disabled = configure_cudnn(args.disable_cudnn, device)
    print(f"cuDNN disabled: {int(cudnn_disabled)}", flush=True)

    saved_config = read_checkpoint_config(args.checkpoint)
    use_task_token = (
        bool(saved_config.get("use_task_token", True))
        if args.use_task_token is None
        else bool(args.use_task_token)
    )
    dtype = args.dtype or str(saved_config.get("dtype", "bf16"))
    attn_implementation = args.attn_implementation or str(saved_config.get("attn_implementation", "sdpa"))
    min_pixels = args.min_pixels or int(saved_config.get("min_pixels", 802816))
    max_pixels = args.max_pixels or int(saved_config.get("max_pixels", 1003520))
    use_lora = bool(saved_config.get("use_lora", False))
    lora_rank = int(saved_config.get("lora_rank", 8))
    lora_alpha = float(saved_config.get("lora_alpha", 16.0))
    lora_dropout = float(saved_config.get("lora_dropout", 0.0))

    dataset = OnePassDataset(args.eval_json, args.data_root, limit=args.limit, offset=args.offset)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        collate_fn=collate_samples,
    )
    model, processor = load_onepass_model(
        stamp_code_dir=args.stamp_code_dir,
        model_name=args.model_name,
        device=device,
        dtype=dtype,
        attn_implementation=attn_implementation,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        use_task_token=use_task_token,
        use_lora=use_lora,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )
    load_checkpoint(model, args.checkpoint)
    model.eval()

    rows = []
    total_intersection = 0
    total_union = 0
    total_iou = 0.0
    total_model_seconds = 0.0
    total_end_to_end_seconds = 0.0
    autocast_enabled = device.type == "cuda" and dtype != "fp32"
    autocast_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16

    with torch.no_grad():
        for samples in tqdm(loader, desc="onepass eval", dynamic_ncols=True):
            synchronize(device)
            end_to_end_start = time.perf_counter()
            prepared = prepare_onepass_batch(samples, processor, device, use_task_token)
            synchronize(device)
            model_start = time.perf_counter()
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
            synchronize(device)
            model_seconds = time.perf_counter() - model_start
            end_to_end_seconds = time.perf_counter() - end_to_end_start
            total_model_seconds += model_seconds
            total_end_to_end_seconds += end_to_end_seconds

            for sample, logits, grid_shape in zip(samples, outputs.mask_logits, outputs.grid_shapes):
                target = sample.mask.bool()
                height, width = target.shape
                probability = torch.sigmoid(logits.float()).reshape(1, 1, *grid_shape)
                prediction = F.interpolate(
                    probability,
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0).squeeze(0).ge(0.5).cpu()
                intersection = int((prediction & target).sum().item())
                union = int((prediction | target).sum().item())
                iou = float(intersection / union) if union else 1.0
                total_intersection += intersection
                total_union += union
                total_iou += iou
                rows.append(
                    {
                        "index": sample.record.index,
                        "name": sample.record.name,
                        "image": str(sample.record.image_path),
                        "mask": str(sample.record.mask_paths[0]),
                        "grid_h": grid_shape[0],
                        "grid_w": grid_shape[1],
                        "intersection": intersection,
                        "union": union,
                        "iou": iou,
                    }
                )

    row_path = args.output_dir / "onepass_eval_rows.csv"
    with row_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["index"])
        writer.writeheader()
        writer.writerows(rows)
    sample_count = len(rows)
    summary = {
        "samples": sample_count,
        "mean_iou": total_iou / max(sample_count, 1),
        "gIoU": total_iou / max(sample_count, 1),
        "cIoU": total_intersection / total_union if total_union else 1.0,
        "total_intersection": total_intersection,
        "total_union": total_union,
        "model_seconds_per_sample": total_model_seconds / max(sample_count, 1),
        "end_to_end_seconds_per_sample": total_end_to_end_seconds / max(sample_count, 1),
        "model_calls": len(loader),
        "autoregressive_generate_calls": 0,
        "use_task_token": use_task_token,
        "rows_csv": str(row_path),
    }
    summary_path = args.output_dir / "onepass_eval_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
