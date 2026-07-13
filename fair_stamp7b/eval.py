from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm

from onepass_qwen7b.data import MASK_TOKEN, SEG_TOKEN, OnePass7BDataset, collate_samples

from .checkpoint import load_checkpoint, read_checkpoint_config
from .runtime import configure_cudnn, load_fair_stamp_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate fair native STAMP-7B with real two-stage inference.")
    parser.add_argument("--stamp-code-dir", required=True, type=Path)
    parser.add_argument("--base-model")
    parser.add_argument("--eval-json", required=True, type=Path, nargs="+")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=64)
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


def prepare_generation_batch(samples: list[Any], processor: Any, device: torch.device) -> dict[str, Any]:
    messages = [
        [{"role": "user", "content": [{"image": sample.image}, {"text": sample.record.instruction}]}]
        for sample in samples
    ]
    texts = [
        processor.apply_chat_template(value, tokenize=False, add_generation_prompt=True)
        for value in messages
    ]
    encoded = processor(
        text=texts,
        images=[sample.image for sample in samples],
        padding=True,
        return_tensors="pt",
    )
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in encoded.items()
    }


def build_classification_batch(
    encoded: dict[str, Any],
    generated_ids: torch.Tensor,
    processor: Any,
) -> tuple[torch.Tensor, torch.Tensor, list[tuple[int, int]], list[bool], list[str]]:
    tokenizer = processor.tokenizer
    seg_id = int(tokenizer.convert_tokens_to_ids(SEG_TOKEN))
    mask_id = int(tokenizer.convert_tokens_to_ids(MASK_TOKEN))
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    prompt_width = int(encoded["input_ids"].shape[1])
    merge_size = int(processor.image_processor.merge_size)
    sequences = []
    masks = []
    grids = []
    generated_seg = []
    generated_text = []
    for index in range(generated_ids.shape[0]):
        prompt = encoded["input_ids"][index][encoded["attention_mask"][index].bool()]
        continuation = generated_ids[index, prompt_width:]
        seg_positions = torch.where(continuation.eq(seg_id))[0]
        found = seg_positions.numel() > 0
        if found:
            continuation_prefix = continuation[: int(seg_positions[0].item()) + 1]
        else:
            continuation_prefix = torch.tensor([seg_id], device=prompt.device, dtype=torch.long)
        prefix = torch.cat([prompt, continuation_prefix])
        temporal, raw_height, raw_width = (
            int(value) for value in encoded["image_grid_thw"][index].tolist()
        )
        if temporal != 1 or raw_height % merge_size or raw_width % merge_size:
            raise ValueError(
                f"Unsupported grid {(temporal, raw_height, raw_width)} for merge_size={merge_size}."
            )
        height, width = raw_height // merge_size, raw_width // merge_size
        query_ids = torch.full((height * width,), mask_id, device=prompt.device, dtype=torch.long)
        sequence = torch.cat([prefix, query_ids])
        sequences.append(sequence)
        masks.append(torch.ones_like(sequence))
        grids.append((height, width))
        generated_seg.append(found)
        generated_text.append(tokenizer.decode(continuation, skip_special_tokens=False))
    return (
        pad_sequence(sequences, batch_first=True, padding_value=int(pad_id)),
        pad_sequence(masks, batch_first=True, padding_value=0),
        grids,
        generated_seg,
        generated_text,
    )


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0 or args.max_new_tokens <= 0:
        raise ValueError("batch-size and max-new-tokens must be positive.")
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("threshold must be in (0, 1).")
    config = read_checkpoint_config(args.checkpoint)
    base_model = args.base_model or config.get("base_model")
    if not base_model:
        raise ValueError("base-model is required and was not found in checkpoint config.")
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
    model, processor = load_fair_stamp_model(
        stamp_code_dir=args.stamp_code_dir,
        base_model=base_model,
        device=device,
        dtype=dtype,
        attn_implementation=attn_implementation,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        use_lora=bool(config.get("use_lora", True)),
        lora_rank=int(config.get("lora_rank", 64)),
        lora_alpha=float(config.get("lora_alpha", 128.0)),
        lora_dropout=float(config.get("lora_dropout", 0.05)),
        use_rslora=bool(config.get("use_rslora", True)),
        allow_existing_segmentation_tokens=args.allow_existing_segmentation_tokens,
        gradient_checkpointing=False,
    )
    load_checkpoint(model, args.checkpoint)
    model.eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = processor.tokenizer
    seg_id = int(tokenizer.convert_tokens_to_ids(SEG_TOKEN))
    rows = []
    total_intersection = 0
    total_union = 0
    model_seconds = 0.0
    end_to_end_seconds = 0.0
    no_seg_count = 0
    autocast_enabled = device.type == "cuda" and dtype != "fp32"
    autocast_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16

    with torch.inference_mode():
        for samples in tqdm(loader, desc="fair STAMP-7B eval", dynamic_ncols=True):
            synchronize(device)
            end_start = time.perf_counter()
            encoded = prepare_generation_batch(samples, processor, device)
            synchronize(device)
            model_start = time.perf_counter()
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_enabled):
                generated_ids = model.backbone.generate(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                    pixel_values=encoded["pixel_values"],
                    image_grid_thw=encoded["image_grid_thw"],
                    do_sample=False,
                    max_new_tokens=args.max_new_tokens,
                    eos_token_id=seg_id,
                    pad_token_id=tokenizer.pad_token_id,
                    use_cache=True,
                )
                cls_ids, cls_mask, grids, generated_seg, generated_text = build_classification_batch(
                    encoded, generated_ids, processor
                )
                mask_logits = model.classify_masks(
                    cls_input_ids=cls_ids,
                    cls_attention_mask=cls_mask,
                    pixel_values=encoded["pixel_values"],
                    image_grid_thw=encoded["image_grid_thw"],
                    grid_shapes=grids,
                )
            synchronize(device)
            model_seconds += time.perf_counter() - model_start
            for sample, logits, grid, has_seg, text in zip(
                samples, mask_logits, grids, generated_seg, generated_text
            ):
                target = sample.mask.to(device=device, dtype=torch.bool)
                if has_seg:
                    probability = torch.sigmoid(logits.float()).reshape(1, 1, *grid)
                    probability = F.interpolate(
                        probability,
                        size=tuple(int(value) for value in target.shape),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze()
                    prediction = probability >= args.threshold
                else:
                    prediction = torch.zeros_like(target)
                    no_seg_count += 1
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
                        "generated_seg": has_seg,
                        "generated_text": text,
                    }
                )
            synchronize(device)
            end_to_end_seconds += time.perf_counter() - end_start

    rows_path = args.output_dir / "fair_stamp7b_eval_rows.csv"
    with rows_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    mean_iou = sum(row["iou"] for row in rows) / max(len(rows), 1)
    summary = {
        "samples": len(rows),
        "mean_iou": mean_iou,
        "gIoU": mean_iou,
        "cIoU": total_intersection / max(total_union, 1),
        "total_intersection": total_intersection,
        "total_union": total_union,
        "no_seg_count": no_seg_count,
        "no_seg_rate": no_seg_count / max(len(rows), 1),
        "model_seconds_per_sample": model_seconds / max(len(rows), 1),
        "end_to_end_seconds_per_sample": end_to_end_seconds / max(len(rows), 1),
        "model_calls": 2 * len(loader),
        "autoregressive_generate_calls": len(loader),
        "inference_mode": "autoregressive_SEG_then_mask_classification",
        "rows_csv": str(rows_path),
    }
    (args.output_dir / "eval_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

