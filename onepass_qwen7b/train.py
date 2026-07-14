from __future__ import annotations

import argparse
import json
import math
import os
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, DistributedSampler, Sampler
from tqdm import tqdm

from .checkpoint import load_checkpoint, save_checkpoint
from .data import OnePass7BDataset, collate_samples, grid_targets, prepare_batch
from .model import segmentation_loss
from .runtime import configure_cudnn, load_base_onepass_model
from .stamp_adapter import load_stamp_adapter_initialization, read_stamp_adapter_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train OnePass segmentation from a base Qwen2-VL-7B checkpoint.")
    parser.add_argument("--stamp-code-dir", required=True, type=Path)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--train-json", required=True, type=Path, nargs="+")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--stamp-adapter",
        type=Path,
        help="Initialize compatible LoRA tensors from a STAMP PEFT adapter directory.",
    )
    parser.add_argument(
        "--stamp-init-classifier",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also initialize the shared scalar mask classifier from the STAMP adapter.",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1, help="Per-GPU micro batch size.")
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--query-learning-rate", type=float, default=1e-4)
    parser.add_argument("--head-learning-rate", type=float, default=1e-4)
    parser.add_argument("--lora-learning-rate", type=float, default=2e-5)
    parser.add_argument("--use-lora", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=float, default=32.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--use-rslora", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--train-visual-projection", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train-position-embeddings", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train-classifier", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-grid-height", type=int, default=512)
    parser.add_argument("--max-grid-width", type=int, default=512)
    parser.add_argument("--bce-weight", type=float, default=0.3)
    parser.add_argument("--dice-weight", type=float, default=0.7)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--min-pixels", type=int, default=802816)
    parser.add_argument("--max-pixels", type=int, default=1003520)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--disable-cudnn", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--allow-existing-segmentation-tokens", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class EpochRandomSampler(Sampler[int]):
    """Deterministic epoch shuffling so single-GPU mid-epoch resume is reproducible."""

    def __init__(self, size: int, seed: int) -> None:
        self.size = int(size)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        return iter(torch.randperm(self.size, generator=generator).tolist())

    def __len__(self) -> int:
        return self.size


def linear_schedule(total_steps: int, warmup_ratio: float):
    warmup_steps = int(total_steps * warmup_ratio)

    def schedule(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        denominator = max(total_steps - warmup_steps, 1)
        return max(0.0, float(total_steps - step) / float(denominator))

    return schedule


def checkpoint_config(args: argparse.Namespace) -> dict[str, Any]:
    initialization = "stamp_7b_lora_warm_start" if args.stamp_adapter else "base_qwen2_vl_without_stamp_weights"
    return {
        "base_model": str(args.base_model),
        "stamp_code_dir": str(args.stamp_code_dir),
        "dtype": args.dtype,
        "attn_implementation": args.attn_implementation,
        "min_pixels": int(args.min_pixels),
        "max_pixels": int(args.max_pixels),
        "max_grid_height": int(args.max_grid_height),
        "max_grid_width": int(args.max_grid_width),
        "use_lora": bool(args.use_lora),
        "lora_rank": int(args.lora_rank),
        "lora_alpha": float(args.lora_alpha),
        "lora_dropout": float(args.lora_dropout),
        "use_rslora": bool(args.use_rslora),
        "train_visual_projection": bool(args.train_visual_projection),
        "train_position_embeddings": bool(args.train_position_embeddings),
        "train_classifier": bool(args.train_classifier),
        "bce_weight": float(args.bce_weight),
        "dice_weight": float(args.dice_weight),
        "initialization": initialization,
        "stamp_adapter": str(args.stamp_adapter.resolve()) if args.stamp_adapter else None,
        "stamp_init_classifier": bool(args.stamp_adapter and args.stamp_init_classifier),
        "prompt_mode": "strict_query",
    }


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def main() -> int:
    args = parse_args()
    if min(args.epochs, args.batch_size, args.gradient_accumulation, args.logging_steps) <= 0:
        raise ValueError("epochs, batch-size, gradient-accumulation and logging-steps must be positive.")
    if not 0.0 <= args.warmup_ratio < 1.0:
        raise ValueError("warmup-ratio must be in [0, 1).")
    if args.bce_weight < 0 or args.dice_weight < 0 or args.bce_weight + args.dice_weight <= 0:
        raise ValueError("Loss weights must be non-negative and not both zero.")
    if not args.train_classifier:
        raise ValueError("The OnePass mask classifier must remain trainable.")
    adapter_config = None
    if args.stamp_adapter is not None:
        if not args.use_lora:
            raise ValueError("--stamp-adapter requires --use-lora.")
        adapter_config = read_stamp_adapter_config(args.stamp_adapter)
        args.lora_rank = adapter_config.rank
        args.lora_alpha = adapter_config.alpha
        args.lora_dropout = adapter_config.dropout
        args.use_rslora = adapter_config.use_rslora

    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
    rank = dist.get_rank() if distributed else 0
    world_size = dist.get_world_size() if distributed else 1
    is_main = rank == 0
    device = torch.device(f"cuda:{local_rank}" if distributed else args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed + rank)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
    cudnn_disabled = configure_cudnn(args.disable_cudnn, device)

    dataset = OnePass7BDataset(args.train_json, args.data_root, limit=args.limit, offset=args.offset)
    if not dataset:
        raise ValueError("Training dataset is empty.")
    sampler = (
        DistributedSampler(dataset, shuffle=True, seed=args.seed, drop_last=False)
        if distributed
        else EpochRandomSampler(len(dataset), args.seed)
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        collate_fn=collate_samples,
    )

    model, processor = load_base_onepass_model(
        stamp_code_dir=args.stamp_code_dir,
        base_model=args.base_model,
        device=device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        max_grid_height=args.max_grid_height,
        max_grid_width=args.max_grid_width,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        use_rslora=args.use_rslora,
        train_visual_projection=args.train_visual_projection,
        train_position_embeddings=args.train_position_embeddings,
        train_classifier=args.train_classifier,
        allow_existing_segmentation_tokens=args.allow_existing_segmentation_tokens,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    raw_model = model
    adapter_report = None
    if args.stamp_adapter is not None:
        adapter_report = load_stamp_adapter_initialization(
            raw_model,
            args.stamp_adapter,
            initialize_classifier=args.stamp_init_classifier,
        )
    query_parameters = [parameter for parameter in raw_model.query_builder.parameters() if parameter.requires_grad]
    parameter_groups = [{"params": query_parameters, "lr": args.query_learning_rate, "name": "queries"}]
    head_parameters = [parameter for parameter in raw_model.mask_classifier.parameters() if parameter.requires_grad]
    if head_parameters:
        parameter_groups.append({"params": head_parameters, "lr": args.head_learning_rate, "name": "head"})
    lora_parameters = raw_model.lora_parameters()
    if args.use_lora:
        if not lora_parameters:
            raise RuntimeError("LoRA was requested but no LoRA parameters were injected.")
        parameter_groups.append({"params": lora_parameters, "lr": args.lora_learning_rate, "name": "lora"})
    optimizer = AdamW(parameter_groups, weight_decay=args.weight_decay)
    updates_per_epoch = math.ceil(len(loader) / args.gradient_accumulation)
    total_steps = updates_per_epoch * args.epochs
    scheduler = LambdaLR(optimizer, linear_schedule(total_steps, args.warmup_ratio))

    start_epoch = 0
    start_batch = 0
    global_step = 0
    if args.resume is not None:
        checkpoint = load_checkpoint(raw_model, args.resume, optimizer=optimizer, scheduler=scheduler)
        start_epoch = int(checkpoint.get("epoch", 0))
        start_batch = int(checkpoint.get("batch_in_epoch", 0))
        global_step = int(checkpoint.get("global_step", 0))
    if distributed:
        model = DistributedDataParallel(raw_model, device_ids=[local_rank], output_device=local_rank)

    trainable = sum(parameter.numel() for parameter in raw_model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in raw_model.parameters())
    settings = {
        **checkpoint_config(args),
        "train_json": [str(path) for path in args.train_json],
        "samples": len(dataset),
        "epochs": args.epochs,
        "per_gpu_batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "world_size": world_size,
        "effective_global_batch_size": args.batch_size * args.gradient_accumulation * world_size,
        "query_learning_rate": args.query_learning_rate,
        "head_learning_rate": args.head_learning_rate,
        "lora_learning_rate": args.lora_learning_rate,
        "trainable_parameters": trainable,
        "total_parameters": total,
        "total_optimizer_steps": total_steps,
        "cudnn_disabled": cudnn_disabled,
        "stamp_adapter_initialization": adapter_report,
    }
    if is_main:
        trainable_rows = [
            f"{name}\t{tuple(parameter.shape)}\t{parameter.numel()}"
            for name, parameter in raw_model.named_parameters()
            if parameter.requires_grad
        ]
        (args.output_dir / "trainable_parameters.txt").write_text(
            "\n".join(trainable_rows) + "\n", encoding="utf-8"
        )
        (args.output_dir / "train_config.json").write_text(
            json.dumps(settings, indent=2, ensure_ascii=True), encoding="utf-8"
        )
        print(json.dumps(settings, indent=2, ensure_ascii=True), flush=True)
        print(
            f"Trainable parameter listing written to: {args.output_dir / 'trainable_parameters.txt'}",
            flush=True,
        )

    model.train()
    optimizer.zero_grad(set_to_none=True)
    history_path = args.output_dir / "train_history.jsonl"
    autocast_enabled = device.type == "cuda" and args.dtype != "fp32"
    autocast_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        iterator = tqdm(
            loader,
            desc=f"onepass7b train epoch {epoch + 1}/{args.epochs}",
            dynamic_ncols=True,
            disable=not is_main,
        )
        running = {"loss": 0.0, "loss_bce": 0.0, "loss_dice": 0.0, "count": 0}
        for batch_index, samples in enumerate(iterator):
            if epoch == start_epoch and batch_index < start_batch:
                continue
            should_update = (
                (batch_index + 1) % args.gradient_accumulation == 0 or batch_index + 1 == len(loader)
            )
            sync_context = nullcontext()
            if distributed and not should_update:
                sync_context = model.no_sync()
            with sync_context:
                prepared = prepare_batch(samples, processor, device)
                targets = grid_targets(samples, prepared["grid_shapes"], device)
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
                    loss, parts = segmentation_loss(
                        outputs.mask_logits,
                        targets,
                        bce_weight=args.bce_weight,
                        dice_weight=args.dice_weight,
                    )
                    scaled_loss = loss / args.gradient_accumulation
                if is_main and epoch == start_epoch and batch_index == start_batch:
                    print(
                        json.dumps(
                            {
                                "first_batch_input_ids": list(prepared["input_ids"].shape),
                                "first_batch_pixel_values": list(prepared["pixel_values"].shape),
                                "first_batch_image_grid_thw": prepared["image_grid_thw"].detach().cpu().tolist(),
                                "first_batch_grid_shapes": prepared["grid_shapes"],
                                "first_batch_mask_logit_shapes": [list(value.shape) for value in outputs.mask_logits],
                                "first_batch_target_shapes": [list(value.shape) for value in targets],
                            },
                            ensure_ascii=True,
                        ),
                        flush=True,
                    )
                scaled_loss.backward()
            for key in ("loss", "loss_bce", "loss_dice"):
                running[key] += parts[key]
            running["count"] += 1

            if should_update:
                torch.nn.utils.clip_grad_norm_(raw_model.trainable_parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if is_main and global_step % args.logging_steps == 0:
                    count = max(running["count"], 1)
                    row = {
                        "epoch": epoch + 1,
                        "batch": batch_index + 1,
                        "global_step": global_step,
                        "loss": running["loss"] / count,
                        "loss_bce": running["loss_bce"] / count,
                        "loss_dice": running["loss_dice"] / count,
                        "learning_rates": {
                            group["name"]: group["lr"] for group in optimizer.param_groups
                        },
                    }
                    append_jsonl(history_path, row)
                    print(json.dumps({"train": row}, ensure_ascii=True), flush=True)
                    iterator.set_postfix(loss=f"{row['loss']:.4f}", step=global_step)
                    running = {"loss": 0.0, "loss_bce": 0.0, "loss_dice": 0.0, "count": 0}
                if is_main and args.save_steps > 0 and global_step % args.save_steps == 0:
                    save_checkpoint(
                        raw_model,
                        args.output_dir / f"onepass7b_step_{global_step}.pt",
                        config=checkpoint_config(args),
                        epoch=epoch,
                        batch_in_epoch=batch_index + 1,
                        global_step=global_step,
                        optimizer=optimizer,
                        scheduler=scheduler,
                    )
        start_batch = 0
        if is_main:
            save_checkpoint(
                raw_model,
                args.output_dir / f"onepass7b_epoch_{epoch + 1}.pt",
                config=checkpoint_config(args),
                epoch=epoch + 1,
                batch_in_epoch=0,
                global_step=global_step,
                optimizer=optimizer,
                scheduler=scheduler,
            )

    if is_main:
        final_path = save_checkpoint(
            raw_model,
            args.output_dir / "onepass_qwen7b.pt",
            config=checkpoint_config(args),
            epoch=args.epochs,
            batch_in_epoch=0,
            global_step=global_step,
        )
        print(f"OnePass Qwen2-VL-7B checkpoint written to: {final_path}", flush=True)
    if distributed:
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
