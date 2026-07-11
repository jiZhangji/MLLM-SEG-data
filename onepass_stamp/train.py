from __future__ import annotations

import argparse
import json
import math
import os
import random
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
from .data import OnePassDataset, collate_samples, grid_targets, prepare_onepass_batch
from .model import onepass_mask_loss
from .runtime import configure_cudnn, load_onepass_model


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class EpochRandomSampler(Sampler[int]):
    """Deterministic per-epoch shuffling so mid-epoch resume skips the same data."""

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
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        remaining = max(total_steps - warmup_steps, 1)
        return max(0.0, float(total_steps - step) / float(remaining))

    return schedule


def checkpoint_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "model_name": str(args.model_name),
        "stamp_code_dir": str(args.stamp_code_dir),
        "use_task_token": bool(args.use_task_token),
        "min_pixels": int(args.min_pixels),
        "max_pixels": int(args.max_pixels),
        "attn_implementation": args.attn_implementation,
        "dtype": args.dtype,
        "dice_weight": float(args.dice_weight),
        "use_lora": bool(args.use_lora),
        "lora_rank": int(args.lora_rank),
        "lora_alpha": float(args.lora_alpha),
        "lora_dropout": float(args.lora_dropout),
    }


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the final query-driven OnePass-STAMP model.")
    parser.add_argument("--stamp-code-dir", type=Path, required=True)
    parser.add_argument("--model-name", type=Path, required=True)
    parser.add_argument("--train-json", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--query-learning-rate", type=float, default=3e-4)
    parser.add_argument("--head-learning-rate", type=float, default=3e-5)
    parser.add_argument("--lora-learning-rate", type=float, default=2e-5)
    parser.add_argument("--use-lora", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--dice-weight", type=float, default=1.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--min-pixels", type=int, default=802816)
    parser.add_argument("--max-pixels", type=int, default=1003520)
    parser.add_argument("--use-task-token", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--disable-cudnn", action=argparse.BooleanOptionalAction, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.gradient_accumulation <= 0:
        raise ValueError("epochs, batch-size, and gradient-accumulation must be positive.")
    if args.logging_steps <= 0:
        raise ValueError("logging-steps must be positive.")
    if not 0.0 <= args.warmup_ratio < 1.0:
        raise ValueError("warmup-ratio must be in [0, 1).")
    if args.use_lora and args.lora_rank <= 0:
        raise ValueError("lora-rank must be positive when LoRA is enabled.")
    if args.limit < 0 or args.offset < 0:
        raise ValueError("limit and offset must be non-negative.")
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank() if distributed else 0
    is_main = rank == 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed + rank)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    device = torch.device(f"cuda:{local_rank}" if distributed else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    cudnn_disabled = configure_cudnn(args.disable_cudnn, device)
    if is_main:
        print(f"DDP world size: {dist.get_world_size() if distributed else 1}", flush=True)
        print(f"cuDNN disabled: {int(cudnn_disabled)}", flush=True)

    dataset = OnePassDataset(args.train_json, args.data_root, limit=args.limit, offset=args.offset)
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

    model, processor = load_onepass_model(
        stamp_code_dir=args.stamp_code_dir,
        model_name=args.model_name,
        device=device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        use_task_token=args.use_task_token,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    raw_model = model
    query_parameters = list(raw_model.query_module.parameters())
    head_parameters = list(raw_model.mask_classifier.parameters())
    parameter_groups = [
        {"params": query_parameters, "lr": args.query_learning_rate, "name": "queries"},
        {"params": head_parameters, "lr": args.head_learning_rate, "name": "mask_head"},
    ]
    if args.use_lora:
        lora_parameters = raw_model.lora_parameters()
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
        "train_json": str(args.train_json),
        "samples": len(dataset),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "query_learning_rate": args.query_learning_rate,
        "head_learning_rate": args.head_learning_rate,
        "lora_learning_rate": args.lora_learning_rate,
        "trainable_parameters": trainable,
        "total_parameters": total,
        "total_optimizer_steps": total_steps,
    }
    if is_main:
        (args.output_dir / "train_config.json").write_text(
            json.dumps(settings, indent=2, ensure_ascii=True), encoding="utf-8"
        )
        print(json.dumps(settings, indent=2, ensure_ascii=True), flush=True)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    running = {"loss": 0.0, "loss_bce": 0.0, "loss_dice": 0.0, "count": 0}
    log_path = args.output_dir / "train_history.jsonl"
    autocast_enabled = device.type == "cuda" and args.dtype != "fp32"
    autocast_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        iterator = tqdm(
            loader,
            desc=f"onepass train epoch {epoch + 1}/{args.epochs}",
            dynamic_ncols=True,
            disable=not is_main,
        )
        for batch_index, samples in enumerate(iterator):
            if epoch == start_epoch and batch_index < start_batch:
                continue
            prepared = prepare_onepass_batch(samples, processor, device, args.use_task_token)
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
                loss, parts = onepass_mask_loss(outputs.mask_logits, targets, args.dice_weight)
                scaled_loss = loss / args.gradient_accumulation
            scaled_loss.backward()

            for key in ("loss", "loss_bce", "loss_dice"):
                running[key] += parts[key]
            running["count"] += 1
            should_update = (
                (batch_index + 1) % args.gradient_accumulation == 0 or batch_index + 1 == len(loader)
            )
            if should_update:
                torch.nn.utils.clip_grad_norm_(raw_model.trainable_parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if is_main and global_step % args.logging_steps == 0:
                    count = max(int(running["count"]), 1)
                    row = {
                        "epoch": epoch + 1,
                        "batch": batch_index + 1,
                        "global_step": global_step,
                        "loss": running["loss"] / count,
                        "loss_bce": running["loss_bce"] / count,
                        "loss_dice": running["loss_dice"] / count,
                        "query_lr": optimizer.param_groups[0]["lr"],
                        "head_lr": optimizer.param_groups[1]["lr"],
                    }
                    if len(optimizer.param_groups) > 2:
                        row["lora_lr"] = optimizer.param_groups[2]["lr"]
                    append_jsonl(log_path, row)
                    iterator.set_postfix(loss=f"{row['loss']:.4f}", step=global_step)
                    running = {"loss": 0.0, "loss_bce": 0.0, "loss_dice": 0.0, "count": 0}

                if is_main and args.save_steps > 0 and global_step % args.save_steps == 0:
                    save_checkpoint(
                        raw_model,
                        args.output_dir / f"onepass_step_{global_step}.pt",
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
                args.output_dir / f"onepass_epoch_{epoch + 1}.pt",
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
            args.output_dir / "onepass_stamp.pt",
            config=checkpoint_config(args),
            epoch=args.epochs,
            batch_in_epoch=0,
            global_step=global_step,
        )
        print(f"OnePass-STAMP checkpoint written to: {final_path}", flush=True)
    if distributed:
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
