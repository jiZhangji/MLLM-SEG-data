from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from token_refine.data import CachedTokenDataset, collate_token_dumps, load_token_cache
from token_refine.model import IndependentUncertaintyMaskHead
from token_refine.train_adapter_from_dumps import grouped_batch_indices_for_items, progress_iter, split_items


def mask_and_uncertainty_loss(
    outputs: dict[str, torch.Tensor],
    target_tokens: torch.Tensor,
    target_soft: torch.Tensor,
    uncertainty_ce_weight: float,
    soft_bce_weight: float,
    uncertainty_bce_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    logits = outputs["mask_logits"].float()
    foreground_logits = logits[..., 1] - logits[..., 0]
    boundary_uncertainty = 1.0 - torch.abs(2.0 * target_soft - 1.0)
    prediction_error = (torch.sigmoid(foreground_logits.detach()) - target_soft).abs()
    uncertainty_target = 1.0 - (1.0 - boundary_uncertainty) * (1.0 - prediction_error)

    ce = F.cross_entropy(
        logits.reshape(-1, 2),
        target_tokens.reshape(-1),
        reduction="none",
    ).reshape_as(target_tokens)
    loss_ce = (ce * (1.0 + uncertainty_ce_weight * uncertainty_target)).mean()
    loss_soft_bce = F.binary_cross_entropy_with_logits(foreground_logits, target_soft)
    loss_uncertainty = F.binary_cross_entropy_with_logits(
        outputs["uncertainty_logits"].float().squeeze(-1),
        uncertainty_target,
    )
    loss = loss_ce + soft_bce_weight * loss_soft_bce + uncertainty_bce_weight * loss_uncertainty
    with torch.no_grad():
        uncertainty_mae = (outputs["uncertainty"].float().squeeze(-1) - uncertainty_target).abs().mean()
    return loss, {
        "loss_ce": float(loss_ce.item()),
        "loss_soft_bce": float(loss_soft_bce.item()),
        "loss_uncertainty": float(loss_uncertainty.item()),
        "uncertainty_mae": float(uncertainty_mae.item()),
    }


@torch.no_grad()
def evaluate(model, loader, device, amp_enabled: bool, show_progress: bool, desc: str) -> dict[str, float]:
    model.eval()
    totals = {"stamp_token_iou": 0.0, "head_token_iou": 0.0, "token_acc": 0.0}
    count = 0
    iterator = progress_iter(loader, show_progress, desc=desc, total=len(loader), dynamic_ncols=True, leave=False)
    for batch in iterator:
        mask_hidden = batch["mask_hidden"].to(device, non_blocking=True)
        stamp_logits = batch["mask_logits"].to(device, non_blocking=True)
        target_tokens = batch["target_tokens"].to(device, non_blocking=True)
        grid_hw = batch["grid_hw"]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp_enabled):
            outputs = model(mask_hidden, grid_hw)
        head_tokens = outputs["mask_logits"].argmax(dim=-1)
        stamp_tokens = stamp_logits.argmax(dim=-1)
        target_bool = target_tokens.bool()
        stamp_inter = (stamp_tokens.bool() & target_bool).sum(dim=1).float()
        stamp_union = (stamp_tokens.bool() | target_bool).sum(dim=1).clamp_min(1).float()
        head_inter = (head_tokens.bool() & target_bool).sum(dim=1).float()
        head_union = (head_tokens.bool() | target_bool).sum(dim=1).clamp_min(1).float()
        batch_size = mask_hidden.shape[0]
        totals["stamp_token_iou"] += float((stamp_inter / stamp_union).sum().item())
        totals["head_token_iou"] += float((head_inter / head_union).sum().item())
        totals["token_acc"] += float((head_tokens == target_tokens).float().mean(dim=1).sum().item())
        count += batch_size
    return {key: value / max(count, 1) for key, value in totals.items()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-file", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--uncertainty-ce-weight", type=float, default=2.0)
    parser.add_argument("--soft-bce-weight", type=float, default=0.5)
    parser.add_argument("--uncertainty-bce-weight", type=float, default=0.3)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mmap-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--disable-cudnn", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if args.disable_cudnn:
        torch.backends.cudnn.enabled = False
        print("[INFO] cuDNN disabled; CUDA remains enabled.", flush=True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")
    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    amp_enabled = args.amp and device.type == "cuda"

    started_at = time.time()
    print(f"[INFO] Loading consolidated token cache: {args.cache_file}", flush=True)
    cache = load_token_cache(args.cache_file, mmap=args.mmap_cache)
    items = cache["items"]
    if args.limit > 0:
        items = items[: args.limit]
    if not items:
        raise ValueError("Token cache is empty.")
    train_items, val_items = split_items(items, args.val_fraction, args.seed)
    token_dim = int(items[0]["mask_hidden"].shape[-1])

    train_loader = DataLoader(
        CachedTokenDataset(train_items),
        batch_sampler=grouped_batch_indices_for_items(train_items, args.batch_size, True, args.seed, 0, 1),
        num_workers=0,
        collate_fn=collate_token_dumps,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        CachedTokenDataset(val_items or train_items),
        batch_sampler=grouped_batch_indices_for_items(val_items or train_items, args.batch_size, False, args.seed, 0, 1),
        num_workers=0,
        collate_fn=collate_token_dumps,
        pin_memory=device.type == "cuda",
    )

    model = IndependentUncertaintyMaskHead(token_dim, args.hidden_size, args.depth).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[INFO] cache_items={len(items)} train={len(train_items)} val={len(val_items)} "
        f"train_batches={len(train_loader)} val_batches={len(val_loader)} token_dim={token_dim} "
        f"device={device} load_seconds={time.time() - started_at:.2f}",
        flush=True,
    )

    best_iou = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = {"loss": 0.0, "loss_ce": 0.0, "loss_soft_bce": 0.0, "loss_uncertainty": 0.0, "uncertainty_mae": 0.0}
        count = 0
        iterator = progress_iter(
            train_loader,
            args.progress,
            desc=f"epoch {epoch}/{args.epochs} train",
            total=len(train_loader),
            dynamic_ncols=True,
            leave=True,
        )
        for batch in iterator:
            mask_hidden = batch["mask_hidden"].to(device, non_blocking=True)
            target_tokens = batch["target_tokens"].to(device, non_blocking=True)
            target_soft = batch["target_soft"].to(device, non_blocking=True)
            grid_hw = batch["grid_hw"]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp_enabled):
                outputs = model(mask_hidden, grid_hw)
                loss, parts = mask_and_uncertainty_loss(
                    outputs,
                    target_tokens,
                    target_soft,
                    args.uncertainty_ce_weight,
                    args.soft_bce_weight,
                    args.uncertainty_bce_weight,
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            batch_size = mask_hidden.shape[0]
            totals["loss"] += float(loss.item()) * batch_size
            for key, value in parts.items():
                totals[key] += value * batch_size
            count += batch_size
            if args.progress and hasattr(iterator, "set_postfix"):
                iterator.set_postfix(loss=f"{totals['loss'] / max(count, 1):.5f}")

        val_metrics = evaluate(model, val_loader, device, amp_enabled, args.progress, f"epoch {epoch}/{args.epochs} val")
        row = {
            "epoch": epoch,
            "train": {key: value / max(count, 1) for key, value in totals.items()},
            "val": val_metrics,
            "val_delta_vs_stamp": val_metrics["head_token_iou"] - val_metrics["stamp_token_iou"],
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if val_metrics["head_token_iou"] > best_iou:
            best_iou = val_metrics["head_token_iou"]
            safe_config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
            torch.save(
                {
                    "format_version": 1,
                    "model_state_dict": model.state_dict(),
                    "token_dim": token_dim,
                    "hidden_size": args.hidden_size,
                    "depth": args.depth,
                    "config": safe_config,
                    "history": history,
                },
                args.output_dir / "independent_uncertainty_head.pt",
            )

    report = {"best_head_token_iou": best_iou, "history": history}
    (args.output_dir / "train_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
