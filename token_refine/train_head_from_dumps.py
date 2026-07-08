from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from token_refine.data import TokenDumpDataset, collate_token_dumps
from token_refine.metrics import binary_iou, logits_to_mask
from token_refine.model import UncertaintyAwareMaskHead
from token_refine.train_adapter_from_dumps import (
    find_dump_paths,
    grouped_batch_indices,
    load_or_build_grid_cache,
    progress_iter,
    split_paths,
)


def head_loss(
    outputs: dict[str, torch.Tensor],
    target_tokens: torch.Tensor,
    target_soft: torch.Tensor,
    uncertainty_ce_weight: float,
    uncertainty_bce_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    target_uncertainty = 1.0 - torch.abs(2.0 * target_soft - 1.0)
    ce = F.cross_entropy(
        outputs["mask_logits"].reshape(-1, 2),
        target_tokens.reshape(-1),
        reduction="none",
    ).reshape_as(target_tokens).float()
    ce_weights = 1.0 + uncertainty_ce_weight * target_uncertainty
    loss_ce = (ce * ce_weights).mean()
    loss_unc = F.binary_cross_entropy_with_logits(
        outputs["uncertainty_logits"].squeeze(-1),
        target_uncertainty,
    )
    loss = loss_ce + uncertainty_bce_weight * loss_unc
    with torch.no_grad():
        unc_mae = (outputs["uncertainty"].squeeze(-1) - target_uncertainty).abs().mean()
    return loss, {
        "loss_ce": float(loss_ce.item()),
        "loss_unc": float(loss_unc.item()),
        "uncertainty_mae": float(unc_mae.item()),
    }


@torch.no_grad()
def evaluate(model, loader, device, image_size: int, show_progress: bool, desc: str) -> dict[str, float]:
    model.eval()
    totals = {
        "coarse_iou": 0.0,
        "head_iou": 0.0,
        "token_acc": 0.0,
        "uncertainty_mae": 0.0,
    }
    count = 0
    iterator = progress_iter(loader, show_progress, desc=desc, total=len(loader), dynamic_ncols=True, leave=False)
    for batch in iterator:
        mask_hidden = batch["mask_hidden"].to(device)
        mask_logits = batch["mask_logits"].to(device)
        target_tokens = batch["target_tokens"].to(device)
        target_soft = batch["target_soft"].to(device)
        gt_mask = batch["gt_mask"].to(device)
        grid_hw = batch["grid_hw"]
        outputs = model(mask_hidden)

        coarse_mask = logits_to_mask(mask_logits, grid_hw, image_size)
        head_mask = logits_to_mask(outputs["mask_logits"], grid_hw, image_size)
        coarse_iou = binary_iou(coarse_mask >= 0.5, gt_mask >= 0.5).mean()
        head_iou = binary_iou(head_mask >= 0.5, gt_mask >= 0.5).mean()
        token_acc = (outputs["mask_logits"].argmax(dim=-1) == target_tokens).float().mean()
        target_uncertainty = 1.0 - torch.abs(2.0 * target_soft - 1.0)
        uncertainty_mae = (outputs["uncertainty"].squeeze(-1) - target_uncertainty).abs().mean()

        batch_size = mask_hidden.shape[0]
        totals["coarse_iou"] += float(coarse_iou.item()) * batch_size
        totals["head_iou"] += float(head_iou.item()) * batch_size
        totals["token_acc"] += float(token_acc.item()) * batch_size
        totals["uncertainty_mae"] += float(uncertainty_mae.item()) * batch_size
        count += batch_size
        if show_progress and hasattr(iterator, "set_postfix"):
            iterator.set_postfix(head_iou=f"{totals['head_iou'] / max(count, 1):.4f}")
    return {key: value / max(count, 1) for key, value in totals.items()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-size", type=int, default=896)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--uncertainty-ce-weight", type=float, default=2.0)
    parser.add_argument("--uncertainty-bce-weight", type=float, default=0.5)
    parser.add_argument("--use-uncertainty-feature", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--grid-cache", type=Path, default=None)
    parser.add_argument("--build-cache-only", action="store_true")
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    paths = find_dump_paths(args.input_dir, args.limit)
    train_paths, val_paths = split_paths(paths, args.val_fraction, args.seed)
    sample = torch.load(paths[0], map_location="cpu", weights_only=False)
    token_dim = int(sample["mask_hidden"].shape[-1])
    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")

    cache_path = args.grid_cache or (args.input_dir / "token_refine_grid_cache.json")
    grid_cache = load_or_build_grid_cache(paths, cache_path, rank=0, world_size=1, started_at=time.time())
    if args.build_cache_only:
        print(f"[INFO] Built grid cache for {len(grid_cache)} dumps: {cache_path}", flush=True)
        return 0

    train_loader = DataLoader(
        TokenDumpDataset(train_paths, args.image_size),
        batch_sampler=grouped_batch_indices(train_paths, args.batch_size, True, args.seed, grid_cache),
        num_workers=args.num_workers,
        collate_fn=collate_token_dumps,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        TokenDumpDataset(val_paths or train_paths, args.image_size),
        batch_sampler=grouped_batch_indices(val_paths or train_paths, args.batch_size, False, args.seed, grid_cache),
        num_workers=args.num_workers,
        collate_fn=collate_token_dumps,
        pin_memory=device.type == "cuda",
    )

    model = UncertaintyAwareMaskHead(
        token_dim=token_dim,
        hidden_size=args.hidden_size,
        use_uncertainty_feature=args.use_uncertainty_feature,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[INFO] train={len(train_paths)} val={len(val_paths)} train_batches={len(train_loader)} "
        f"val_batches={len(val_loader)} device={device}",
        flush=True,
    )

    best_head_iou = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        loss_parts = {"loss_ce": 0.0, "loss_unc": 0.0, "uncertainty_mae": 0.0}
        iterator = progress_iter(
            train_loader,
            args.progress,
            desc=f"epoch {epoch}/{args.epochs} train",
            total=len(train_loader),
            dynamic_ncols=True,
            leave=True,
        )
        for batch in iterator:
            mask_hidden = batch["mask_hidden"].to(device)
            target_tokens = batch["target_tokens"].to(device)
            target_soft = batch["target_soft"].to(device)
            outputs = model(mask_hidden)
            loss, parts = head_loss(
                outputs,
                target_tokens,
                target_soft,
                args.uncertainty_ce_weight,
                args.uncertainty_bce_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            batch_size = mask_hidden.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_count += batch_size
            for key, value in parts.items():
                loss_parts[key] += value * batch_size
            if args.progress and hasattr(iterator, "set_postfix"):
                iterator.set_postfix(loss=f"{total_loss / max(total_count, 1):.5f}")

        val_metrics = evaluate(
            model,
            val_loader,
            device,
            args.image_size,
            show_progress=args.progress,
            desc=f"epoch {epoch}/{args.epochs} val",
        )
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(total_count, 1),
            "train_loss_parts": {key: value / max(total_count, 1) for key, value in loss_parts.items()},
            "val": val_metrics,
            "val_delta_vs_coarse": val_metrics["head_iou"] - val_metrics["coarse_iou"],
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if val_metrics["head_iou"] > best_head_iou:
            best_head_iou = val_metrics["head_iou"]
            safe_config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": safe_config,
                    "token_dim": token_dim,
                    "history": history,
                },
                args.output_dir / "head.pt",
            )

    report = {
        "train_count": len(train_paths),
        "val_count": len(val_paths),
        "best_head_iou": best_head_iou,
        "history": history,
    }
    (args.output_dir / "train_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
