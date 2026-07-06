from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from token_refine.data import TokenDumpDataset, collate_token_dumps
from token_refine.metrics import binary_iou, logits_to_mask
from token_refine.model import MaskTokenRefinementAdapter


def find_dump_paths(input_dir: Path, limit: int) -> list[Path]:
    paths = sorted(input_dir.glob("*.pt"))
    if limit > 0:
        paths = paths[:limit]
    if not paths:
        raise FileNotFoundError(f"No .pt dumps found in {input_dir}")
    return paths


def split_paths(paths: list[Path], val_fraction: float, seed: int) -> tuple[list[Path], list[Path]]:
    rng = random.Random(seed)
    shuffled = list(paths)
    rng.shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * val_fraction))) if len(shuffled) > 1 else 0
    return shuffled[val_count:], shuffled[:val_count]


def token_loss(
    outputs: dict[str, torch.Tensor],
    target_tokens: torch.Tensor,
    uncertainty_loss_weight: float,
    delta_reg_weight: float,
) -> torch.Tensor:
    refined_logits = outputs["refined_logits"]
    uncertainty = outputs["uncertainty"].squeeze(-1).detach()
    ce = F.cross_entropy(
        refined_logits.reshape(-1, 2),
        target_tokens.reshape(-1),
        reduction="none",
    ).reshape_as(target_tokens).float()
    weights = 1.0 + uncertainty_loss_weight * uncertainty
    loss_ce = (ce * weights).mean()
    delta_reg = ((1.0 - uncertainty) * outputs["delta_logits"].pow(2).sum(dim=-1)).mean()
    return loss_ce + delta_reg_weight * delta_reg


@torch.no_grad()
def evaluate(model, loader, device, image_size: int) -> dict[str, float]:
    model.eval()
    totals = {"coarse_iou": 0.0, "refined_iou": 0.0, "token_acc": 0.0}
    count = 0
    for batch in loader:
        mask_logits = batch["mask_logits"].to(device)
        mask_hidden = batch["mask_hidden"].to(device)
        target_tokens = batch["target_tokens"].to(device)
        gt_mask = batch["gt_mask"].to(device)
        grid_hw = batch["grid_hw"]
        outputs = model(mask_hidden, mask_logits)

        coarse_mask = logits_to_mask(mask_logits, grid_hw, image_size)
        refined_mask = logits_to_mask(outputs["refined_logits"], grid_hw, image_size)
        coarse_iou = binary_iou(coarse_mask >= 0.5, gt_mask >= 0.5).mean()
        refined_iou = binary_iou(refined_mask >= 0.5, gt_mask >= 0.5).mean()
        token_acc = (outputs["refined_logits"].argmax(dim=-1) == target_tokens).float().mean()

        batch_size = mask_logits.shape[0]
        totals["coarse_iou"] += float(coarse_iou.item()) * batch_size
        totals["refined_iou"] += float(refined_iou.item()) * batch_size
        totals["token_acc"] += float(token_acc.item()) * batch_size
        count += batch_size
    return {key: value / max(count, 1) for key, value in totals.items()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-size", type=int, default=896)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--uncertainty-loss-weight", type=float, default=2.0)
    parser.add_argument("--delta-reg-weight", type=float, default=0.01)
    parser.add_argument("--use-uncertainty-gate", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    paths = find_dump_paths(args.input_dir, args.limit)
    train_paths, val_paths = split_paths(paths, args.val_fraction, args.seed)
    sample = torch.load(paths[0], map_location="cpu", weights_only=False)
    token_dim = int(sample["mask_hidden"].shape[-1])

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = MaskTokenRefinementAdapter(
        token_dim=token_dim,
        hidden_size=args.hidden_size,
        use_uncertainty_gate=args.use_uncertainty_gate,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    train_loader = DataLoader(
        TokenDumpDataset(train_paths, args.image_size),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_token_dumps,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        TokenDumpDataset(val_paths or train_paths, args.image_size),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_token_dumps,
        pin_memory=device.type == "cuda",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_refined_iou = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        for batch in train_loader:
            mask_logits = batch["mask_logits"].to(device)
            mask_hidden = batch["mask_hidden"].to(device)
            target_tokens = batch["target_tokens"].to(device)
            outputs = model(mask_hidden, mask_logits)
            loss = token_loss(outputs, target_tokens, args.uncertainty_loss_weight, args.delta_reg_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.item()) * mask_logits.shape[0]
            total_count += mask_logits.shape[0]

        val_metrics = evaluate(model, val_loader, device, args.image_size)
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(total_count, 1),
            "val": val_metrics,
            "val_delta": val_metrics["refined_iou"] - val_metrics["coarse_iou"],
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if val_metrics["refined_iou"] > best_refined_iou:
            best_refined_iou = val_metrics["refined_iou"]
            safe_config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": safe_config,
                    "token_dim": token_dim,
                    "history": history,
                },
                args.output_dir / "adapter.pt",
            )

    report = {
        "train_count": len(train_paths),
        "val_count": len(val_paths),
        "best_refined_iou": best_refined_iou,
        "history": history,
    }
    (args.output_dir / "train_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

