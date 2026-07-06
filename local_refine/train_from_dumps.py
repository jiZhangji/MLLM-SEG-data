from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from local_refine.data import DumpDataset, collate_dump_batch
from local_refine.model import LocalPatchRefiner
from local_refine.ops import binary_iou, crop_mask_targets
from local_refine.selector import PatchSelector


def split_paths(paths: list[Path], val_fraction: float, seed: int) -> tuple[list[Path], list[Path]]:
    rng = random.Random(seed)
    shuffled = list(paths)
    rng.shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * val_fraction))) if len(shuffled) > 1 else 0
    return shuffled[val_count:], shuffled[:val_count]


def run_epoch(model, selector, loader, optimizer, device, train: bool) -> dict[str, float]:
    model.train(train)
    total_loss = 0.0
    total_iou = 0.0
    total_count = 0
    for batch in loader:
        images = batch["image"].to(device)
        gt_mask = batch["gt_mask"].to(device)
        mask_logits = batch["mask_logits"].to(device)
        mask_hidden = batch["mask_hidden"].to(device)
        grid_hw = batch["grid_hw"]

        with torch.no_grad():
            selected_ids = selector(mask_logits, grid_hw)["selected_ids"]

        output = model(images, mask_hidden, selected_ids, grid_hw)
        targets = crop_mask_targets(gt_mask, output["boxes"], model.output_size)
        loss = F.binary_cross_entropy_with_logits(output["local_logits"], targets)

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        with torch.no_grad():
            pred = torch.sigmoid(output["local_logits"]) >= 0.5
            iou = binary_iou(pred.squeeze(2), targets.squeeze(2) >= 0.5).mean()
        total_loss += float(loss.item()) * images.shape[0]
        total_iou += float(iou.item()) * images.shape[0]
        total_count += images.shape[0]
    return {"loss": total_loss / max(total_count, 1), "patch_iou": total_iou / max(total_count, 1)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--selector", choices=["random", "uncertainty", "boundary", "hybrid"], default="hybrid")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-size", type=int, default=896)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--boundary-weight", type=float, default=0.5)
    parser.add_argument("--token-channels", type=int, default=32)
    parser.add_argument("--crop-size", type=int, default=64)
    parser.add_argument("--output-size", type=int, default=16)
    parser.add_argument("--box-scale", type=float, default=1.0)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    paths = sorted(args.input_dir.glob("*.pt"))
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        raise FileNotFoundError(f"No .pt dumps found in {args.input_dir}")
    train_paths, val_paths = split_paths(paths, args.val_fraction, args.seed)
    sample = torch.load(paths[0], map_location="cpu", weights_only=False)
    token_dim = int(sample["mask_hidden"].shape[-1])

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    selector = PatchSelector(args.top_k, args.boundary_weight, args.selector).to(device)
    model = LocalPatchRefiner(
        token_dim=token_dim,
        token_channels=args.token_channels,
        crop_size=args.crop_size,
        output_size=args.output_size,
        box_scale=args.box_scale,
        chunk_size=args.chunk_size,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    train_loader = DataLoader(DumpDataset(train_paths, args.image_size), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_dump_batch)
    val_loader = DataLoader(DumpDataset(val_paths or train_paths, args.image_size), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_dump_batch)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, selector, train_loader, optimizer, device, True)
        val_metrics = run_epoch(model, selector, val_loader, optimizer, device, False)
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row)
        print(json.dumps(row), flush=True)
        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            safe_config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
            torch.save({"model": model.state_dict(), "config": safe_config, "token_dim": token_dim, "selector": args.selector}, args.output_dir / "best.pt")

    report = {"train_count": len(train_paths), "val_count": len(val_paths), "best_val_loss": best_loss, "history": history}
    (args.output_dir / "train_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
