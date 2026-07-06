from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from refine_stamp.data.dump_dataset import (
    RefinementDumpDataset,
    collate_refinement_dumps,
    find_dump_paths,
    split_paths,
    torch_load_trusted,
)
from refine_stamp.losses.refinement_losses import crop_gt_masks, refinement_loss
from refine_stamp.models.local_refiner import LocalPatchRefiner
from refine_stamp.models.patch_selector import PatchSelector


def infer_token_dim(path: Path) -> int:
    payload = torch_load_trusted(path, map_location="cpu")
    hidden = payload["mask_hidden"]
    return int(hidden.shape[-1])


def json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def train(args: argparse.Namespace) -> dict:
    if args.disable_cudnn:
        torch.backends.cudnn.enabled = False
        print("[INFO] cuDNN disabled for Stage-2 local refiner training.")

    paths = find_dump_paths(args.input_dir, args.limit)
    train_paths, val_paths = split_paths(paths, val_fraction=args.val_fraction, seed=args.seed)
    token_dim = infer_token_dim(train_paths[0])
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    train_dataset = RefinementDumpDataset(train_paths, image_size=args.image_size)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_refinement_dumps,
        pin_memory=device.type == "cuda",
    )

    model = LocalPatchRefiner(
        token_dim=token_dim,
        token_channels=args.token_channels,
        crop_size=args.crop_size,
        output_size=args.output_size,
        box_scale=args.box_scale,
        refiner_chunk_size=args.refiner_chunk_size,
    ).to(device)
    selector = PatchSelector(top_k=args.top_k, boundary_weight=args.boundary_weight, mode=args.selector)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    history = []
    for epoch in range(args.epochs):
        model.train()
        totals = {"loss": 0.0, "loss_bce": 0.0, "loss_dice": 0.0}
        steps = 0
        for batch in tqdm(train_loader, desc=f"train epoch {epoch + 1}/{args.epochs}"):
            images = batch["images"].to(device)
            masks = batch["masks"].to(device)
            mask_logits = batch["mask_logits"].to(device)
            mask_hidden = batch["mask_hidden"].to(device)
            grid_hw = batch["grid_hw"]

            with torch.no_grad():
                selected_ids = selector(mask_logits=mask_logits, grid_hw=grid_hw)["selected_ids"].to(device)
            outputs = model(
                images=images,
                mask_hidden=mask_hidden,
                selected_ids=selected_ids,
                grid_hw=grid_hw,
            )
            targets = crop_gt_masks(masks, outputs["boxes"], output_size=outputs["local_logits"].shape[-1])
            losses = refinement_loss(outputs["local_logits"], targets)

            optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            for key in totals:
                totals[key] += float(losses[key].item())
            steps += 1

        epoch_metrics = {key: value / max(steps, 1) for key, value in totals.items()}
        epoch_metrics["epoch"] = epoch + 1
        history.append(epoch_metrics)
        print(json.dumps(epoch_metrics, indent=2))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.output_dir / "local_refiner.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "token_dim": token_dim,
            "selector": args.selector,
            "top_k": args.top_k,
            "config": json_safe(vars(args)),
            "train_paths": [str(p) for p in train_paths],
            "val_paths": [str(p) for p in val_paths],
            "history": history,
        },
        ckpt_path,
    )
    report = {
        "checkpoint": str(ckpt_path),
        "num_paths": len(paths),
        "num_train": len(train_paths),
        "num_val": len(val_paths),
        "token_dim": token_dim,
        "history": history,
    }
    (args.output_dir / "train_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


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
    parser.add_argument("--refiner-chunk-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--disable-cudnn",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disable cuDNN for the small local refiner. This does not affect STAMP because Stage 2 trains from dumps.",
    )
    args = parser.parse_args()
    report = train(args)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
