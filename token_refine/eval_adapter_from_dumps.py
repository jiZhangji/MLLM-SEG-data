from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from token_refine.data import TokenDumpDataset, collate_token_dumps
from token_refine.metrics import binary_iou, logits_to_mask
from token_refine.model import MaskTokenRefinementAdapter


def read_grid_hw(path: Path) -> tuple[int, int]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return tuple(payload["grid_hw"])


def grouped_batch_indices(paths: list[Path], batch_size: int) -> list[list[int]]:
    groups: dict[tuple[int, int], list[int]] = {}
    for index, path in enumerate(paths):
        groups.setdefault(read_grid_hw(path), []).append(index)
    batches = []
    for indices in groups.values():
        for start in range(0, len(indices), batch_size):
            batches.append(indices[start : start + batch_size])
    return batches


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-size", type=int, default=896)
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = ckpt.get("config", {})
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = MaskTokenRefinementAdapter(
        token_dim=int(ckpt["token_dim"]),
        hidden_size=int(config.get("hidden_size", 128)),
        use_uncertainty_gate=str(config.get("use_uncertainty_gate", "True")).lower() != "false",
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    paths = sorted(args.input_dir.glob("*.pt"))
    if args.limit > 0:
        paths = paths[: args.limit]
    if not paths:
        raise FileNotFoundError(f"No .pt dumps found in {args.input_dir}")
    loader = DataLoader(
        TokenDumpDataset(paths, args.image_size),
        batch_sampler=grouped_batch_indices(paths, args.batch_size),
        num_workers=args.num_workers,
        collate_fn=collate_token_dumps,
        pin_memory=device.type == "cuda",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with torch.no_grad():
        for batch in loader:
            mask_logits = batch["mask_logits"].to(device)
            mask_hidden = batch["mask_hidden"].to(device)
            gt_mask = batch["gt_mask"].to(device)
            target_tokens = batch["target_tokens"].to(device)
            grid_hw = batch["grid_hw"]
            outputs = model(mask_hidden, mask_logits)
            coarse_mask = logits_to_mask(mask_logits, grid_hw, args.image_size)
            refined_mask = logits_to_mask(outputs["refined_logits"], grid_hw, args.image_size)
            coarse_iou = binary_iou(coarse_mask >= 0.5, gt_mask >= 0.5)
            refined_iou = binary_iou(refined_mask >= 0.5, gt_mask >= 0.5)
            token_acc = (outputs["refined_logits"].argmax(dim=-1) == target_tokens).float().mean(dim=1)
            for index, name in enumerate(batch["name"]):
                rows.append(
                    {
                        "name": name,
                        "path": batch["path"][index],
                        "coarse_iou": float(coarse_iou[index].item()),
                        "refined_iou": float(refined_iou[index].item()),
                        "delta": float((refined_iou[index] - coarse_iou[index]).item()),
                        "token_acc": float(token_acc[index].item()),
                    }
                )
                print(json.dumps(rows[-1]), flush=True)

    csv_path = args.output_dir / "eval_rows.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "samples": len(rows),
        "coarse_iou": sum(row["coarse_iou"] for row in rows) / max(len(rows), 1),
        "refined_iou": sum(row["refined_iou"] for row in rows) / max(len(rows), 1),
        "delta": sum(row["delta"] for row in rows) / max(len(rows), 1),
        "token_acc": sum(row["token_acc"] for row in rows) / max(len(rows), 1),
        "rows_csv": str(csv_path),
    }
    (args.output_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
