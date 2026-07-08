from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from token_refine.data import TokenDumpDataset, collate_token_dumps
from token_refine.eval_adapter_from_dumps import load_original_mask, logits_to_mask_hw
from token_refine.metrics import binary_intersection_union, binary_iou, logits_to_mask
from token_refine.model import UncertaintyAwareMaskHead
from token_refine.train_adapter_from_dumps import grouped_batch_indices, load_or_build_grid_cache


def read_grid_hw(path: Path) -> tuple[int, int]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return tuple(payload["grid_hw"])


def build_grid_cache(paths: list[Path], cache_path: Path) -> dict[str, list[int]]:
    return {str(path.resolve()): list(read_grid_hw(path)) for path in paths}


def simple_load_or_build_grid_cache(paths: list[Path], cache_path: Path) -> dict[str, tuple[int, int]]:
    import time

    return load_or_build_grid_cache(paths, cache_path, rank=0, world_size=1, started_at=time.time())


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    coarse_intersection = sum(int(row["coarse_intersection"]) for row in rows)
    coarse_union = sum(int(row["coarse_union"]) for row in rows)
    head_intersection = sum(int(row["head_intersection"]) for row in rows)
    head_union = sum(int(row["head_union"]) for row in rows)
    original_coarse_intersection = sum(int(row["original_coarse_intersection"]) for row in rows)
    original_coarse_union = sum(int(row["original_coarse_union"]) for row in rows)
    original_head_intersection = sum(int(row["original_head_intersection"]) for row in rows)
    original_head_union = sum(int(row["original_head_union"]) for row in rows)

    coarse_mean_iou = sum(float(row["coarse_iou"]) for row in rows) / max(len(rows), 1)
    head_mean_iou = sum(float(row["head_iou"]) for row in rows) / max(len(rows), 1)
    original_coarse_mean_iou = sum(float(row["original_coarse_iou"]) for row in rows) / max(len(rows), 1)
    original_head_mean_iou = sum(float(row["original_head_iou"]) for row in rows) / max(len(rows), 1)
    coarse_ciou = coarse_intersection / coarse_union if coarse_union > 0 else None
    head_ciou = head_intersection / head_union if head_union > 0 else None
    original_coarse_ciou = original_coarse_intersection / original_coarse_union if original_coarse_union > 0 else None
    original_head_ciou = original_head_intersection / original_head_union if original_head_union > 0 else None

    return {
        "samples": len(rows),
        "coarse_iou": coarse_mean_iou,
        "head_iou": head_mean_iou,
        "delta": head_mean_iou - coarse_mean_iou,
        "coarse_mean_iou": coarse_mean_iou,
        "head_mean_iou": head_mean_iou,
        "mean_iou_delta": head_mean_iou - coarse_mean_iou,
        "coarse_gIoU": coarse_mean_iou,
        "head_gIoU": head_mean_iou,
        "gIoU_delta": head_mean_iou - coarse_mean_iou,
        "coarse_cIoU": coarse_ciou,
        "head_cIoU": head_ciou,
        "cIoU_delta": head_ciou - coarse_ciou if coarse_ciou is not None and head_ciou is not None else None,
        "coarse_original_mean_iou": original_coarse_mean_iou,
        "head_original_mean_iou": original_head_mean_iou,
        "original_mean_iou_delta": original_head_mean_iou - original_coarse_mean_iou,
        "coarse_original_gIoU": original_coarse_mean_iou,
        "head_original_gIoU": original_head_mean_iou,
        "original_gIoU_delta": original_head_mean_iou - original_coarse_mean_iou,
        "coarse_original_cIoU": original_coarse_ciou,
        "head_original_cIoU": original_head_ciou,
        "original_cIoU_delta": (
            original_head_ciou - original_coarse_ciou
            if original_coarse_ciou is not None and original_head_ciou is not None
            else None
        ),
        "token_acc": sum(float(row["token_acc"]) for row in rows) / max(len(rows), 1),
        "uncertainty_mae": sum(float(row["uncertainty_mae"]) for row in rows) / max(len(rows), 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-size", type=int, default=896)
    parser.add_argument("--grid-cache", type=Path, default=None)
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = ckpt.get("config", {})
    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    model = UncertaintyAwareMaskHead(
        token_dim=int(ckpt["token_dim"]),
        hidden_size=int(config.get("hidden_size", 128)),
        use_uncertainty_feature=str(config.get("use_uncertainty_feature", "True")).lower() != "false",
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    paths = sorted(args.input_dir.glob("*.pt"))
    if args.limit > 0:
        paths = paths[: args.limit]
    if not paths:
        raise FileNotFoundError(f"No .pt dumps found in {args.input_dir}")
    cache_path = args.grid_cache or (args.input_dir / "token_refine_grid_cache.json")
    grid_cache = simple_load_or_build_grid_cache(paths, cache_path)
    loader = DataLoader(
        TokenDumpDataset(paths, args.image_size),
        batch_sampler=grouped_batch_indices(paths, args.batch_size, False, 0, grid_cache),
        num_workers=args.num_workers,
        collate_fn=collate_token_dumps,
        pin_memory=device.type == "cuda",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    with torch.no_grad():
        for batch in loader:
            mask_hidden = batch["mask_hidden"].to(device)
            mask_logits = batch["mask_logits"].to(device)
            gt_mask = batch["gt_mask"].to(device)
            target_tokens = batch["target_tokens"].to(device)
            target_soft = batch["target_soft"].to(device)
            grid_hw = batch["grid_hw"]
            outputs = model(mask_hidden)
            head_logits = outputs["mask_logits"]

            coarse_mask = logits_to_mask(mask_logits, grid_hw, args.image_size)
            head_mask = logits_to_mask(head_logits, grid_hw, args.image_size)
            coarse_pred = coarse_mask >= 0.5
            head_pred = head_mask >= 0.5
            gt_pred = gt_mask >= 0.5
            coarse_iou = binary_iou(coarse_pred, gt_pred)
            head_iou = binary_iou(head_pred, gt_pred)
            coarse_inter, coarse_union = binary_intersection_union(coarse_pred, gt_pred)
            head_inter, head_union = binary_intersection_union(head_pred, gt_pred)
            token_acc = (head_logits.argmax(dim=-1) == target_tokens).float().mean(dim=1)
            target_uncertainty = 1.0 - torch.abs(2.0 * target_soft - 1.0)
            uncertainty_mae = (outputs["uncertainty"].squeeze(-1) - target_uncertainty).abs().mean(dim=1)

            for index, name in enumerate(batch["name"]):
                dump_path = Path(batch["path"][index])
                original_gt = load_original_mask(dump_path).to(device)
                original_hw = tuple(int(x) for x in original_gt.shape[-2:])
                original_coarse = logits_to_mask_hw(mask_logits[index : index + 1], grid_hw, original_hw) >= 0.5
                original_head = logits_to_mask_hw(head_logits[index : index + 1], grid_hw, original_hw) >= 0.5
                original_coarse_inter, original_coarse_union = binary_intersection_union(original_coarse, original_gt)
                original_head_inter, original_head_union = binary_intersection_union(original_head, original_gt)
                original_coarse_iou = original_coarse_inter.float() / original_coarse_union.float().clamp_min(1.0)
                original_head_iou = original_head_inter.float() / original_head_union.float().clamp_min(1.0)
                rows.append(
                    {
                        "name": name,
                        "path": batch["path"][index],
                        "coarse_iou": float(coarse_iou[index].item()),
                        "head_iou": float(head_iou[index].item()),
                        "delta": float((head_iou[index] - coarse_iou[index]).item()),
                        "coarse_intersection": int(coarse_inter[index].item()),
                        "coarse_union": int(coarse_union[index].item()),
                        "head_intersection": int(head_inter[index].item()),
                        "head_union": int(head_union[index].item()),
                        "original_coarse_iou": float(original_coarse_iou.item()),
                        "original_head_iou": float(original_head_iou.item()),
                        "original_delta": float((original_head_iou - original_coarse_iou).item()),
                        "original_coarse_intersection": int(original_coarse_inter.item()),
                        "original_coarse_union": int(original_coarse_union.item()),
                        "original_head_intersection": int(original_head_inter.item()),
                        "original_head_union": int(original_head_union.item()),
                        "token_acc": float(token_acc[index].item()),
                        "uncertainty_mae": float(uncertainty_mae[index].item()),
                    }
                )
                print(json.dumps(rows[-1]), flush=True)

    csv_path = args.output_dir / "eval_rows.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows)
    summary["rows_csv"] = str(csv_path)
    (args.output_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
