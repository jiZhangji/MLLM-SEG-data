from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from token_refine.data import TokenDumpDataset, collate_token_dumps
from token_refine.metrics import binary_intersection_union, binary_iou, logits_to_mask
from token_refine.model import MaskTokenRefinementAdapter


def read_grid_hw(path: Path) -> tuple[int, int]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return tuple(payload["grid_hw"])


def build_grid_cache(paths: list[Path], cache_path: Path) -> dict[str, list[int]]:
    return {str(path.resolve()): list(read_grid_hw(path)) for path in paths}


def load_or_build_grid_cache(paths: list[Path], cache_path: Path) -> dict[str, tuple[int, int]]:
    resolved_keys = {str(path.resolve()) for path in paths}
    rebuild = True
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            grids = cached.get("grids", {})
            rebuild = not resolved_keys.issubset(set(grids))
        except Exception:
            rebuild = True
    if rebuild:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        grids = build_grid_cache(paths, cache_path)
        tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
        tmp_path.write_text(json.dumps({"grids": grids}, indent=2), encoding="utf-8")
        tmp_path.replace(cache_path)
        print(f"[INFO] Wrote grid cache: {cache_path}", flush=True)
    else:
        print(f"[INFO] Using grid cache: {cache_path}", flush=True)
    cached = json.loads(cache_path.read_text(encoding="utf-8"))
    return {key: tuple(value) for key, value in cached["grids"].items() if key in resolved_keys}


def grouped_batch_indices(paths: list[Path], batch_size: int, grid_cache: dict[str, tuple[int, int]]) -> list[list[int]]:
    groups: dict[tuple[int, int], list[int]] = {}
    for index, path in enumerate(paths):
        groups.setdefault(grid_cache[str(path.resolve())], []).append(index)
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
    parser.add_argument("--grid-cache", type=Path, default=None)
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
    cache_path = args.grid_cache or (args.input_dir / "token_refine_grid_cache.json")
    grid_cache = load_or_build_grid_cache(paths, cache_path)
    loader = DataLoader(
        TokenDumpDataset(paths, args.image_size),
        batch_sampler=grouped_batch_indices(paths, args.batch_size, grid_cache),
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
            coarse_pred = coarse_mask >= 0.5
            refined_pred = refined_mask >= 0.5
            gt_pred = gt_mask >= 0.5
            coarse_iou = binary_iou(coarse_pred, gt_pred)
            refined_iou = binary_iou(refined_pred, gt_pred)
            coarse_inter, coarse_union = binary_intersection_union(coarse_pred, gt_pred)
            refined_inter, refined_union = binary_intersection_union(refined_pred, gt_pred)
            token_acc = (outputs["refined_logits"].argmax(dim=-1) == target_tokens).float().mean(dim=1)
            for index, name in enumerate(batch["name"]):
                rows.append(
                    {
                        "name": name,
                        "path": batch["path"][index],
                        "coarse_iou": float(coarse_iou[index].item()),
                        "refined_iou": float(refined_iou[index].item()),
                        "delta": float((refined_iou[index] - coarse_iou[index]).item()),
                        "coarse_intersection": int(coarse_inter[index].item()),
                        "coarse_union": int(coarse_union[index].item()),
                        "refined_intersection": int(refined_inter[index].item()),
                        "refined_union": int(refined_union[index].item()),
                        "token_acc": float(token_acc[index].item()),
                    }
                )
                print(json.dumps(rows[-1]), flush=True)

    csv_path = args.output_dir / "eval_rows.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    coarse_intersection = sum(row["coarse_intersection"] for row in rows)
    coarse_union = sum(row["coarse_union"] for row in rows)
    refined_intersection = sum(row["refined_intersection"] for row in rows)
    refined_union = sum(row["refined_union"] for row in rows)
    coarse_mean_iou = sum(row["coarse_iou"] for row in rows) / max(len(rows), 1)
    refined_mean_iou = sum(row["refined_iou"] for row in rows) / max(len(rows), 1)
    coarse_ciou = coarse_intersection / coarse_union if coarse_union > 0 else None
    refined_ciou = refined_intersection / refined_union if refined_union > 0 else None
    summary = {
        "samples": len(rows),
        "coarse_iou": coarse_mean_iou,
        "refined_iou": refined_mean_iou,
        "delta": refined_mean_iou - coarse_mean_iou,
        "coarse_mean_iou": coarse_mean_iou,
        "refined_mean_iou": refined_mean_iou,
        "mean_iou_delta": refined_mean_iou - coarse_mean_iou,
        "coarse_gIoU": coarse_mean_iou,
        "refined_gIoU": refined_mean_iou,
        "gIoU_delta": refined_mean_iou - coarse_mean_iou,
        "coarse_cIoU": coarse_ciou,
        "refined_cIoU": refined_ciou,
        "cIoU_delta": refined_ciou - coarse_ciou if coarse_ciou is not None and refined_ciou is not None else None,
        "coarse_total_intersection": coarse_intersection,
        "coarse_total_union": coarse_union,
        "refined_total_intersection": refined_intersection,
        "refined_total_union": refined_union,
        "token_acc": sum(row["token_acc"] for row in rows) / max(len(rows), 1),
        "rows_csv": str(csv_path),
    }
    (args.output_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
