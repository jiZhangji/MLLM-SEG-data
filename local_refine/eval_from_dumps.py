from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from local_refine.data import DumpDataset, collate_dump_batch
from local_refine.model import LocalPatchRefiner
from local_refine.ops import binary_iou, crop_mask_targets, stitch_logits
from local_refine.selector import PatchSelector


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--selector", choices=["random", "uncertainty", "boundary", "hybrid"], default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-size", type=int, default=896)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--boundary-weight", type=float, default=0.5)
    parser.add_argument("--blend-weight", type=float, default=0.8)
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = ckpt.get("config", {})
    selector_name = args.selector or ckpt.get("selector") or config.get("selector", "hybrid")
    top_k = args.top_k or int(config.get("top_k", 64))
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    model = LocalPatchRefiner(
        token_dim=int(ckpt.get("token_dim", config.get("token_dim", 32))),
        token_channels=int(config.get("token_channels", 32)),
        crop_size=int(config.get("crop_size", 64)),
        output_size=int(config.get("output_size", 16)),
        box_scale=float(config.get("box_scale", 1.0)),
        chunk_size=int(config.get("chunk_size", 16)),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    selector = PatchSelector(top_k, args.boundary_weight, selector_name).to(device)

    paths = sorted(args.input_dir.glob("*.pt"))
    if args.limit:
        paths = paths[: args.limit]
    loader = DataLoader(DumpDataset(paths, args.image_size), batch_size=1, shuffle=False, collate_fn=collate_dump_batch)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            gt_mask = batch["gt_mask"].to(device)
            mask_logits = batch["mask_logits"].to(device)
            mask_hidden = batch["mask_hidden"].to(device)
            grid_hw = batch["grid_hw"]
            selected_ids = selector(mask_logits, grid_hw)["selected_ids"]
            output = model(images, mask_hidden, selected_ids, grid_hw)

            coarse_fg = F.interpolate(
                torch.softmax(mask_logits, dim=-1)[..., 1].reshape(1, 1, *grid_hw),
                size=(args.image_size, args.image_size),
                mode="bilinear",
                align_corners=False,
            )
            refined_fg = torch.sigmoid(stitch_logits(mask_logits, output["local_logits"], output["boxes"], (args.image_size, args.image_size), args.blend_weight))
            targets = crop_mask_targets(gt_mask, output["boxes"], model.output_size)
            row = {
                "name": batch["name"][0],
                "path": batch["path"][0],
                "coarse_iou": float(binary_iou(coarse_fg >= 0.5, gt_mask >= 0.5).item()),
                "refined_iou": float(binary_iou(refined_fg >= 0.5, gt_mask >= 0.5).item()),
                "patch_iou": float(binary_iou(torch.sigmoid(output["local_logits"]).squeeze(2) >= 0.5, targets.squeeze(2) >= 0.5).mean().item()),
                "top_k": int(selected_ids.shape[1]),
            }
            rows.append(row)
            print(json.dumps(row), flush=True)

    csv_path = args.output_dir / "eval_rows.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["name"])
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "samples": len(rows),
        "selector": selector_name,
        "top_k": top_k,
        "coarse_iou": sum(row["coarse_iou"] for row in rows) / max(len(rows), 1),
        "refined_iou": sum(row["refined_iou"] for row in rows) / max(len(rows), 1),
        "patch_iou": sum(row["patch_iou"] for row in rows) / max(len(rows), 1),
        "rows_csv": str(csv_path),
    }
    (args.output_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
