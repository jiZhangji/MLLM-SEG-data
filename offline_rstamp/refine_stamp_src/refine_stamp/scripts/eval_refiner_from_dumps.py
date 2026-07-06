from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from refine_stamp.data.dump_dataset import (
    RefinementDumpDataset,
    collate_refinement_dumps,
    find_dump_paths,
    split_paths,
)
from refine_stamp.models.local_refiner import LocalPatchRefiner
from refine_stamp.models.patch_selector import PatchSelector
from refine_stamp.utils.metrics import boundary_iou, mask_iou
from refine_stamp.utils.stitch import stitch_masks
from refine_stamp.utils.visualization import make_panel, overlay_mask, save_image


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[LocalPatchRefiner, dict]:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    config = ckpt["config"]
    model = LocalPatchRefiner(
        token_dim=int(ckpt["token_dim"]),
        token_channels=int(config.get("token_channels", 32)),
        crop_size=int(config.get("crop_size", 64)),
        output_size=int(config.get("output_size", 16)),
        box_scale=float(config.get("box_scale", 1.0)),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, ckpt


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {}
    total_inter_coarse = sum(int(r["coarse_intersection"]) for r in rows)
    total_union_coarse = sum(int(r["coarse_union"]) for r in rows)
    total_inter_refined = sum(int(r["refined_intersection"]) for r in rows)
    total_union_refined = sum(int(r["refined_union"]) for r in rows)
    out = {
        "samples": len(rows),
        "coarse_mean_iou": sum(float(r["coarse_iou"]) for r in rows) / len(rows),
        "refined_mean_iou": sum(float(r["refined_iou"]) for r in rows) / len(rows),
        "coarse_boundary_iou": sum(float(r["coarse_boundary_iou"]) for r in rows) / len(rows),
        "refined_boundary_iou": sum(float(r["refined_boundary_iou"]) for r in rows) / len(rows),
        "coarse_cIoU": total_inter_coarse / total_union_coarse if total_union_coarse else 0.0,
        "refined_cIoU": total_inter_refined / total_union_refined if total_union_refined else 0.0,
    }
    out["mean_iou_delta"] = out["refined_mean_iou"] - out["coarse_mean_iou"]
    out["boundary_iou_delta"] = out["refined_boundary_iou"] - out["coarse_boundary_iou"]
    out["cIoU_delta"] = out["refined_cIoU"] - out["coarse_cIoU"]
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, summary: dict, settings: dict) -> None:
    def fmt(value):
        return f"{value:.6f}" if isinstance(value, float) else str(value)

    lines = [
        "# Refine-STAMP Refiner Evaluation",
        "",
        "## Settings",
        "",
        f"- selector: `{settings['selector']}`",
        f"- top_k: `{settings['top_k']}`",
        f"- checkpoint: `{settings['checkpoint']}`",
        "",
        "| Metric | Coarse STAMP | Refined | Delta |",
        "|---|---:|---:|---:|",
        f"| mean IoU | {fmt(summary['coarse_mean_iou'])} | {fmt(summary['refined_mean_iou'])} | {fmt(summary['mean_iou_delta'])} |",
        f"| cIoU | {fmt(summary['coarse_cIoU'])} | {fmt(summary['refined_cIoU'])} | {fmt(summary['cIoU_delta'])} |",
        f"| Boundary IoU | {fmt(summary['coarse_boundary_iou'])} | {fmt(summary['refined_boundary_iou'])} | {fmt(summary['boundary_iou_delta'])} |",
        "",
        "Interpretation:",
        "",
        "- This compares STAMP's coarse mask against the local-refiner stitched mask on the same dump samples.",
        "- Early MVP success is mainly a positive `Boundary IoU` delta without a large negative cIoU delta.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate(args: argparse.Namespace) -> dict:
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model, ckpt = load_model(args.checkpoint, device)
    selector_name = args.selector or ckpt.get("selector") or ckpt["config"].get("selector", "hybrid")
    top_k = args.top_k or int(ckpt["config"].get("top_k", 64))

    if args.use_checkpoint_val and ckpt.get("val_paths"):
        paths = [Path(p) for p in ckpt["val_paths"]]
    else:
        all_paths = find_dump_paths(args.input_dir, args.limit)
        _train_paths, paths = split_paths(all_paths, val_fraction=args.val_fraction, seed=args.seed)

    dataset = RefinementDumpDataset(paths, image_size=args.image_size)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_refinement_dumps)
    selector = PatchSelector(top_k=top_k, boundary_weight=args.boundary_weight, mode=selector_name)

    rows = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = args.output_dir / "visualizations"

    with torch.no_grad():
        for index, batch in enumerate(tqdm(loader, desc="eval_refiner")):
            images = batch["images"].to(device)
            masks = batch["masks"].to(device)
            mask_logits = batch["mask_logits"].to(device)
            mask_hidden = batch["mask_hidden"].to(device)
            grid_hw = batch["grid_hw"]
            image_h, image_w = images.shape[-2:]

            selected_ids = selector(mask_logits=mask_logits, grid_hw=grid_hw)["selected_ids"].to(device)
            outputs = model(images=images, mask_hidden=mask_hidden, selected_ids=selected_ids, grid_hw=grid_hw)
            refined_prob, refined_mask = stitch_masks(
                coarse_logits=mask_logits,
                local_logits=outputs["local_logits"],
                selected_ids=selected_ids,
                grid_hw=grid_hw,
                image_hw=(image_h, image_w),
                blend_weight=args.blend_weight,
            )

            coarse_prob = torch.softmax(mask_logits, dim=-1)[..., 1].reshape(1, 1, *grid_hw)
            coarse_mask = F.interpolate(coarse_prob, size=(image_h, image_w), mode="nearest") >= 0.5
            gt = masks >= 0.5

            coarse_iou = mask_iou(coarse_mask, gt)[0]
            refined_iou = mask_iou(refined_mask, gt)[0]
            coarse_biou = boundary_iou(coarse_mask.float(), gt.float())[0]
            refined_biou = boundary_iou(refined_mask.float(), gt.float())[0]
            coarse_inter = int((coarse_mask.bool() & gt.bool()).sum().item())
            coarse_union = int((coarse_mask.bool() | gt.bool()).sum().item())
            refined_inter = int((refined_mask.bool() & gt.bool()).sum().item())
            refined_union = int((refined_mask.bool() | gt.bool()).sum().item())
            row = {
                "name": batch["names"][0],
                "path": batch["paths"][0],
                "selector": selector_name,
                "coarse_iou": float(coarse_iou.item()),
                "refined_iou": float(refined_iou.item()),
                "iou_delta": float((refined_iou - coarse_iou).item()),
                "coarse_boundary_iou": float(coarse_biou.item()),
                "refined_boundary_iou": float(refined_biou.item()),
                "boundary_iou_delta": float((refined_biou - coarse_biou).item()),
                "coarse_intersection": coarse_inter,
                "coarse_union": coarse_union,
                "refined_intersection": refined_inter,
                "refined_union": refined_union,
            }
            rows.append(row)

            if index < args.visualize_limit:
                image_path = torch.load(batch["paths"][0], map_location="cpu")["image_path"]
                image_pil = Image.open(image_path).convert("RGB")
                image_pil = image_pil.resize((args.image_size, args.image_size))
                panel = make_panel(
                    [
                        ("image", image_pil),
                        ("GT", overlay_mask(image_pil, gt[0].cpu(), color=(40, 220, 80))),
                        ("coarse", overlay_mask(image_pil, coarse_mask[0].cpu(), color=(255, 40, 40))),
                        ("refined", overlay_mask(image_pil, refined_mask[0].cpu(), color=(40, 120, 255))),
                    ],
                    columns=2,
                )
                save_image(vis_dir / f"{batch['names'][0]}_refiner_panel.png", panel)

    summary = summarize(rows)
    settings = {
        "selector": selector_name,
        "top_k": top_k,
        "checkpoint": str(args.checkpoint),
    }
    write_csv(args.output_dir / "refiner_eval_rows.csv", rows)
    write_markdown(args.output_dir / "refiner_eval_summary.md", summary, settings)
    report = {"settings": settings, "summary": summary, "rows_csv": str(args.output_dir / "refiner_eval_rows.csv")}
    (args.output_dir / "refiner_eval_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--selector", choices=["random", "uncertainty", "boundary", "hybrid"], default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-checkpoint-val", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-size", type=int, default=896)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--boundary-weight", type=float, default=0.5)
    parser.add_argument("--blend-weight", type=float, default=0.8)
    parser.add_argument("--visualize-limit", type=int, default=8)
    args = parser.parse_args()
    report = evaluate(args)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
