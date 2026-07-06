from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image

from refine_stamp.models.patch_selector import PatchSelector
from refine_stamp.utils.visualization import (
    draw_patch_boxes,
    heatmap_image,
    make_panel,
    mask_image,
    overlay_mask,
    save_image,
)


def make_demo_payload() -> dict[str, Any]:
    grid_h, grid_w = 16, 16
    y = torch.linspace(-1.0, 1.0, grid_h).view(grid_h, 1)
    x = torch.linspace(-1.0, 1.0, grid_w).view(1, grid_w)
    dist = torch.sqrt((x + 0.15) ** 2 + (y - 0.05) ** 2)
    fg_prob = torch.sigmoid((0.48 - dist) * 12.0)
    logits = torch.stack([1.0 - fg_prob, fg_prob], dim=-1).log().view(1, grid_h * grid_w, 2)
    hidden = torch.randn(1, grid_h * grid_w, 32)

    image = Image.new("RGB", (384, 384), (235, 238, 242))
    gt = (dist < 0.45).float().view(1, 1, grid_h, grid_w)
    gt = F.interpolate(gt, size=(384, 384), mode="nearest")[0, 0]

    return {
        "mask_logits": logits,
        "mask_hidden": hidden,
        "grid_hw": (grid_h, grid_w),
        "image": image,
        "gt_mask": gt,
        "name": "demo",
    }


def load_payload(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    if "grid_hw" in payload:
        payload["grid_hw"] = tuple(int(x) for x in payload["grid_hw"])
    if "image" not in payload:
        image_path = payload.get("image_path")
        if image_path:
            payload["image"] = Image.open(image_path).convert("RGB")
    elif isinstance(payload["image"], (str, Path)):
        payload["image"] = Image.open(payload["image"]).convert("RGB")

    if "gt_mask" not in payload:
        mask_path = payload.get("mask_path") or payload.get("gt_mask_path")
        if mask_path:
            payload["gt_mask"] = torch.from_numpy(
                __import__("numpy").array(Image.open(mask_path).convert("L")) > 0
            ).float()

    payload.setdefault("name", path.stem)
    return payload


def validate_payload(payload: dict[str, Any]) -> None:
    required = {"mask_logits", "grid_hw", "image"}
    missing = required - set(payload)
    if missing:
        raise KeyError(f"Missing payload keys: {sorted(missing)}")
    mask_logits = payload["mask_logits"]
    grid_hw = payload["grid_hw"]
    if mask_logits.ndim != 3 or mask_logits.shape[-1] != 2:
        raise ValueError(f"mask_logits must be [B, N, 2], got {tuple(mask_logits.shape)}")
    if grid_hw[0] * grid_hw[1] != mask_logits.shape[1]:
        raise ValueError(f"grid_hw={grid_hw} does not match N={mask_logits.shape[1]}")


def visualize_one(
    payload: dict[str, Any],
    output_dir: Path,
    top_k: int,
    boundary_weight: float,
    selector_mode: str,
) -> dict[str, Any]:
    validate_payload(payload)
    image: Image.Image = payload["image"].convert("RGB")
    mask_logits: torch.Tensor = payload["mask_logits"].float()
    grid_hw = payload["grid_hw"]
    name = str(payload.get("name") or "sample")

    selector = PatchSelector(top_k=top_k, boundary_weight=boundary_weight, mode=selector_mode)
    selection = selector(mask_logits=mask_logits, grid_hw=grid_hw)

    fg_prob = selection["fg_prob_map"][0]
    coarse_mask = (fg_prob >= 0.5).float()
    uncertainty = selection["uncertainty_map"][0]
    boundary = selection["boundary_map"][0]
    score = selection["score_map"][0]
    selected_ids = selection["selected_ids"][0]

    selected_overlay = draw_patch_boxes(image, selected_ids, grid_hw)
    coarse_overlay = overlay_mask(image, coarse_mask)

    panels = [
        ("image", image),
        ("coarse mask overlay", coarse_overlay),
        ("selected patches", selected_overlay),
        ("foreground probability", heatmap_image(fg_prob, image.size)),
        ("uncertainty", heatmap_image(uncertainty, image.size)),
        ("boundary", heatmap_image(boundary, image.size)),
        ("selector score", heatmap_image(score, image.size)),
    ]
    if "gt_mask" in payload:
        panels.append(("GT mask", mask_image(payload["gt_mask"], image.size)))

    panel = make_panel(panels, columns=3)
    panel_path = output_dir / f"{name}_selector_panel.png"
    save_image(panel_path, panel)

    selected_path = output_dir / f"{name}_selected_patches.png"
    save_image(selected_path, selected_overlay)

    return {
        "name": name,
        "panel": str(panel_path),
        "selected_patches": str(selected_path),
        "grid_hw": list(grid_hw),
        "num_tokens": int(mask_logits.shape[1]),
        "top_k": int(selected_ids.numel()),
        "foreground_ratio": float(coarse_mask.mean().item()),
        "mean_uncertainty": float(uncertainty.mean().item()),
        "mean_boundary": float(boundary.mean().item()),
        "selected_ids": [int(x) for x in selected_ids.tolist()],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=None, help="A .pt payload from STAMP refinement export.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--boundary-weight", type=float, default=0.5)
    parser.add_argument("--selector-mode", default="hybrid", choices=["hybrid", "uncertainty", "boundary", "random"])
    parser.add_argument("--demo", action="store_true", help="Run a synthetic demo without STAMP outputs.")
    args = parser.parse_args()

    if args.demo:
        payload = make_demo_payload()
    elif args.input:
        payload = load_payload(args.input)
    else:
        raise SystemExit("Pass --demo or --input PATH.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = visualize_one(
        payload=payload,
        output_dir=args.output_dir,
        top_k=args.top_k,
        boundary_weight=args.boundary_weight,
        selector_mode=args.selector_mode,
    )
    report_path = args.output_dir / "selector_visualization_summary.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
