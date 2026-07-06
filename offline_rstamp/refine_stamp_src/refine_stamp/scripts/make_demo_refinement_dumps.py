from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw


def make_one(index: int, output_dir: Path, grid_hw: tuple[int, int] = (16, 16)) -> Path:
    grid_h, grid_w = grid_hw
    image_size = 384
    y = torch.linspace(-1.0, 1.0, grid_h).view(grid_h, 1)
    x = torch.linspace(-1.0, 1.0, grid_w).view(1, grid_w)
    cx = -0.2 + 0.08 * index
    cy = 0.05
    radius = 0.42 + 0.02 * (index % 3)
    dist = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    gt_grid = (dist < radius).float()
    fg_prob = torch.sigmoid((radius - dist) * 10.0)

    # Add a deterministic local error patch so selector/error metrics are meaningful.
    if index % 2 == 0:
        fg_prob[grid_h // 2, grid_w // 2 + 3] = 0.8
    else:
        fg_prob[grid_h // 2, grid_w // 2] = 0.2

    logits = torch.stack([1.0 - fg_prob, fg_prob], dim=-1).clamp_min(1e-4).log()
    logits = logits.view(1, grid_h * grid_w, 2)
    hidden = torch.randn(1, grid_h * grid_w, 32)

    gt_mask = F.interpolate(
        gt_grid.view(1, 1, grid_h, grid_w),
        size=(image_size, image_size),
        mode="nearest",
    )[0, 0]

    image = Image.new("RGB", (image_size, image_size), (235, 238, 242))
    draw = ImageDraw.Draw(image)
    draw.ellipse(
        [
            int((cx + 1.0 - radius) * 0.5 * image_size),
            int((cy + 1.0 - radius) * 0.5 * image_size),
            int((cx + 1.0 + radius) * 0.5 * image_size),
            int((cy + 1.0 + radius) * 0.5 * image_size),
        ],
        fill=(160, 190, 230),
        outline=(40, 80, 140),
        width=3,
    )

    image_path = output_dir / f"demo_{index:03d}.png"
    mask_path = output_dir / f"demo_{index:03d}_mask.png"
    dump_path = output_dir / f"demo_{index:03d}.pt"
    image.save(image_path)
    Image.fromarray((gt_mask.numpy() * 255).astype("uint8")).save(mask_path)
    torch.save(
        {
            "name": f"demo_{index:03d}",
            "mask_logits": logits,
            "mask_hidden": hidden,
            "grid_hw": grid_hw,
            "image_path": str(image_path),
            "mask_path": str(mask_path),
            "query": "demo object",
        },
        dump_path,
    )
    return dump_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--count", type=int, default=8)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = [make_one(i, args.output_dir) for i in range(args.count)]
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
