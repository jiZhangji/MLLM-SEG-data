from __future__ import annotations

import torch


def patch_ids_to_boxes(
    selected_ids: torch.Tensor,
    grid_hw: tuple[int, int],
    image_hw: tuple[int, int],
    scale: float = 1.0,
) -> torch.Tensor:
    """Convert flattened patch ids to xyxy image-space boxes."""
    if selected_ids.ndim != 2:
        raise ValueError(f"selected_ids must have shape [B, K], got {tuple(selected_ids.shape)}")
    if scale <= 0:
        raise ValueError("scale must be positive.")

    grid_h, grid_w = grid_hw
    image_h, image_w = image_hw
    if grid_h <= 0 or grid_w <= 0:
        raise ValueError(f"Invalid grid_hw: {grid_hw}")
    if image_h <= 0 or image_w <= 0:
        raise ValueError(f"Invalid image_hw: {image_hw}")

    rows = selected_ids // grid_w
    cols = selected_ids % grid_w

    patch_h = image_h / grid_h
    patch_w = image_w / grid_w

    x1 = cols.float() * patch_w
    y1 = rows.float() * patch_h
    x2 = (cols.float() + 1.0) * patch_w
    y2 = (rows.float() + 1.0) * patch_h

    if scale != 1.0:
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        half_w = (x2 - x1) * 0.5 * scale
        half_h = (y2 - y1) * 0.5 * scale
        x1 = (cx - half_w).clamp(0, image_w)
        x2 = (cx + half_w).clamp(0, image_w)
        y1 = (cy - half_h).clamp(0, image_h)
        y2 = (cy + half_h).clamp(0, image_h)

    return torch.stack([x1, y1, x2, y2], dim=-1)
