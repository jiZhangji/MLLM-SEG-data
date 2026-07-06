from __future__ import annotations

import torch


def patch_ids_to_boxes(
    selected_ids: torch.Tensor,
    grid_hw: tuple[int, int],
    image_hw: tuple[int, int],
    scale: float = 1.0,
) -> torch.Tensor:
    """Convert flattened token ids to image-space boxes for roi_align."""

    if selected_ids.ndim != 2:
        raise ValueError(f"selected_ids must be [B, K], got {tuple(selected_ids.shape)}")
    grid_h, grid_w = grid_hw
    image_h, image_w = image_hw
    rows = torch.div(selected_ids, grid_w, rounding_mode="floor").float()
    cols = (selected_ids % grid_w).float()

    cell_h = float(image_h) / float(grid_h)
    cell_w = float(image_w) / float(grid_w)
    cy = (rows + 0.5) * cell_h
    cx = (cols + 0.5) * cell_w
    box_h = cell_h * scale
    box_w = cell_w * scale

    x1 = (cx - box_w / 2.0).clamp(0.0, float(image_w))
    y1 = (cy - box_h / 2.0).clamp(0.0, float(image_h))
    x2 = (cx + box_w / 2.0).clamp(0.0, float(image_w))
    y2 = (cy + box_h / 2.0).clamp(0.0, float(image_h))
    return torch.stack([x1, y1, x2, y2], dim=-1)

