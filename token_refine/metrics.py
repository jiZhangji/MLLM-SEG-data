from __future__ import annotations

import torch
import torch.nn.functional as F


def logits_to_mask(logits: torch.Tensor, grid_hw: tuple[int, int], image_size: int) -> torch.Tensor:
    batch_size = logits.shape[0]
    fg = torch.softmax(logits, dim=-1)[..., 1].reshape(batch_size, 1, *grid_hw)
    return F.interpolate(fg, size=(image_size, image_size), mode="bilinear", align_corners=False)


def binary_iou(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_bin = pred.bool()
    target_bin = target.bool()
    inter = (pred_bin & target_bin).float().sum(dim=(-2, -1))
    union = (pred_bin | target_bin).float().sum(dim=(-2, -1)).clamp_min(1.0)
    return inter / union


def binary_intersection_union(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    pred_bin = pred.bool()
    target_bin = target.bool()
    inter = (pred_bin & target_bin).sum(dim=(-2, -1))
    union = (pred_bin | target_bin).sum(dim=(-2, -1))
    return inter, union
