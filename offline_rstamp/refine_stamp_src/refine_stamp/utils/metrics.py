from __future__ import annotations

import torch
import torch.nn.functional as F


def mask_iou(pred_mask: torch.Tensor, target_mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred = pred_mask.bool()
    target = target_mask.bool()
    intersection = (pred & target).flatten(start_dim=1).sum(dim=1).float()
    union = (pred | target).flatten(start_dim=1).sum(dim=1).float()
    return (intersection + eps) / (union + eps)


def boundary_map(mask: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    if mask.ndim != 4:
        raise ValueError(f"mask must be [B, 1, H, W], got {tuple(mask.shape)}")
    mask = mask.float()
    local_max = F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    local_min = -F.max_pool2d(-mask, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    return (local_max - local_min).clamp(0.0, 1.0)


def boundary_iou(pred_mask: torch.Tensor, target_mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return mask_iou(boundary_map(pred_mask), boundary_map(target_mask), eps=eps)
