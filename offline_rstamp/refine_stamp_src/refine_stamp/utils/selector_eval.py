from __future__ import annotations

import torch
import torch.nn.functional as F

from refine_stamp.utils.metrics import boundary_iou, boundary_map, mask_iou


def resize_binary_to_grid(mask: torch.Tensor, grid_hw: tuple[int, int]) -> torch.Tensor:
    """Downsample a binary mask to patch grid occupancy."""
    if mask.ndim == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.ndim == 3:
        mask = mask.unsqueeze(0)
    if mask.ndim != 4:
        raise ValueError(f"mask must be [H, W] or [B, 1, H, W], got {tuple(mask.shape)}")
    pooled = F.interpolate(mask.float(), size=grid_hw, mode="area")
    return (pooled > 0.0).float()


def logits_to_coarse_mask(mask_logits: torch.Tensor, grid_hw: tuple[int, int]) -> torch.Tensor:
    if mask_logits.ndim != 3 or mask_logits.shape[-1] != 2:
        raise ValueError(f"mask_logits must be [B, N, 2], got {tuple(mask_logits.shape)}")
    batch_size, num_tokens, _ = mask_logits.shape
    grid_h, grid_w = grid_hw
    if grid_h * grid_w != num_tokens:
        raise ValueError(f"grid_hw={grid_hw} does not match num_tokens={num_tokens}")
    fg_prob = torch.softmax(mask_logits, dim=-1)[..., 1]
    return (fg_prob.reshape(batch_size, 1, grid_h, grid_w) >= 0.5).float()


def selected_patch_rate(selected_ids: torch.Tensor, grid_map: torch.Tensor) -> torch.Tensor:
    """Return per-sample selected-patch hit rate for a grid map."""
    if selected_ids.ndim != 2:
        raise ValueError(f"selected_ids must be [B, K], got {tuple(selected_ids.shape)}")
    if grid_map.ndim != 4:
        raise ValueError(f"grid_map must be [B, 1, H, W], got {tuple(grid_map.shape)}")
    batch_size, _, grid_h, grid_w = grid_map.shape
    if selected_ids.shape[0] != batch_size:
        raise ValueError("Batch mismatch between selected_ids and grid_map.")
    flat = grid_map.reshape(batch_size, grid_h * grid_w)
    hits = torch.gather(flat, dim=1, index=selected_ids)
    return hits.float().mean(dim=1)


def selector_quality_metrics(
    *,
    selected_ids: torch.Tensor,
    score_map: torch.Tensor,
    coarse_grid_mask: torch.Tensor,
    gt_grid_mask: torch.Tensor,
) -> dict[str, float]:
    gt_boundary = boundary_map(gt_grid_mask)
    coarse_error = (coarse_grid_mask != gt_grid_mask).float()

    selected_gt_boundary_rate = selected_patch_rate(selected_ids, gt_boundary)
    selected_error_rate = selected_patch_rate(selected_ids, coarse_error)
    selected_fg_rate = selected_patch_rate(selected_ids, coarse_grid_mask)
    selected_gt_fg_rate = selected_patch_rate(selected_ids, gt_grid_mask)

    flat_score = score_map.reshape(score_map.shape[0], -1)
    selected_score = torch.gather(flat_score, dim=1, index=selected_ids).mean(dim=1)

    return {
        "selected_gt_boundary_rate": float(selected_gt_boundary_rate.mean().item()),
        "selected_error_rate": float(selected_error_rate.mean().item()),
        "selected_fg_rate": float(selected_fg_rate.mean().item()),
        "selected_gt_fg_rate": float(selected_gt_fg_rate.mean().item()),
        "score_mean_on_selected": float(selected_score.mean().item()),
    }


def coarse_mask_metrics(
    *,
    coarse_grid_mask: torch.Tensor,
    gt_mask: torch.Tensor,
    grid_hw: tuple[int, int],
) -> dict[str, float]:
    if gt_mask.ndim == 2:
        gt_mask_4d = gt_mask.unsqueeze(0).unsqueeze(0).float()
    elif gt_mask.ndim == 4:
        gt_mask_4d = gt_mask.float()
    else:
        raise ValueError(f"gt_mask must be [H, W] or [B, 1, H, W], got {tuple(gt_mask.shape)}")

    coarse_full = F.interpolate(
        coarse_grid_mask.float(),
        size=gt_mask_4d.shape[-2:],
        mode="nearest",
    )
    gt_grid = resize_binary_to_grid(gt_mask_4d, grid_hw)

    return {
        "coarse_iou": float(mask_iou(coarse_full, gt_mask_4d).mean().item()),
        "coarse_boundary_iou": float(boundary_iou(coarse_full, gt_mask_4d).mean().item()),
        "coarse_fg_ratio": float(coarse_grid_mask.mean().item()),
        "gt_fg_ratio": float(gt_grid.mean().item()),
    }
