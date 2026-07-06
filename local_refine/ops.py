from __future__ import annotations

import torch
import torch.nn.functional as F

from local_refine.model import crop_and_resize


def crop_mask_targets(gt_mask: torch.Tensor, boxes: torch.Tensor, output_size: int) -> torch.Tensor:
    targets = crop_and_resize(gt_mask, boxes, output_size)
    return targets.reshape(gt_mask.shape[0], boxes.shape[1], 1, output_size, output_size)


def stitch_logits(
    coarse_logits: torch.Tensor,
    local_logits: torch.Tensor,
    boxes: torch.Tensor,
    image_hw: tuple[int, int],
    blend_weight: float = 0.8,
) -> torch.Tensor:
    batch_size, top_k, _, _, _ = local_logits.shape
    image_h, image_w = image_hw
    fg_logits = coarse_logits[..., 1].reshape(batch_size, 1, *infer_grid(coarse_logits))
    canvas = F.interpolate(fg_logits, size=image_hw, mode="bilinear", align_corners=False)
    weight = torch.ones_like(canvas)

    for batch_index in range(batch_size):
        for patch_index in range(top_k):
            x1, y1, x2, y2 = boxes[batch_index, patch_index].round().long().tolist()
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(x2, image_w), min(y2, image_h)
            if x2 <= x1 or y2 <= y1:
                continue
            patch = F.interpolate(
                local_logits[batch_index, patch_index].unsqueeze(0),
                size=(y2 - y1, x2 - x1),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            canvas[batch_index, :, y1:y2, x1:x2] = (
                (1.0 - blend_weight) * canvas[batch_index, :, y1:y2, x1:x2] + blend_weight * patch
            )
            weight[batch_index, :, y1:y2, x1:x2] = 1.0
    return canvas / weight


def infer_grid(mask_logits: torch.Tensor) -> tuple[int, int]:
    tokens = mask_logits.shape[1]
    side = int(tokens ** 0.5)
    if side * side == tokens:
        return side, side
    raise ValueError("Cannot infer non-square grid; pass grid-specific stitching if needed.")


def binary_iou(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_bin = pred.bool()
    target_bin = target.bool()
    inter = (pred_bin & target_bin).float().sum(dim=(-2, -1))
    union = (pred_bin | target_bin).float().sum(dim=(-2, -1)).clamp_min(1.0)
    return inter / union
