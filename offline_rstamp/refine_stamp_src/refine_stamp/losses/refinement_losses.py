from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    from torchvision.ops import roi_align
except Exception:  # pragma: no cover
    roi_align = None


def _require_roi_align() -> None:
    if roi_align is None:
        raise ImportError("torchvision.ops.roi_align is required for mask cropping.")


def crop_gt_masks(
    gt_masks: torch.Tensor,
    boxes: torch.Tensor,
    output_size: int,
) -> torch.Tensor:
    _require_roi_align()
    batch_size, top_k, _ = boxes.shape
    box_list = [boxes[index] for index in range(batch_size)]
    targets = roi_align(
        input=gt_masks.float(),
        boxes=box_list,
        output_size=(output_size, output_size),
        spatial_scale=1.0,
        aligned=True,
    )
    targets = targets.reshape(batch_size, top_k, 1, output_size, output_size)
    return (targets >= 0.5).float()


def dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    probabilities = torch.sigmoid(logits).flatten(start_dim=2)
    targets = targets.flatten(start_dim=2)
    intersection = (probabilities * targets).sum(dim=-1)
    denominator = probabilities.sum(dim=-1) + targets.sum(dim=-1)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


def refinement_loss(
    local_logits: torch.Tensor,
    local_targets: torch.Tensor,
) -> dict[str, torch.Tensor]:
    bce = F.binary_cross_entropy_with_logits(local_logits, local_targets)
    dice = dice_loss(local_logits, local_targets)
    total = bce + dice
    return {"loss": total, "loss_bce": bce, "loss_dice": dice}
