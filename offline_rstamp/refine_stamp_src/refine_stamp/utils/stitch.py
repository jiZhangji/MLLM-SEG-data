from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def stitch_masks(
    coarse_logits: torch.Tensor,
    local_logits: torch.Tensor,
    selected_ids: torch.Tensor,
    grid_hw: tuple[int, int],
    image_hw: tuple[int, int],
    blend_weight: float = 0.8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Blend local predictions into an upsampled STAMP coarse mask."""
    if not 0.0 <= blend_weight <= 1.0:
        raise ValueError("blend_weight must be in [0, 1].")
    if coarse_logits.ndim != 3 or coarse_logits.shape[-1] != 2:
        raise ValueError(f"coarse_logits must be [B, N, 2], got {tuple(coarse_logits.shape)}")

    batch_size, num_tokens, _ = coarse_logits.shape
    grid_h, grid_w = grid_hw
    image_h, image_w = image_hw
    if grid_h * grid_w != num_tokens:
        raise ValueError(f"grid_hw={grid_hw} does not match num_tokens={num_tokens}")

    coarse_prob = torch.softmax(coarse_logits, dim=-1)[..., 1]
    coarse_prob = coarse_prob.reshape(batch_size, 1, grid_h, grid_w)
    final_prob = F.interpolate(
        coarse_prob,
        size=(image_h, image_w),
        mode="bilinear",
        align_corners=False,
    )

    local_prob = torch.sigmoid(local_logits)
    patch_h = image_h / grid_h
    patch_w = image_w / grid_w
    top_k = selected_ids.shape[1]

    for batch_index in range(batch_size):
        for patch_index in range(top_k):
            patch_id = int(selected_ids[batch_index, patch_index].item())
            row = patch_id // grid_w
            col = patch_id % grid_w
            y1 = round(row * patch_h)
            y2 = round((row + 1) * patch_h)
            x1 = round(col * patch_w)
            x2 = round((col + 1) * patch_w)
            if y2 <= y1 or x2 <= x1:
                continue

            refined_patch = F.interpolate(
                local_prob[batch_index, patch_index].unsqueeze(0),
                size=(y2 - y1, x2 - x1),
                mode="bilinear",
                align_corners=False,
            )[0]
            old_patch = final_prob[batch_index, :, y1:y2, x1:x2]
            final_prob[batch_index, :, y1:y2, x1:x2] = (
                blend_weight * refined_patch + (1.0 - blend_weight) * old_patch
            )

    return final_prob, final_prob >= 0.5
