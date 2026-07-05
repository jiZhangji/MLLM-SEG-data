from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchSelector(nn.Module):
    """Select difficult patches from STAMP foreground logits."""

    def __init__(
        self,
        top_k: int = 64,
        boundary_weight: float = 0.5,
        mode: str = "hybrid",
    ) -> None:
        super().__init__()
        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        if mode not in {"hybrid", "uncertainty", "boundary", "random"}:
            raise ValueError(f"Unsupported selector mode: {mode}")

        self.top_k = top_k
        self.boundary_weight = boundary_weight
        self.mode = mode

    @torch.no_grad()
    def forward(
        self,
        mask_logits: torch.Tensor,
        grid_hw: tuple[int, int],
    ) -> dict[str, torch.Tensor]:
        if mask_logits.ndim != 3 or mask_logits.shape[-1] != 2:
            raise ValueError(
                f"mask_logits must have shape [B, N, 2], got {tuple(mask_logits.shape)}"
            )

        batch_size, num_tokens, _ = mask_logits.shape
        grid_h, grid_w = grid_hw

        if grid_h * grid_w != num_tokens:
            raise ValueError(f"grid_hw={grid_hw} does not match num_tokens={num_tokens}")

        fg_prob = torch.softmax(mask_logits, dim=-1)[..., 1]
        fg_prob = fg_prob.reshape(batch_size, 1, grid_h, grid_w)

        uncertainty = 1.0 - torch.abs(2.0 * fg_prob - 1.0)
        coarse_binary = (fg_prob >= 0.5).float()

        local_max = F.max_pool2d(coarse_binary, kernel_size=3, stride=1, padding=1)
        local_min = -F.max_pool2d(-coarse_binary, kernel_size=3, stride=1, padding=1)
        boundary = (local_max - local_min).clamp(0.0, 1.0)

        if self.mode == "uncertainty":
            score = uncertainty
        elif self.mode == "boundary":
            score = boundary
        elif self.mode == "random":
            score = torch.rand_like(uncertainty)
        else:
            score = uncertainty + self.boundary_weight * boundary

        flat_score = score.flatten(start_dim=1)
        k = min(self.top_k, num_tokens)
        selected_ids = torch.topk(flat_score, k=k, dim=1, largest=True, sorted=True).indices

        return {
            "selected_ids": selected_ids,
            "uncertainty_map": uncertainty,
            "boundary_map": boundary,
            "score_map": score,
            "fg_prob_map": fg_prob,
        }
