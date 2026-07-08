from __future__ import annotations

import torch
import torch.nn as nn


class MaskTokenRefinementAdapter(nn.Module):
    """Predict a residual correction for STAMP mask-token logits."""

    def __init__(
        self,
        token_dim: int,
        hidden_size: int = 128,
        use_uncertainty_gate: bool = True,
    ) -> None:
        super().__init__()
        self.use_uncertainty_gate = use_uncertainty_gate
        self.mlp = nn.Sequential(
            nn.Linear(token_dim + 4, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 2),
        )

    def forward(self, mask_hidden: torch.Tensor, mask_logits: torch.Tensor) -> dict[str, torch.Tensor]:
        fg_prob = torch.softmax(mask_logits, dim=-1)[..., 1:2]
        uncertainty = 1.0 - torch.abs(2.0 * fg_prob - 1.0)
        features = torch.cat([mask_hidden, mask_logits, fg_prob, uncertainty], dim=-1)
        delta_logits = self.mlp(features)
        gate = uncertainty if self.use_uncertainty_gate else torch.ones_like(uncertainty)
        refined_logits = mask_logits + gate * delta_logits
        return {
            "refined_logits": refined_logits,
            "delta_logits": delta_logits,
            "uncertainty": uncertainty,
            "fg_prob": fg_prob,
        }


class UncertaintyAwareMaskHead(nn.Module):
    """Predict mask-token logits directly from frozen STAMP mask hidden states."""

    def __init__(
        self,
        token_dim: int,
        hidden_size: int = 128,
        use_uncertainty_feature: bool = True,
    ) -> None:
        super().__init__()
        self.use_uncertainty_feature = use_uncertainty_feature
        self.uncertainty_head = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )
        mask_input_dim = token_dim + (1 if use_uncertainty_feature else 0)
        self.mask_head = nn.Sequential(
            nn.LayerNorm(mask_input_dim),
            nn.Linear(mask_input_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 2),
        )

    def forward(self, mask_hidden: torch.Tensor) -> dict[str, torch.Tensor]:
        uncertainty_logits = self.uncertainty_head(mask_hidden)
        uncertainty = torch.sigmoid(uncertainty_logits)
        if self.use_uncertainty_feature:
            features = torch.cat([mask_hidden, uncertainty], dim=-1)
        else:
            features = mask_hidden
        mask_logits = self.mask_head(features)
        return {
            "mask_logits": mask_logits,
            "uncertainty": uncertainty,
            "uncertainty_logits": uncertainty_logits,
        }
