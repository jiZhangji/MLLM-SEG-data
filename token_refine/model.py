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
        trainable_logit_calibration: bool = False,
    ) -> None:
        super().__init__()
        self.use_uncertainty_gate = use_uncertainty_gate
        self.trainable_logit_calibration = trainable_logit_calibration
        if trainable_logit_calibration:
            self.logit_scale = nn.Parameter(torch.zeros(2))
            self.logit_bias = nn.Parameter(torch.zeros(2))
        else:
            self.register_buffer("logit_scale", torch.zeros(2), persistent=False)
            self.register_buffer("logit_bias", torch.zeros(2), persistent=False)
        self.mlp = nn.Sequential(
            nn.Linear(token_dim + 4, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 2),
        )

    def forward(self, mask_hidden: torch.Tensor, mask_logits: torch.Tensor) -> dict[str, torch.Tensor]:
        base_logits = mask_logits * torch.exp(self.logit_scale).view(1, 1, 2) + self.logit_bias.view(1, 1, 2)
        fg_prob = torch.softmax(base_logits, dim=-1)[..., 1:2]
        uncertainty = 1.0 - torch.abs(2.0 * fg_prob - 1.0)
        features = torch.cat([mask_hidden, base_logits, fg_prob, uncertainty], dim=-1)
        delta_logits = self.mlp(features)
        gate = uncertainty if self.use_uncertainty_gate else torch.ones_like(uncertainty)
        refined_logits = base_logits + gate * delta_logits
        return {
            "base_logits": base_logits,
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


class SpatialTokenBlock(nn.Module):
    """Lightweight local interaction between neighboring mask-token embeddings."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.block = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features: [B, H, W, C]. Aggregate valid 4-neighborhoods without
        # Conv2d/cuDNN so this remains fast on CUDA stacks with broken cuDNN.
        context = features.clone()
        counts = torch.ones((*features.shape[1:3], 1), device=features.device, dtype=features.dtype)
        context[:, 1:] = context[:, 1:] + features[:, :-1]
        counts[1:] = counts[1:] + 1
        context[:, :-1] = context[:, :-1] + features[:, 1:]
        counts[:-1] = counts[:-1] + 1
        context[:, :, 1:] = context[:, :, 1:] + features[:, :, :-1]
        counts[:, 1:] = counts[:, 1:] + 1
        context[:, :, :-1] = context[:, :, :-1] + features[:, :, 1:]
        counts[:, :-1] = counts[:, :-1] + 1
        context = context / counts.unsqueeze(0)
        return features + self.block(self.norm(context))


class IndependentUncertaintyMaskHead(nn.Module):
    """A new mask head that consumes mask embeddings, not STAMP head logits.

    Mask tokens first exchange spatial context on their 2D grid. A supervised
    uncertainty branch then gates a feature correction before the final mask
    classifier. No original STAMP classifier output is used in the forward.
    """

    def __init__(self, token_dim: int, hidden_size: int = 256, depth: int = 2) -> None:
        super().__init__()
        self.token_dim = int(token_dim)
        self.hidden_size = int(hidden_size)
        self.depth = int(depth)
        self.input_projection = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, hidden_size),
            nn.GELU(),
        )
        self.context_blocks = nn.ModuleList([SpatialTokenBlock(hidden_size) for _ in range(depth)])
        self.uncertainty_head = nn.Linear(hidden_size, 1)
        self.correction = nn.Sequential(
            nn.Linear(hidden_size + 1, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.mask_classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 2),
        )
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)

    def forward(self, mask_hidden: torch.Tensor, grid_hw: tuple[int, int]) -> dict[str, torch.Tensor]:
        batch_size, num_tokens, _ = mask_hidden.shape
        grid_h, grid_w = int(grid_hw[0]), int(grid_hw[1])
        if num_tokens != grid_h * grid_w:
            raise ValueError(f"Token count {num_tokens} does not match grid {grid_h}x{grid_w}.")
        features = self.input_projection(mask_hidden)
        features = features.reshape(batch_size, grid_h, grid_w, self.hidden_size)
        for block in self.context_blocks:
            features = block(features)
        uncertainty_logits = self.uncertainty_head(features)
        uncertainty = torch.sigmoid(uncertainty_logits)
        correction = self.correction(torch.cat([features, uncertainty], dim=-1))
        refined_features = features + uncertainty * correction
        mask_logits = self.mask_classifier(refined_features).reshape(batch_size, num_tokens, 2)
        uncertainty_logits = uncertainty_logits.reshape(batch_size, num_tokens, 1)
        return {
            "mask_logits": mask_logits,
            "uncertainty": uncertainty.reshape(batch_size, num_tokens, 1),
            "uncertainty_logits": uncertainty_logits,
        }
