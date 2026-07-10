from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class OnlineCalibratedResidualRefiner(nn.Module):
    """Online scheme B: calibrate STAMP logits, then predict residual corrections.

    This module is intentionally independent from STAMP internals. The official
    training forward only needs to provide:

    - mask_hidden: [B, N, D]
    - mask_logits: [B, N, 2] or binary foreground logits [B, N]

    It preserves the original STAMP head output as the teacher/base signal, but
    lets a tiny trainable scale/bias and residual branch adapt it to the target
    mask tokens.
    """

    def __init__(
        self,
        token_dim: int,
        hidden_size: int = 128,
        use_uncertainty_gate: bool = False,
        trainable_logit_calibration: bool = True,
    ) -> None:
        super().__init__()
        self.use_uncertainty_gate = use_uncertainty_gate
        if trainable_logit_calibration:
            self.logit_scale = nn.Parameter(torch.zeros(2))
            self.logit_bias = nn.Parameter(torch.zeros(2))
        else:
            self.register_buffer("logit_scale", torch.zeros(2), persistent=False)
            self.register_buffer("logit_bias", torch.zeros(2), persistent=False)
        self.residual = nn.Sequential(
            nn.LayerNorm(token_dim + 4),
            nn.Linear(token_dim + 4, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 2),
        )
        self._init_residual_as_identity()

    def _init_residual_as_identity(self) -> None:
        last = self.residual[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    @staticmethod
    def ensure_two_class_logits(mask_logits: torch.Tensor) -> torch.Tensor:
        if mask_logits.ndim == 2:
            return torch.stack([-mask_logits, mask_logits], dim=-1)
        if mask_logits.ndim == 3 and mask_logits.shape[-1] == 1:
            binary_logits = mask_logits.squeeze(-1)
            return torch.stack([-binary_logits, binary_logits], dim=-1)
        if mask_logits.ndim == 3 and mask_logits.shape[-1] == 2:
            return mask_logits
        raise ValueError(f"Expected mask logits [B,N], [B,N,1], or [B,N,2], got {tuple(mask_logits.shape)}")

    def forward(self, mask_hidden: torch.Tensor, mask_logits: torch.Tensor) -> dict[str, torch.Tensor]:
        mask_logits = self.ensure_two_class_logits(mask_logits).float()
        base_logits = mask_logits * torch.exp(self.logit_scale).view(1, 1, 2) + self.logit_bias.view(1, 1, 2)
        fg_prob = torch.softmax(base_logits, dim=-1)[..., 1:2]
        uncertainty = 1.0 - torch.abs(2.0 * fg_prob - 1.0)
        features = torch.cat([mask_hidden.float(), base_logits, fg_prob, uncertainty], dim=-1)
        delta_logits = self.residual(features)
        gate = uncertainty if self.use_uncertainty_gate else torch.ones_like(uncertainty)
        refined_logits = base_logits + gate * delta_logits
        return {
            "base_logits": base_logits,
            "refined_logits": refined_logits,
            "delta_logits": delta_logits,
            "fg_prob": fg_prob,
            "uncertainty": uncertainty,
        }


class UncertaintyAwareMaskHead(nn.Module):
    """Enhanced STAMP mask head: mask_hidden -> uncertainty-aware final logits.

    This is the cleaner online variant. It does not take STAMP's previous
    `bi_logits` as an external input. Instead, it extends the original linear
    mask classifier with an uncertainty branch and a hidden-state residual head.
    """

    def __init__(
        self,
        token_dim: int,
        hidden_size: int = 128,
        use_uncertainty_gate: bool = False,
    ) -> None:
        super().__init__()
        self.use_uncertainty_gate = use_uncertainty_gate
        self.base_classifier = nn.Linear(token_dim, 1)
        self.uncertainty_head = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )
        self.residual_head = nn.Sequential(
            nn.LayerNorm(token_dim + 2),
            nn.Linear(token_dim + 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )
        self._init_residual_as_identity()

    def _init_residual_as_identity(self) -> None:
        last = self.residual_head[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    @torch.no_grad()
    def initialize_from_classifier(self, classifier: nn.Linear) -> None:
        self.base_classifier.weight.copy_(classifier.weight)
        if classifier.bias is not None and self.base_classifier.bias is not None:
            self.base_classifier.bias.copy_(classifier.bias)

    def forward(self, mask_hidden: torch.Tensor) -> dict[str, torch.Tensor]:
        mask_hidden = mask_hidden.float()
        base_binary_logits = self.base_classifier(mask_hidden)
        fg_prob = torch.sigmoid(base_binary_logits)
        logit_uncertainty = 1.0 - torch.abs(2.0 * fg_prob - 1.0)
        learned_uncertainty = torch.sigmoid(self.uncertainty_head(mask_hidden))
        uncertainty = 0.5 * (logit_uncertainty + learned_uncertainty)
        features = torch.cat([mask_hidden, fg_prob, uncertainty], dim=-1)
        delta_binary_logits = self.residual_head(features)
        gate = uncertainty if self.use_uncertainty_gate else torch.ones_like(uncertainty)
        refined_binary_logits = base_binary_logits + gate * delta_binary_logits
        refined_logits = torch.cat([-refined_binary_logits, refined_binary_logits], dim=-1)
        return {
            "base_binary_logits": base_binary_logits,
            "refined_binary_logits": refined_binary_logits,
            "refined_logits": refined_logits,
            "delta_binary_logits": delta_binary_logits,
            "fg_prob": fg_prob,
            "uncertainty": uncertainty,
            "learned_uncertainty": learned_uncertainty,
            "logit_uncertainty": logit_uncertainty,
        }


def token_refine_loss(
    refined_logits: torch.Tensor,
    target_tokens: torch.Tensor,
    uncertainty: torch.Tensor | None = None,
    uncertainty_loss_weight: float = 2.0,
    delta_logits: torch.Tensor | None = None,
    delta_reg_weight: float = 0.01,
) -> torch.Tensor:
    ce = F.cross_entropy(
        refined_logits.reshape(-1, 2),
        target_tokens.reshape(-1).long(),
        reduction="none",
    ).reshape_as(target_tokens).float()
    if uncertainty is not None:
        weights = 1.0 + uncertainty_loss_weight * uncertainty.squeeze(-1).detach()
        loss = (ce * weights).mean()
    else:
        loss = ce.mean()
    if delta_logits is not None and uncertainty is not None and delta_reg_weight > 0:
        loss = loss + delta_reg_weight * ((1.0 - uncertainty) * delta_logits.pow(2).sum(dim=-1)).mean()
    return loss
