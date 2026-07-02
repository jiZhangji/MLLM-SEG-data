"""PyTorch modules for R-STAMP integration.

These modules are intentionally small. They are meant to be imported by the
STAMP model code after the official STAMP baseline is running.
"""

from __future__ import annotations


class MissingTorchError(RuntimeError):
    pass


def _require_torch():
    try:
        import torch
        import torch.nn as nn
        return torch, nn
    except Exception as exc:  # pragma: no cover - used on offline servers
        raise MissingTorchError("PyTorch is required for rstamp.modules") from exc


def build_prior_projector(input_dim: int, hidden_dim: int, num_layers: int = 2):
    """Build a small MLP projector for structured-prior embeddings."""
    torch, nn = _require_torch()
    layers = []
    dim = input_dim
    for _ in range(max(1, num_layers - 1)):
        layers.extend([nn.Linear(dim, hidden_dim), nn.GELU()])
        dim = hidden_dim
    layers.append(nn.Linear(dim, hidden_dim))
    return nn.Sequential(*layers)


class PriorGatedFusion:
    """Factory wrapper for a gated fusion module.

    Usage inside STAMP-like model code:

    ```python
    fusion = PriorGatedFusion.create(hidden_size)
    mask_tokens = fusion(mask_tokens, prior_context)
    ```
    """

    @staticmethod
    def create(hidden_size: int):
        torch, nn = _require_torch()

        class _PriorGatedFusion(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate = nn.Sequential(
                    nn.Linear(hidden_size * 2, hidden_size),
                    nn.Sigmoid(),
                )
                self.proj = nn.Linear(hidden_size, hidden_size)

            def forward(self, mask_tokens, prior_context):
                if prior_context.dim() == 3:
                    prior_context = prior_context.mean(dim=1, keepdim=True)
                prior_context = prior_context.expand_as(mask_tokens)
                gate = self.gate(torch.cat([mask_tokens, prior_context], dim=-1))
                return mask_tokens + gate * self.proj(prior_context)

        return _PriorGatedFusion()


class UncertaintyRefinementHead:
    """Factory wrapper for a lightweight uncertainty/refinement head."""

    @staticmethod
    def create(hidden_size: int):
        torch, nn = _require_torch()

        class _UncertaintyRefinementHead(nn.Module):
            def __init__(self):
                super().__init__()
                self.uncertainty = nn.Linear(hidden_size, 1)
                self.delta = nn.Linear(hidden_size, 1)

            def forward(self, patch_features):
                uncertainty_logits = self.uncertainty(patch_features)
                delta_logits = self.delta(patch_features)
                return {
                    "uncertainty_logits": uncertainty_logits,
                    "delta_logits": delta_logits,
                }

        return _UncertaintyRefinementHead()

