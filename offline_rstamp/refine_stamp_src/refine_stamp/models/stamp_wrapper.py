from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class FrozenSTAMP(nn.Module):
    """Thin wrapper around a STAMP model that exposes refinement tensors.

    The wrapped model is expected to implement ``forward_for_refinement`` and
    return ``mask_logits``, ``mask_hidden`` and ``grid_hw``.
    """

    def __init__(self, stamp_model: nn.Module) -> None:
        super().__init__()
        self.stamp = stamp_model
        self.freeze_stamp()

    def freeze_stamp(self) -> None:
        self.stamp.eval()
        for parameter in self.stamp.parameters():
            parameter.requires_grad = False

    @torch.no_grad()
    def forward(
        self,
        images: torch.Tensor,
        texts: list[str],
        **kwargs: Any,
    ) -> dict[str, Any]:
        if not hasattr(self.stamp, "forward_for_refinement"):
            raise AttributeError(
                "The wrapped STAMP model must implement forward_for_refinement()."
            )

        outputs = self.stamp.forward_for_refinement(
            images=images,
            texts=texts,
            **kwargs,
        )

        required_keys = {"mask_logits", "mask_hidden", "grid_hw"}
        missing = required_keys - set(outputs)
        if missing:
            raise KeyError(f"Missing STAMP outputs: {sorted(missing)}")

        return outputs
