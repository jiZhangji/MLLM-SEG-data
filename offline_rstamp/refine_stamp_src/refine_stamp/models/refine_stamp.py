from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from refine_stamp.models.local_refiner import LocalPatchRefiner
from refine_stamp.models.patch_selector import PatchSelector


class RefineSTAMP(nn.Module):
    """Frozen STAMP plus sparse local refinement."""

    def __init__(
        self,
        frozen_stamp: nn.Module,
        token_dim: int,
        top_k: int = 64,
        boundary_weight: float = 0.5,
        selector_mode: str = "hybrid",
        crop_size: int = 64,
        output_size: int = 16,
        token_channels: int = 32,
        box_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.stamp = frozen_stamp
        self.selector = PatchSelector(
            top_k=top_k,
            boundary_weight=boundary_weight,
            mode=selector_mode,
        )
        self.local_refiner = LocalPatchRefiner(
            token_dim=token_dim,
            token_channels=token_channels,
            crop_size=crop_size,
            output_size=output_size,
            box_scale=box_scale,
        )

    def train(self, mode: bool = True) -> "RefineSTAMP":
        super().train(mode)
        self.stamp.eval()
        return self

    def assert_stamp_frozen(self) -> None:
        for name, parameter in self.stamp.named_parameters():
            if parameter.requires_grad:
                raise AssertionError(f"STAMP parameter is trainable: {name}")

    def forward(
        self,
        images: torch.Tensor,
        texts: list[str],
        **stamp_kwargs: Any,
    ) -> dict[str, Any]:
        self.assert_stamp_frozen()
        with torch.no_grad():
            stamp_outputs = self.stamp(images=images, texts=texts, **stamp_kwargs)

        mask_logits = stamp_outputs["mask_logits"]
        mask_hidden = stamp_outputs["mask_hidden"]
        grid_hw = stamp_outputs["grid_hw"]

        selection = self.selector(mask_logits=mask_logits, grid_hw=grid_hw)
        refinement = self.local_refiner(
            images=images,
            mask_hidden=mask_hidden,
            selected_ids=selection["selected_ids"],
            grid_hw=grid_hw,
        )

        return {**stamp_outputs, **selection, **refinement}
