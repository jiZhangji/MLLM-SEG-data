from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from refine_stamp.utils.geometry import patch_ids_to_boxes

try:
    from torchvision.ops import roi_align
except Exception:  # pragma: no cover
    roi_align = None


class LocalPatchRefiner(nn.Module):
    """Lightweight RGB-crop and mask-token refiner."""

    def __init__(
        self,
        token_dim: int,
        token_channels: int = 32,
        crop_size: int = 64,
        output_size: int = 16,
        box_scale: float = 1.0,
        refiner_chunk_size: int = 0,
    ) -> None:
        super().__init__()
        if token_dim <= 0:
            raise ValueError("token_dim must be positive.")
        if crop_size <= 0 or output_size <= 0:
            raise ValueError("crop_size and output_size must be positive.")

        self.crop_size = crop_size
        self.output_size = output_size
        self.box_scale = box_scale
        self.refiner_chunk_size = refiner_chunk_size

        self.token_projection = nn.Sequential(
            nn.Linear(token_dim, token_channels),
            nn.GELU(),
            nn.Linear(token_channels, token_channels),
        )

        self.refiner = nn.Sequential(
            nn.Conv2d(3 + token_channels, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 1, kernel_size=1),
        )

    def forward(
        self,
        images: torch.Tensor,
        mask_hidden: torch.Tensor,
        selected_ids: torch.Tensor,
        grid_hw: tuple[int, int],
    ) -> dict[str, torch.Tensor]:
        if roi_align is None:
            raise ImportError("torchvision.ops.roi_align is required for LocalPatchRefiner.")
        if images.ndim != 4:
            raise ValueError(f"images must be [B, 3, H, W], got {tuple(images.shape)}")
        if mask_hidden.ndim != 3:
            raise ValueError(f"mask_hidden must be [B, N, D], got {tuple(mask_hidden.shape)}")
        if selected_ids.ndim != 2:
            raise ValueError(f"selected_ids must be [B, K], got {tuple(selected_ids.shape)}")

        batch_size, _, image_h, image_w = images.shape
        hidden_batch, num_tokens, hidden_dim = mask_hidden.shape
        ids_batch, top_k = selected_ids.shape
        grid_h, grid_w = grid_hw

        if hidden_batch != batch_size or ids_batch != batch_size:
            raise ValueError("Batch size mismatch among images, mask_hidden and selected_ids.")
        if grid_h * grid_w != num_tokens:
            raise ValueError(f"grid_hw={grid_hw} does not match num_tokens={num_tokens}")

        boxes = patch_ids_to_boxes(
            selected_ids=selected_ids,
            grid_hw=grid_hw,
            image_hw=(image_h, image_w),
            scale=self.box_scale,
        )
        box_list = [boxes[index] for index in range(batch_size)]

        image_crops = roi_align(
            input=images,
            boxes=box_list,
            output_size=(self.crop_size, self.crop_size),
            spatial_scale=1.0,
            aligned=True,
        )

        gather_ids = selected_ids.unsqueeze(-1).expand(-1, -1, hidden_dim)
        selected_hidden = torch.gather(mask_hidden, dim=1, index=gather_ids)

        token_features = self.token_projection(selected_hidden)
        token_features = token_features.reshape(batch_size * top_k, -1, 1, 1)
        token_features = token_features.expand(-1, -1, self.crop_size, self.crop_size)

        features = torch.cat([image_crops, token_features], dim=1)
        if self.refiner_chunk_size and self.refiner_chunk_size > 0:
            chunks = []
            for start in range(0, features.shape[0], self.refiner_chunk_size):
                chunks.append(self.refiner(features[start : start + self.refiner_chunk_size]))
            local_logits = torch.cat(chunks, dim=0)
        else:
            local_logits = self.refiner(features)
        local_logits = F.interpolate(
            local_logits,
            size=(self.output_size, self.output_size),
            mode="bilinear",
            align_corners=False,
        )
        local_logits = local_logits.reshape(batch_size, top_k, 1, self.output_size, self.output_size)

        return {"local_logits": local_logits, "boxes": boxes}
