from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from local_refine.geometry import patch_ids_to_boxes


class LocalPatchRefiner(nn.Module):
    """Small CNN that predicts refined logits for selected local crops."""

    def __init__(
        self,
        token_dim: int,
        token_channels: int = 32,
        crop_size: int = 64,
        output_size: int = 16,
        box_scale: float = 1.0,
        chunk_size: int = 16,
    ) -> None:
        super().__init__()
        self.crop_size = crop_size
        self.output_size = output_size
        self.box_scale = box_scale
        self.chunk_size = chunk_size

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
        self._cudnn_fallback_active = False

    def _run_refiner(self, features: torch.Tensor) -> torch.Tensor:
        if self._cudnn_fallback_active:
            with torch.backends.cudnn.flags(enabled=False):
                return self.refiner(features)
        try:
            return self.refiner(features)
        except RuntimeError as exc:
            message = str(exc)
            if "CUDNN_STATUS_NOT_INITIALIZED" not in message and "cuDNN error" not in message:
                raise
            self._cudnn_fallback_active = True
            print(
                "[WARN] cuDNN failed inside LocalPatchRefiner; falling back to native CUDA conv.",
                flush=True,
            )
            with torch.backends.cudnn.flags(enabled=False):
                return self.refiner(features)

    def forward(
        self,
        images: torch.Tensor,
        mask_hidden: torch.Tensor,
        selected_ids: torch.Tensor,
        grid_hw: tuple[int, int],
    ) -> dict[str, torch.Tensor]:
        batch_size, _, image_h, image_w = images.shape
        _, _, hidden_dim = mask_hidden.shape
        _, top_k = selected_ids.shape

        boxes = patch_ids_to_boxes(selected_ids, grid_hw, (image_h, image_w), self.box_scale)
        image_crops = crop_and_resize(images, boxes, self.crop_size)

        gather_ids = selected_ids.unsqueeze(-1).expand(-1, -1, hidden_dim)
        selected_hidden = torch.gather(mask_hidden, dim=1, index=gather_ids)
        token_features = self.token_projection(selected_hidden).reshape(batch_size * top_k, -1, 1, 1)
        token_features = token_features.expand(-1, -1, self.crop_size, self.crop_size)
        features = torch.cat([image_crops, token_features], dim=1)

        if self.chunk_size and self.chunk_size > 0:
            logits = torch.cat(
                [self._run_refiner(features[start : start + self.chunk_size]) for start in range(0, features.shape[0], self.chunk_size)],
                dim=0,
            )
        else:
            logits = self._run_refiner(features)
        logits = F.interpolate(logits, size=(self.output_size, self.output_size), mode="bilinear", align_corners=False)
        return {"local_logits": logits.reshape(batch_size, top_k, 1, self.output_size, self.output_size), "boxes": boxes}


def crop_and_resize(images: torch.Tensor, boxes: torch.Tensor, output_size: int) -> torch.Tensor:
    crops = []
    _, _, image_h, image_w = images.shape
    for batch_index in range(images.shape[0]):
        for box in boxes[batch_index]:
            x1, y1, x2, y2 = box.round().long().tolist()
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(x2, image_w), min(y2, image_h)
            if x2 <= x1 or y2 <= y1:
                crop = images.new_zeros((1, images.shape[1], output_size, output_size))
            else:
                crop = F.interpolate(
                    images[batch_index : batch_index + 1, :, y1:y2, x1:x2],
                    size=(output_size, output_size),
                    mode="bilinear",
                    align_corners=False,
                )
            crops.append(crop)
    return torch.cat(crops, dim=0)
