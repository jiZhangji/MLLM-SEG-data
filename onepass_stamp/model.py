from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lora import OnePassLoRALinear, lora_parameters


def _hidden_size(config: Any) -> int:
    value = getattr(config, "hidden_size", None)
    if value is None and hasattr(config, "text_config"):
        value = getattr(config.text_config, "hidden_size", None)
    if value is None:
        raise AttributeError("Could not determine STAMP hidden size from model config.")
    return int(value)


@dataclass
class OnePassOutput:
    mask_logits: list[torch.Tensor]
    grid_shapes: list[tuple[int, int]]
    seg_hidden: torch.Tensor


class OnePassQueryModule(nn.Module):
    """Trainable semantic and spatial queries initialized from STAMP behavior."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.seg_delta = nn.Parameter(torch.zeros(hidden_size))
        self.mask_delta = nn.Parameter(torch.zeros(hidden_size))
        self.task_delta = nn.Parameter(torch.zeros(hidden_size))
        self.visual_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.position_projection = nn.Linear(2, hidden_size, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.eye_(self.visual_projection.weight)
        nn.init.zeros_(self.position_projection.weight)
        nn.init.zeros_(self.seg_delta)
        nn.init.zeros_(self.mask_delta)
        nn.init.zeros_(self.task_delta)

    def spatial_queries(
        self,
        visual_features: torch.Tensor,
        grid_shapes: list[tuple[int, int]],
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        expected = sum(height * width for height, width in grid_shapes)
        if visual_features.shape[0] != expected:
            raise ValueError(
                "Visual feature/query count mismatch: "
                f"features={visual_features.shape[0]}, expected={expected}, grids={grid_shapes}"
            )

        parameter_dtype = self.visual_projection.weight.dtype
        projected_visual = self.visual_projection(visual_features.to(parameter_dtype))
        coordinates = []
        for height, width in grid_shapes:
            rows = torch.linspace(-1.0, 1.0, height, device=visual_features.device, dtype=parameter_dtype)
            cols = torch.linspace(-1.0, 1.0, width, device=visual_features.device, dtype=parameter_dtype)
            row_grid, col_grid = torch.meshgrid(rows, cols, indexing="ij")
            coordinates.append(torch.stack([row_grid, col_grid], dim=-1).reshape(-1, 2))
        position_features = self.position_projection(torch.cat(coordinates, dim=0))
        queries = projected_visual + position_features + self.mask_delta.to(parameter_dtype)
        return queries.to(output_dtype)


class OnePassSTAMP(nn.Module):
    """Single-forward STAMP wrapper with pre-inserted semantic/spatial queries.

    The wrapped STAMP backbone supplies the vision encoder, multimodal
    Transformer, hybrid mask-query attention, and original linear mask
    classifier. No generated logits, correction adapter, or uncertainty branch
    is used.
    """

    def __init__(
        self,
        backbone: nn.Module,
        seg_token_id: int,
        mask_token_id: int,
        task_token_id: int | None,
        merge_size: int,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.seg_token_id = int(seg_token_id)
        self.mask_token_id = int(mask_token_id)
        self.task_token_id = None if task_token_id is None else int(task_token_id)
        self.merge_size = int(merge_size)
        self.query_module = OnePassQueryModule(_hidden_size(backbone.config))
        if self.task_token_id is None:
            self.query_module.task_delta.requires_grad_(False)
        self.freeze_backbone()

    @property
    def mask_classifier(self) -> nn.Module:
        return self.backbone.classifier

    def freeze_backbone(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        for parameter in lora_parameters(self.backbone):
            parameter.requires_grad = True
        for parameter in self.mask_classifier.parameters():
            parameter.requires_grad = True

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep the pretrained representation deterministic while query and
        # classifier parameters are optimized.
        self.backbone.eval()
        for module in self.backbone.modules():
            if isinstance(module, OnePassLoRALinear):
                module.train(mode)
        self.query_module.train(mode)
        self.mask_classifier.train(mode)
        return self

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def lora_parameters(self) -> list[nn.Parameter]:
        return lora_parameters(self.backbone)

    @staticmethod
    def _flatten_image_features(features: Any) -> torch.Tensor:
        if torch.is_tensor(features):
            if features.ndim == 3:
                return features.reshape(-1, features.shape[-1])
            if features.ndim == 2:
                return features
            raise ValueError(f"Unexpected image feature shape: {tuple(features.shape)}")
        values = list(features)
        if not values:
            raise ValueError("STAMP vision encoder returned no image features.")
        return torch.cat(values, dim=0)

    def _grid_shapes(self, image_grid_thw: torch.Tensor) -> list[tuple[int, int]]:
        shapes = []
        for grid in image_grid_thw:
            temporal, height, width = (int(value) for value in grid.tolist())
            if temporal != 1:
                raise ValueError(f"OnePass-STAMP currently supports images only, got temporal grid {temporal}.")
            if height % self.merge_size or width % self.merge_size:
                raise ValueError(
                    f"Image grid {(height, width)} is not divisible by merge_size={self.merge_size}."
                )
            shapes.append((height // self.merge_size, width // self.merge_size))
        return shapes

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> OnePassOutput:
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError("input_ids and attention_mask must have matching [batch, sequence] shapes.")

        seg_positions = input_ids.eq(self.seg_token_id) & attention_mask.bool()
        mask_positions = input_ids.eq(self.mask_token_id) & attention_mask.bool()
        seg_counts = seg_positions.sum(dim=1)
        if not torch.all(seg_counts.eq(1)):
            raise ValueError(f"Each sample must contain exactly one <|seg|>; got {seg_counts.tolist()}.")

        grid_shapes = self._grid_shapes(image_grid_thw)
        expected_counts = torch.tensor(
            [height * width for height, width in grid_shapes], device=input_ids.device, dtype=torch.long
        )
        actual_counts = mask_positions.sum(dim=1)
        if not torch.equal(actual_counts, expected_counts):
            raise ValueError(
                f"Mask-query counts {actual_counts.tolist()} do not match image grids {expected_counts.tolist()}."
            )

        inputs_embeds = self.backbone.model.get_input_embeddings()(input_ids).clone()
        raw_image_features = self.backbone.model.get_image_features(pixel_values, image_grid_thw)
        image_features = self._flatten_image_features(raw_image_features).to(
            device=inputs_embeds.device, dtype=inputs_embeds.dtype
        )
        image_placeholder_mask, _ = self.backbone.model.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, image_features=image_features
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_placeholder_mask, image_features)

        if image_features.shape[0] != int(expected_counts.sum().item()):
            raise ValueError(
                "Merged image feature count does not match spatial-query count: "
                f"features={image_features.shape[0]}, queries={int(expected_counts.sum().item())}."
            )

        inputs_embeds[seg_positions] = (
            inputs_embeds[seg_positions] + self.query_module.seg_delta.to(inputs_embeds.dtype)
        )
        if self.task_token_id is not None:
            task_positions = input_ids.eq(self.task_token_id) & attention_mask.bool()
            task_counts = task_positions.sum(dim=1)
            if not torch.all(task_counts.eq(1)):
                raise ValueError(f"Each sample must contain exactly one task token; got {task_counts.tolist()}.")
            inputs_embeds[task_positions] = (
                inputs_embeds[task_positions] + self.query_module.task_delta.to(inputs_embeds.dtype)
            )

        spatial_queries = self.query_module.spatial_queries(
            image_features, grid_shapes, output_dtype=inputs_embeds.dtype
        )
        inputs_embeds[mask_positions] = inputs_embeds[mask_positions] + spatial_queries

        outputs = self.backbone.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            pixel_values=None,
            image_grid_thw=image_grid_thw,
            seg_mask=mask_positions,
            use_cache=False,
            output_hidden_states=False,
            return_dict=True,
        )
        hidden = outputs.last_hidden_state
        flat_mask_logits = self.mask_classifier(hidden[mask_positions]).squeeze(-1)
        mask_logits = list(flat_mask_logits.split(expected_counts.tolist()))
        seg_hidden = hidden[seg_positions].reshape(input_ids.shape[0], -1)
        return OnePassOutput(mask_logits=mask_logits, grid_shapes=grid_shapes, seg_hidden=seg_hidden)


def onepass_mask_loss(
    mask_logits: list[torch.Tensor],
    targets: list[torch.Tensor],
    dice_weight: float = 1.0,
    eps: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    if len(mask_logits) != len(targets) or not mask_logits:
        raise ValueError("mask_logits and targets must be non-empty lists with equal length.")

    bce_terms = []
    dice_terms = []
    for logits, target in zip(mask_logits, targets):
        target = target.to(device=logits.device, dtype=torch.float32).reshape(-1)
        logits = logits.float().reshape(-1)
        if logits.numel() != target.numel():
            raise ValueError(f"Logit/target size mismatch: {logits.numel()} vs {target.numel()}.")
        bce_terms.append(F.binary_cross_entropy_with_logits(logits, target))
        probabilities = torch.sigmoid(logits)
        intersection = (probabilities * target).sum()
        dice = 1.0 - (2.0 * intersection + eps) / (probabilities.sum() + target.sum() + eps)
        dice_terms.append(dice)

    loss_bce = torch.stack(bce_terms).mean()
    loss_dice = torch.stack(dice_terms).mean()
    loss = loss_bce + float(dice_weight) * loss_dice
    return loss, {
        "loss": float(loss.detach().item()),
        "loss_bce": float(loss_bce.detach().item()),
        "loss_dice": float(loss_dice.detach().item()),
    }
