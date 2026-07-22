from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from onepass_stamp.lora import OnePassLoRALinear, lora_parameters


def hidden_size(config: Any) -> int:
    value = getattr(config, "hidden_size", None)
    if value is None and hasattr(config, "text_config"):
        value = getattr(config.text_config, "hidden_size", None)
    if value is None:
        raise AttributeError("Could not determine Qwen2-VL hidden size.")
    return int(value)


@dataclass
class OnePass7BOutput:
    mask_logits: list[torch.Tensor]
    raw_mask_logits: list[torch.Tensor]
    seg_logits: list[torch.Tensor] | None
    mask_hidden: list[torch.Tensor]
    grid_shapes: list[tuple[int, int]]
    seg_hidden: torch.Tensor


class SegGroundingHead(nn.Module):
    """Ground a contextual SEG representation against spatial MASK states."""

    def __init__(
        self,
        hidden_size: int,
        projection_size: int = 256,
        temperature: float = 0.1,
        use_fusion: bool = True,
        fixed_fusion_alpha: float | None = None,
    ) -> None:
        super().__init__()
        if projection_size <= 0:
            raise ValueError("SEG grounding projection size must be positive.")
        if temperature <= 0:
            raise ValueError("SEG grounding temperature must be positive.")
        self.seg_projection = nn.Linear(hidden_size, projection_size, bias=False)
        self.mask_projection = nn.Linear(hidden_size, projection_size, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / temperature)))
        self.fusion_weight = nn.Parameter(torch.zeros(()), requires_grad=bool(use_fusion))
        self.use_fusion = bool(use_fusion)
        self.fixed_fusion_alpha = (
            None if fixed_fusion_alpha is None else float(fixed_fusion_alpha)
        )
        if self.fixed_fusion_alpha is not None:
            self.fusion_weight.requires_grad_(False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.seg_projection.weight)
        nn.init.xavier_uniform_(self.mask_projection.weight)

    def forward(
        self,
        seg_hidden: torch.Tensor,
        mask_hidden: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        if seg_hidden.ndim != 2 or seg_hidden.shape[0] != len(mask_hidden):
            raise ValueError(
                f"SEG batch {tuple(seg_hidden.shape)} does not match {len(mask_hidden)} MASK groups."
            )
        seg_query = F.normalize(self.seg_projection(seg_hidden.float()), dim=-1)
        scale = self.logit_scale.clamp(max=math.log(100.0)).exp()
        values = []
        for query, hidden in zip(seg_query, mask_hidden):
            mask_keys = F.normalize(self.mask_projection(hidden.float()), dim=-1)
            values.append(torch.matmul(mask_keys, query) * scale)
        return values

    def fusion_alpha(self) -> torch.Tensor:
        if not self.use_fusion:
            return self.fusion_weight.detach() * 0.0
        if self.fixed_fusion_alpha is not None:
            return self.fusion_weight.new_tensor(self.fixed_fusion_alpha)
        return torch.tanh(self.fusion_weight)


class SegMaskQueryBuilder(nn.Module):
    def __init__(self, hidden_size: int, max_grid_height: int, max_grid_width: int) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.max_grid_height = int(max_grid_height)
        self.max_grid_width = int(max_grid_width)
        self.seg_embedding = nn.Parameter(torch.empty(hidden_size))
        self.mask_embedding = nn.Parameter(torch.empty(hidden_size))
        self.visual_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.row_embedding = nn.Embedding(max_grid_height, hidden_size)
        self.col_embedding = nn.Embedding(max_grid_width, hidden_size)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.seg_embedding, std=0.02)
        nn.init.normal_(self.mask_embedding, std=0.02)
        nn.init.eye_(self.visual_projection.weight)
        nn.init.zeros_(self.row_embedding.weight)
        nn.init.zeros_(self.col_embedding.weight)

    def initialize_semantic_embeddings(self, seg_vector: torch.Tensor, mask_vector: torch.Tensor) -> None:
        with torch.no_grad():
            self.seg_embedding.copy_(seg_vector.to(self.seg_embedding))
            self.mask_embedding.copy_(mask_vector.to(self.mask_embedding))

    def spatial_queries(
        self,
        visual_features: torch.Tensor,
        grid_shapes: list[tuple[int, int]],
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        expected = sum(height * width for height, width in grid_shapes)
        if visual_features.shape[0] != expected:
            raise ValueError(f"Visual feature count {visual_features.shape[0]} does not match {expected} queries.")
        parameter_dtype = self.visual_projection.weight.dtype
        projected = self.visual_projection(visual_features.to(parameter_dtype))
        rows = []
        cols = []
        for height, width in grid_shapes:
            if height > self.max_grid_height or width > self.max_grid_width:
                raise ValueError(
                    f"Grid {(height, width)} exceeds configured maximum "
                    f"{(self.max_grid_height, self.max_grid_width)}."
                )
            rows.append(torch.arange(height, device=visual_features.device).repeat_interleave(width))
            cols.append(torch.arange(width, device=visual_features.device).repeat(height))
        row_ids = torch.cat(rows)
        col_ids = torch.cat(cols)
        position = self.row_embedding(row_ids) + self.col_embedding(col_ids)
        queries = projected + position + self.mask_embedding.to(parameter_dtype)
        return queries.to(output_dtype)


class OnePassQwen7B(nn.Module):
    """One-forward segmentation with either base or STAMP-LoRA initialization."""

    def __init__(
        self,
        backbone: nn.Module,
        seg_token_id: int,
        mask_token_id: int,
        merge_size: int,
        max_grid_height: int = 512,
        max_grid_width: int = 512,
        train_visual_projection: bool = True,
        train_position_embeddings: bool = True,
        train_classifier: bool = True,
        use_seg_grounding: bool = False,
        seg_grounding_size: int = 256,
        seg_grounding_temperature: float = 0.1,
        use_seg_fusion: bool = True,
        seg_fusion_alpha: float | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.seg_token_id = int(seg_token_id)
        self.mask_token_id = int(mask_token_id)
        self.merge_size = int(merge_size)
        self.query_builder = SegMaskQueryBuilder(
            hidden_size(backbone.config), max_grid_height, max_grid_width
        )
        self.seg_grounding_head = (
            SegGroundingHead(
                hidden_size(backbone.config),
                projection_size=seg_grounding_size,
                temperature=seg_grounding_temperature,
                use_fusion=use_seg_fusion,
                fixed_fusion_alpha=seg_fusion_alpha,
            )
            if use_seg_grounding
            else None
        )
        self.train_visual_projection = bool(train_visual_projection)
        self.train_position_embeddings = bool(train_position_embeddings)
        self.train_classifier = bool(train_classifier)
        self.freeze_base_parameters()

    @property
    def mask_classifier(self) -> nn.Module:
        return self.backbone.classifier

    def freeze_base_parameters(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        for parameter in lora_parameters(self.backbone):
            parameter.requires_grad = True
        for parameter in self.query_builder.parameters():
            parameter.requires_grad = True
        if not self.train_visual_projection:
            for parameter in self.query_builder.visual_projection.parameters():
                parameter.requires_grad = False
        if not self.train_position_embeddings:
            for parameter in self.query_builder.row_embedding.parameters():
                parameter.requires_grad = False
            for parameter in self.query_builder.col_embedding.parameters():
                parameter.requires_grad = False
        for parameter in self.mask_classifier.parameters():
            parameter.requires_grad = self.train_classifier

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.train(mode)
        for module in self.backbone.modules():
            if isinstance(module, OnePassLoRALinear):
                module.train(mode)
        self.query_builder.train(mode)
        if self.seg_grounding_head is not None:
            self.seg_grounding_head.train(mode)
        self.mask_classifier.train(mode and self.train_classifier)
        return self

    def lora_parameters(self) -> list[nn.Parameter]:
        return lora_parameters(self.backbone)

    def seg_grounding_parameters(self) -> list[nn.Parameter]:
        if self.seg_grounding_head is None:
            return []
        return [parameter for parameter in self.seg_grounding_head.parameters() if parameter.requires_grad]

    def seg_fusion_alpha(self) -> float:
        if self.seg_grounding_head is None:
            return 0.0
        return float(self.seg_grounding_head.fusion_alpha().detach().item())

    def logits_for_seg_hidden(
        self,
        raw_mask_logits: list[torch.Tensor],
        mask_hidden: list[torch.Tensor],
        seg_hidden: torch.Tensor,
        fusion_alpha: float | torch.Tensor | None = None,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Recompute only the explicit SEG-grounding branch for diagnostics."""
        if self.seg_grounding_head is None:
            raise RuntimeError("SEG-grounding diagnostics require a SEG grounding head.")
        if len(raw_mask_logits) != len(mask_hidden):
            raise ValueError("Raw-logit and MASK-hidden group counts do not match.")
        seg_logits = self.seg_grounding_head(seg_hidden, mask_hidden)
        alpha = (
            self.seg_grounding_head.fusion_alpha()
            if fusion_alpha is None
            else raw_mask_logits[0].new_tensor(fusion_alpha)
        )
        mask_logits = [raw + alpha * seg for raw, seg in zip(raw_mask_logits, seg_logits)]
        return mask_logits, seg_logits

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    @staticmethod
    def flatten_image_features(features: Any) -> torch.Tensor:
        if torch.is_tensor(features):
            if features.ndim == 3:
                return features.reshape(-1, features.shape[-1])
            if features.ndim == 2:
                return features
            raise ValueError(f"Unexpected image feature shape {tuple(features.shape)}")
        values = list(features)
        if not values:
            raise ValueError("Vision encoder returned no image features.")
        return torch.cat(values, dim=0)

    def grid_shapes(self, image_grid_thw: torch.Tensor) -> list[tuple[int, int]]:
        shapes = []
        for grid in image_grid_thw:
            temporal, raw_height, raw_width = (int(value) for value in grid.tolist())
            if temporal != 1 or raw_height % self.merge_size or raw_width % self.merge_size:
                raise ValueError(
                    f"Unsupported image grid {(temporal, raw_height, raw_width)} "
                    f"for merge_size={self.merge_size}."
                )
            shapes.append((raw_height // self.merge_size, raw_width // self.merge_size))
        return shapes

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        disable_seg_query: bool = False,
        seg_query_targets: list[torch.Tensor] | None = None,
    ) -> OnePass7BOutput:
        if disable_seg_query and seg_query_targets is not None:
            raise ValueError("disable_seg_query and seg_query_targets cannot be used together.")
        seg_positions = input_ids.eq(self.seg_token_id) & attention_mask.bool()
        mask_positions = input_ids.eq(self.mask_token_id) & attention_mask.bool()
        seg_counts = seg_positions.sum(dim=1)
        if not torch.all(seg_counts.eq(1)):
            raise ValueError(f"Each sample must contain one SEG query; got {seg_counts.tolist()}.")
        grid_shapes = self.grid_shapes(image_grid_thw)
        expected = torch.tensor(
            [height * width for height, width in grid_shapes],
            device=input_ids.device,
            dtype=torch.long,
        )
        actual = mask_positions.sum(dim=1)
        if not torch.equal(actual, expected):
            raise ValueError(f"Mask counts {actual.tolist()} do not match visual grids {expected.tolist()}.")

        inputs_embeds = self.backbone.model.get_input_embeddings()(input_ids).clone()
        raw_features = self.backbone.model.get_image_features(pixel_values, image_grid_thw)
        image_features = self.flatten_image_features(raw_features).to(inputs_embeds)
        image_mask, _ = self.backbone.model.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, image_features=image_features
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_features)
        if image_features.shape[0] != int(expected.sum().item()):
            raise ValueError(
                f"Merged image feature count {image_features.shape[0]} does not match "
                f"mask-query count {int(expected.sum().item())}."
            )
        if disable_seg_query:
            inputs_embeds[seg_positions] = 0
        elif seg_query_targets is not None:
            if len(seg_query_targets) != input_ids.shape[0]:
                raise ValueError(
                    f"Received {len(seg_query_targets)} SEG targets for batch size {input_ids.shape[0]}."
                )
            image_groups = image_features.split(expected.tolist())
            prototypes = []
            for features, target in zip(image_groups, seg_query_targets):
                weights = target.to(device=features.device, dtype=torch.float32).reshape(-1)
                if weights.numel() != features.shape[0]:
                    raise ValueError(
                        f"SEG target length {weights.numel()} does not match "
                        f"visual feature count {features.shape[0]}."
                    )
                denominator = weights.sum()
                if denominator <= 0:
                    raise ValueError("GT foreground prototype has zero foreground weight.")
                prototype = (features.float() * weights.unsqueeze(-1)).sum(dim=0) / denominator
                prototypes.append(prototype)
            inputs_embeds[seg_positions] = torch.stack(prototypes).to(inputs_embeds)
        else:
            inputs_embeds[seg_positions] = self.query_builder.seg_embedding.to(inputs_embeds.dtype)
        spatial_queries = self.query_builder.spatial_queries(
            image_features, grid_shapes, output_dtype=inputs_embeds.dtype
        )
        inputs_embeds[mask_positions] = spatial_queries

        model_attention_mask = attention_mask
        if disable_seg_query:
            model_attention_mask = attention_mask.clone()
            model_attention_mask[seg_positions] = 0

        outputs = self.backbone.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=model_attention_mask,
            pixel_values=None,
            image_grid_thw=image_grid_thw,
            seg_mask=mask_positions,
            use_cache=False,
            output_hidden_states=False,
            return_dict=True,
        )
        hidden = outputs.last_hidden_state
        flat_mask_hidden = hidden[mask_positions]
        mask_hidden = list(flat_mask_hidden.split(expected.tolist()))
        flat_logits = self.mask_classifier(flat_mask_hidden).squeeze(-1)
        raw_mask_logits = list(flat_logits.split(expected.tolist()))
        seg_hidden = hidden[seg_positions].reshape(input_ids.shape[0], -1)
        seg_logits = None
        mask_logits = raw_mask_logits
        if self.seg_grounding_head is not None and not disable_seg_query:
            mask_logits, seg_logits = self.logits_for_seg_hidden(
                raw_mask_logits, mask_hidden, seg_hidden
            )
        return OnePass7BOutput(
            mask_logits=mask_logits,
            raw_mask_logits=raw_mask_logits,
            seg_logits=seg_logits,
            mask_hidden=mask_hidden,
            grid_shapes=grid_shapes,
            seg_hidden=seg_hidden,
        )


def segmentation_loss(
    mask_logits: list[torch.Tensor],
    targets: list[torch.Tensor],
    bce_weight: float = 0.3,
    dice_weight: float = 0.7,
    eps: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    if len(mask_logits) != len(targets) or not mask_logits:
        raise ValueError("mask_logits and targets must be non-empty lists with equal length.")
    bce_values = []
    dice_values = []
    for logits, target in zip(mask_logits, targets):
        logits = logits.float().reshape(-1)
        target = target.to(device=logits.device, dtype=torch.float32).reshape(-1)
        if logits.shape != target.shape:
            raise ValueError(f"Logit/target mismatch: {tuple(logits.shape)} vs {tuple(target.shape)}")
        bce_values.append(F.binary_cross_entropy_with_logits(logits, target))
        probability = torch.sigmoid(logits)
        intersection = (probability * target).sum()
        dice = 1.0 - (2.0 * intersection + eps) / (probability.sum() + target.sum() + eps)
        dice_values.append(dice)
    loss_bce = torch.stack(bce_values).mean()
    loss_dice = torch.stack(dice_values).mean()
    loss = float(bce_weight) * loss_bce + float(dice_weight) * loss_dice
    return loss, {
        "loss": float(loss.detach().item()),
        "loss_bce": float(loss_bce.detach().item()),
        "loss_dice": float(loss_dice.detach().item()),
    }
