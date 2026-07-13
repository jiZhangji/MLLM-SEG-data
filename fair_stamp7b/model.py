from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from onepass_qwen7b.model import segmentation_loss
from onepass_stamp.lora import OnePassLoRALinear, lora_parameters


class SpecialTokenInputAdapter(nn.Module):
    """Train two token embeddings without optimizing the full embedding table."""

    def __init__(self, base: nn.Module, token_ids: tuple[int, int]) -> None:
        super().__init__()
        self.base = base
        self.register_buffer("token_ids", torch.tensor(token_ids, dtype=torch.long), persistent=True)
        self.delta = nn.Parameter(torch.zeros(2, base.weight.shape[1]))

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    @property
    def num_embeddings(self) -> int:
        return int(self.base.weight.shape[0])

    @property
    def embedding_dim(self) -> int:
        return int(self.base.weight.shape[1])

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        output = self.base(input_ids)
        for index, token_id in enumerate(self.token_ids.tolist()):
            output = output + input_ids.eq(token_id).unsqueeze(-1) * self.delta[index].to(output.dtype)
        return output


class SpecialTokenOutputAdapter(nn.Module):
    """Add trainable SEG/MASK output directions while retaining the frozen LM head."""

    def __init__(self, base: nn.Module, token_ids: tuple[int, int], hidden_size: int) -> None:
        super().__init__()
        self.base = base
        self.register_buffer("token_ids", torch.tensor(token_ids, dtype=torch.long), persistent=True)
        self.delta = nn.Parameter(torch.zeros(2, hidden_size))
        self.bias = nn.Parameter(torch.zeros(2))

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        logits = self.base(hidden)
        correction = torch.matmul(hidden.float(), self.delta.t()) + self.bias
        updated = logits.clone()
        updated[..., self.token_ids] = updated[..., self.token_ids] + correction.to(updated.dtype)
        return updated


@dataclass
class FairStamp7BOutput:
    lm_loss: torch.Tensor
    mask_logits: list[torch.Tensor]
    grid_shapes: list[tuple[int, int]]


class FairSTAMP7B(nn.Module):
    """Native two-stage STAMP trained from the same plain Qwen base as OnePass."""

    def __init__(
        self,
        backbone: nn.Module,
        input_adapter: SpecialTokenInputAdapter,
        output_adapter: SpecialTokenOutputAdapter,
        mask_token_id: int,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.mask_token_id = int(mask_token_id)
        self._input_adapter = input_adapter
        self._output_adapter = output_adapter
        self.freeze_base_parameters()

    @property
    def mask_classifier(self) -> nn.Module:
        return self.backbone.classifier

    def freeze_base_parameters(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        for parameter in lora_parameters(self.backbone):
            parameter.requires_grad = True
        self._input_adapter.delta.requires_grad = True
        self._output_adapter.delta.requires_grad = True
        self._output_adapter.bias.requires_grad = True
        for parameter in self.mask_classifier.parameters():
            parameter.requires_grad = True

    def train(self, mode: bool = True):
        super().train(mode)
        for module in self.backbone.modules():
            if isinstance(module, OnePassLoRALinear):
                module.train(mode)
        self.mask_classifier.train(mode)
        return self

    def lora_parameters(self) -> list[nn.Parameter]:
        return lora_parameters(self.backbone)

    def token_parameters(self) -> list[nn.Parameter]:
        return [self._input_adapter.delta, self._output_adapter.delta, self._output_adapter.bias]

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def token_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            "input_delta": self._input_adapter.delta.detach().cpu(),
            "output_delta": self._output_adapter.delta.detach().cpu(),
            "output_bias": self._output_adapter.bias.detach().cpu(),
        }

    def load_token_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        with torch.no_grad():
            self._input_adapter.delta.copy_(state["input_delta"].to(self._input_adapter.delta))
            self._output_adapter.delta.copy_(state["output_delta"].to(self._output_adapter.delta))
            self._output_adapter.bias.copy_(state["output_bias"].to(self._output_adapter.bias))

    def forward(
        self,
        lm_input_ids: torch.Tensor,
        lm_attention_mask: torch.Tensor,
        lm_labels: torch.Tensor,
        cls_input_ids: torch.Tensor,
        cls_attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        grid_shapes: list[tuple[int, int]],
    ) -> FairStamp7BOutput:
        lm_output = self.backbone(
            input_ids=lm_input_ids,
            attention_mask=lm_attention_mask,
            labels=lm_labels,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            use_cache=False,
            return_dict=True,
        )
        mask_logits = self.classify_masks(
            cls_input_ids=cls_input_ids,
            cls_attention_mask=cls_attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            grid_shapes=grid_shapes,
        )
        return FairStamp7BOutput(
            lm_loss=lm_output.loss,
            mask_logits=mask_logits,
            grid_shapes=grid_shapes,
        )

    def classify_masks(
        self,
        *,
        cls_input_ids: torch.Tensor,
        cls_attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        grid_shapes: list[tuple[int, int]],
    ) -> list[torch.Tensor]:
        mask_positions = cls_input_ids.eq(self.mask_token_id) & cls_attention_mask.bool()
        expected = [height * width for height, width in grid_shapes]
        actual = mask_positions.sum(dim=1).tolist()
        if actual != expected:
            raise ValueError(f"Mask counts {actual} do not match grids {expected}.")
        cls_output = self.backbone(
            input_ids=cls_input_ids,
            attention_mask=cls_attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            output_hidden_states=False,
            do_classification=True,
            use_cache=False,
            return_dict=True,
        )
        flat_logits = cls_output.bi_logits[mask_positions].reshape(-1)
        return list(flat_logits.split(expected))


__all__ = [
    "FairSTAMP7B",
    "FairStamp7BOutput",
    "SpecialTokenInputAdapter",
    "SpecialTokenOutputAdapter",
    "segmentation_loss",
]
