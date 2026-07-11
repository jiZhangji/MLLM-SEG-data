from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn


DEFAULT_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class OnePassLoRALinear(nn.Module):
    """Minimal in-place LoRA wrapper that preserves the original Linear API."""

    def __init__(self, base_layer: nn.Linear, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive.")
        self.base_layer = base_layer
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()
        self.lora_a = nn.Linear(base_layer.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base_layer.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)
        # Keep trainable adapters in fp32 for stable AdamW states. Autocast will
        # still execute their GEMMs in bf16/fp16 on CUDA.
        self.lora_a.to(device=base_layer.weight.device)
        self.lora_b.to(device=base_layer.weight.device)
        for parameter in self.base_layer.parameters():
            parameter.requires_grad = False

    @property
    def in_features(self) -> int:
        return self.base_layer.in_features

    @property
    def out_features(self) -> int:
        return self.base_layer.out_features

    @property
    def weight(self) -> torch.Tensor:
        return self.base_layer.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return self.base_layer.bias

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base = self.base_layer(inputs)
        lora_inputs = self.dropout(inputs).to(self.lora_a.weight.dtype)
        update = self.lora_b(self.lora_a(lora_inputs)) * self.scaling
        return base + update.to(base.dtype)


def inject_lora(
    model: nn.Module,
    *,
    rank: int,
    alpha: float,
    dropout: float,
    target_names: Iterable[str] = DEFAULT_TARGETS,
) -> int:
    targets = set(target_names)
    replacements: list[tuple[nn.Module, str, nn.Linear]] = []
    for parent in model.modules():
        for child_name, child in parent.named_children():
            if child_name in targets and isinstance(child, nn.Linear):
                replacements.append((parent, child_name, child))
    if not replacements:
        raise ValueError(f"No Linear modules matched LoRA targets {sorted(targets)}.")
    for parent, child_name, child in replacements:
        setattr(parent, child_name, OnePassLoRALinear(child, rank, alpha, dropout))
    return len(replacements)


def lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    values = []
    for module in model.modules():
        if isinstance(module, OnePassLoRALinear):
            values.extend(module.lora_a.parameters())
            values.extend(module.lora_b.parameters())
    return values


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if ".lora_a." in key or ".lora_b." in key
    }


def load_lora_state_dict(model: nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    expected = {key for key in model.state_dict() if ".lora_a." in key or ".lora_b." in key}
    received = set(state_dict)
    if received != expected:
        missing = sorted(expected - received)
        unexpected = sorted(received - expected)
        raise ValueError(f"LoRA checkpoint mismatch; missing={missing[:5]}, unexpected={unexpected[:5]}.")
    current = model.state_dict()
    with torch.no_grad():
        for key, value in state_dict.items():
            current[key].copy_(value.to(device=current[key].device, dtype=current[key].dtype))
