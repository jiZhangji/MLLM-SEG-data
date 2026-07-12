from __future__ import annotations

import torch
import torch.nn as nn

from fair_stamp7b.data import _last_subsequence
from fair_stamp7b.model import SpecialTokenInputAdapter, SpecialTokenOutputAdapter
from onepass_stamp.lora import OnePassLoRALinear
from onepass_qwen7b.runtime import enable_gradient_checkpointing


def test_last_subsequence_uses_final_assistant_header() -> None:
    sequence = torch.tensor([1, 7, 8, 2, 7, 8, 3])
    assert _last_subsequence(sequence, [7, 8]) == 4
    assert _last_subsequence(sequence, [9]) == -1


def test_special_input_adapter_changes_only_selected_tokens() -> None:
    embedding = nn.Embedding(6, 4)
    adapter = SpecialTokenInputAdapter(embedding, (4, 5))
    with torch.no_grad():
        adapter.delta[0].fill_(2.0)
        adapter.delta[1].fill_(3.0)
    ids = torch.tensor([[1, 4, 5]])
    expected = embedding(ids).detach()
    output = adapter(ids)
    assert torch.allclose(output[:, 0], expected[:, 0])
    assert torch.allclose(output[:, 1], expected[:, 1] + 2.0)
    assert torch.allclose(output[:, 2], expected[:, 2] + 3.0)


def test_special_output_adapter_updates_only_selected_columns() -> None:
    head = nn.Linear(4, 6, bias=False)
    adapter = SpecialTokenOutputAdapter(head, (4, 5), hidden_size=4)
    with torch.no_grad():
        adapter.delta[0].fill_(1.0)
        adapter.delta[1].fill_(2.0)
        adapter.bias.copy_(torch.tensor([0.5, -0.5]))
    hidden = torch.ones(1, 2, 4)
    baseline = head(hidden).detach()
    output = adapter(hidden)
    assert torch.allclose(output[..., :4], baseline[..., :4])
    assert torch.allclose(output[..., 4], baseline[..., 4] + 4.5)
    assert torch.allclose(output[..., 5], baseline[..., 5] + 7.5)


def test_rslora_uses_sqrt_rank_scaling() -> None:
    layer = OnePassLoRALinear(nn.Linear(4, 4), rank=4, alpha=8.0, dropout=0.0, use_rslora=True)
    assert layer.scaling == 4.0


def test_gradient_checkpointing_is_non_reentrant() -> None:
    class FakeModel:
        def __init__(self) -> None:
            self.config = type("Config", (), {"use_cache": True})()
            self.kwargs = None

        def gradient_checkpointing_enable(self, *, gradient_checkpointing_kwargs):
            self.kwargs = gradient_checkpointing_kwargs

    model = FakeModel()
    enable_gradient_checkpointing(model)
    assert model.kwargs == {"use_reentrant": False}
    assert model.config.use_cache is False
