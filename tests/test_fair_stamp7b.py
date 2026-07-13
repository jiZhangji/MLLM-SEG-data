from __future__ import annotations

import torch
import torch.nn as nn
from types import SimpleNamespace

from fair_stamp7b.data import _last_subsequence
from fair_stamp7b.eval import build_classification_batch
from fair_stamp7b.model import SpecialTokenInputAdapter, SpecialTokenOutputAdapter
from onepass_stamp.lora import OnePassLoRALinear
from onepass_qwen7b.runtime import enable_gradient_checkpointing
from onepass_qwen7b.model import SegMaskQueryBuilder


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


def test_onepass_query_table_supports_wide_dynamic_grid() -> None:
    builder = SegMaskQueryBuilder(hidden_size=4, max_grid_height=512, max_grid_width=512)
    queries = builder.spatial_queries(
        torch.randn(13 * 80, 4), [(13, 80)], output_dtype=torch.float32
    )
    assert queries.shape == (1040, 4)


def test_stamp_eval_builds_queries_after_generated_seg() -> None:
    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 9

        def convert_tokens_to_ids(self, token):
            return {"<|seg|>": 5, "<|mask|>": 6}[token]

        def decode(self, values, skip_special_tokens=False):
            del skip_special_tokens
            return " ".join(str(int(value)) for value in values)

    processor = SimpleNamespace(
        tokenizer=Tokenizer(), image_processor=SimpleNamespace(merge_size=2)
    )
    encoded = {
        "input_ids": torch.tensor([[0, 10, 11], [20, 21, 22]]),
        "attention_mask": torch.tensor([[0, 1, 1], [1, 1, 1]]),
        "image_grid_thw": torch.tensor([[1, 4, 6], [1, 8, 4]]),
    }
    generated = torch.tensor([[0, 10, 11, 30, 5, 0], [20, 21, 22, 31, 32, 0]])
    ids, attention, grids, found, _ = build_classification_batch(
        encoded, generated, processor
    )
    assert grids == [(2, 3), (4, 2)]
    assert found == [True, False]
    assert int((ids[0] == 6).sum()) == 6
    assert int((ids[1] == 6).sum()) == 8
    assert int(attention[0].sum()) == 10
    assert int(attention[1].sum()) == 12
