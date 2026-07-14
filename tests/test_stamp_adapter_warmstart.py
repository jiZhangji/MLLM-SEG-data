from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import save_file

from onepass_qwen7b.stamp_adapter import (
    load_stamp_adapter_initialization,
    read_stamp_adapter_config,
)
from onepass_stamp.lora import DEFAULT_TARGETS, OnePassLoRALinear, inject_lora


class FakeBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(4, 4, bias=False)


class FakeLanguageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([FakeBlock()])


class FakeCore(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.language_model = FakeLanguageModel()


class FakeBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = FakeCore()
        self.classifier = nn.Linear(4, 1)


class FakeOnePass(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = FakeBackbone()
        inject_lora(
            self.backbone.model.language_model,
            rank=2,
            alpha=4.0,
            dropout=0.0,
            target_names=("q_proj",),
            use_rslora=True,
        )

    @property
    def mask_classifier(self) -> nn.Module:
        return self.backbone.classifier


class StampAdapterWarmStartTests(unittest.TestCase):
    def _adapter(self, root: Path) -> Path:
        adapter = root / "adapter"
        adapter.mkdir()
        config = {
            "peft_type": "LORA",
            "r": 2,
            "lora_alpha": 4,
            "lora_dropout": 0.0,
            "use_rslora": True,
            "target_modules": list(DEFAULT_TARGETS),
        }
        (adapter / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")
        save_file(
            {
                "base_model.model.model.language_model.layers.0.q_proj.lora_A.weight": torch.full((2, 4), 0.25),
                "base_model.model.model.language_model.layers.0.q_proj.lora_B.weight": torch.full((4, 2), 0.5),
                "base_model.model.classifier.modules_to_save.default.weight": torch.full((1, 4), 0.75),
                "base_model.model.classifier.modules_to_save.default.bias": torch.full((1,), 0.125),
            },
            adapter / "adapter_model.safetensors",
        )
        return adapter

    def test_reads_adapter_hyperparameters(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = read_stamp_adapter_config(self._adapter(Path(temporary)))
            self.assertEqual(config.rank, 2)
            self.assertEqual(config.alpha, 4.0)
            self.assertTrue(config.use_rslora)

    def test_loads_peft_lora_and_classifier_into_custom_model(self):
        model = FakeOnePass()
        with tempfile.TemporaryDirectory() as temporary:
            report = load_stamp_adapter_initialization(model, self._adapter(Path(temporary)))
        layer = model.backbone.model.language_model.layers[0].q_proj
        self.assertIsInstance(layer, OnePassLoRALinear)
        self.assertTrue(torch.allclose(layer.lora_a.weight, torch.full((2, 4), 0.25)))
        self.assertTrue(torch.allclose(layer.lora_b.weight, torch.full((4, 2), 0.5)))
        self.assertTrue(torch.allclose(model.mask_classifier.weight, torch.full((1, 4), 0.75)))
        self.assertTrue(torch.allclose(model.mask_classifier.bias, torch.full((1,), 0.125)))
        self.assertEqual(report["lora_layers_loaded"], 1)
        self.assertTrue(report["classifier_loaded"])


if __name__ == "__main__":
    unittest.main()
