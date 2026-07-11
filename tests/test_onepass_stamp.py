from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from onepass_stamp.checkpoint import load_checkpoint, save_checkpoint
from onepass_stamp.data import OnePassDataset, OnePassSample, prepare_onepass_batch
from onepass_stamp.lora import OnePassLoRALinear, inject_lora, lora_parameters
from onepass_stamp.model import OnePassQueryModule, OnePassSTAMP, onepass_mask_loss
from onepass_stamp.runtime import configure_cudnn


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 9
    unk_token_id = 99

    ids = {
        "<|seg|>": 5,
        "<|mask|>": 6,
        "<|image_pad|>": 7,
        "<|task_seg|>": 8,
    }

    def convert_tokens_to_ids(self, token):
        return self.ids.get(token, self.unk_token_id)


class FakeProcessor:
    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.image_processor = SimpleNamespace(merge_size=2)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        del messages, tokenize, add_generation_prompt
        return "prepared"

    def __call__(self, text, images, padding, return_tensors):
        del images, padding, return_tensors
        batch = len(text)
        ids = torch.tensor([[10, 7, 7, 8, 11, 5, 9]] * batch)
        return {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "pixel_values": torch.zeros(batch * 8, 3),
            "image_grid_thw": torch.tensor([[1, 2, 4]] * batch),
        }


class FakeCore(nn.Module):
    def __init__(self, hidden_size=4):
        super().__init__()
        self.embedding = nn.Embedding(32, hidden_size)
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.last_seg_mask = None

    def get_input_embeddings(self):
        return self.embedding

    def get_image_features(self, pixel_values, image_grid_thw):
        del pixel_values
        return [torch.ones(2, self.embedding.embedding_dim) for _ in image_grid_thw]

    def get_placeholder_mask(self, input_ids, inputs_embeds, image_features):
        del image_features
        mask = input_ids.eq(7).unsqueeze(-1).expand_as(inputs_embeds)
        return mask, None

    def forward(self, inputs_embeds, seg_mask, **kwargs):
        del kwargs
        self.last_seg_mask = seg_mask
        projected = self.q_proj(inputs_embeds)
        contextual = projected + projected.mean(dim=1, keepdim=True)
        return SimpleNamespace(last_hidden_state=contextual)


class FakeBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=4)
        self.model = FakeCore(4)
        self.classifier = nn.Linear(4, 1)


class OnePassTests(unittest.TestCase):
    def test_cudnn_probe_is_noop_on_cpu(self):
        self.assertFalse(configure_cudnn(None, torch.device("cpu")))

    def test_lora_injection_preserves_initial_output(self):
        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(4, 4)

            def forward(self, value):
                return self.q_proj(value)

        block = Block()
        value = torch.randn(2, 4)
        expected = block(value).detach()
        self.assertEqual(inject_lora(block, rank=2, alpha=4.0, dropout=0.0, target_names=("q_proj",)), 1)
        self.assertIsInstance(block.q_proj, OnePassLoRALinear)
        self.assertTrue(torch.allclose(block(value), expected))
        self.assertEqual(len(lora_parameters(block)), 2)

    def test_query_module_starts_from_stamp_visual_behavior(self):
        module = OnePassQueryModule(4)
        visual = torch.randn(6, 4)
        output = module.spatial_queries(visual, [(2, 3)], torch.float32)
        self.assertTrue(torch.allclose(output, visual))

    def test_forward_and_backward(self):
        model = OnePassSTAMP(FakeBackbone(), seg_token_id=5, mask_token_id=6, task_token_id=8, merge_size=2)
        input_ids = torch.tensor([[7, 7, 8, 5, 6, 6]])
        output = model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            pixel_values=torch.zeros(8, 3),
            image_grid_thw=torch.tensor([[1, 2, 4]]),
        )
        self.assertEqual(output.grid_shapes, [(1, 2)])
        self.assertEqual(tuple(output.mask_logits[0].shape), (2,))
        loss, _ = onepass_mask_loss(output.mask_logits, [torch.tensor([1.0, 0.0])])
        loss.backward()
        self.assertIsNotNone(model.query_module.visual_projection.weight.grad)
        self.assertIsNotNone(model.query_module.seg_delta.grad)
        self.assertIsNotNone(model.mask_classifier.weight.grad)
        self.assertTrue(torch.equal(model.backbone.model.last_seg_mask, input_ids.eq(6)))

    def test_dataset_and_batch_construction(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(root / "image.png")
            mask = np.zeros((4, 4), dtype=np.uint8)
            mask[:, :2] = 255
            Image.fromarray(mask).save(root / "mask.png")
            data = [
                {
                    "id": "sample",
                    "images": ["image.png"],
                    "masks": ["mask.png"],
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "text": None},
                                {"type": "text", "text": 'Please segment "left half".'},
                            ],
                        },
                        {"role": "assistant", "content": [{"type": "text", "text": "Target <|seg|>."}]},
                    ],
                }
            ]
            (root / "data.json").write_text(json.dumps(data), encoding="utf-8")
            dataset = OnePassDataset(root / "data.json")
            sample: OnePassSample = dataset[0]
            prepared = prepare_onepass_batch([sample], FakeProcessor(), torch.device("cpu"), True)
            valid_ids = prepared["input_ids"][0][prepared["attention_mask"][0].bool()]
            self.assertEqual(int(valid_ids.eq(5).sum()), 1)
            self.assertEqual(int(valid_ids.eq(6).sum()), 2)
            self.assertEqual(prepared["grid_shapes"], [(1, 2)])

    def test_checkpoint_round_trip(self):
        backbone = FakeBackbone()
        inject_lora(backbone.model, rank=2, alpha=4.0, dropout=0.0, target_names=("q_proj",))
        model = OnePassSTAMP(backbone, seg_token_id=5, mask_token_id=6, task_token_id=8, merge_size=2)
        with torch.no_grad():
            model.query_module.seg_delta.fill_(0.25)
            model.mask_classifier.bias.fill_(0.5)
            model.lora_parameters()[0].fill_(0.125)
        with tempfile.TemporaryDirectory() as temporary:
            path = save_checkpoint(
                model,
                Path(temporary) / "onepass.pt",
                config={"use_task_token": True},
                epoch=1,
                batch_in_epoch=0,
                global_step=7,
            )
            restored_backbone = FakeBackbone()
            inject_lora(restored_backbone.model, rank=2, alpha=4.0, dropout=0.0, target_names=("q_proj",))
            restored = OnePassSTAMP(restored_backbone, seg_token_id=5, mask_token_id=6, task_token_id=8, merge_size=2)
            checkpoint = load_checkpoint(restored, path)
            self.assertEqual(checkpoint["global_step"], 7)
            self.assertTrue(torch.allclose(restored.query_module.seg_delta, model.query_module.seg_delta))
            self.assertTrue(torch.allclose(restored.mask_classifier.bias, model.mask_classifier.bias))
            self.assertTrue(torch.allclose(restored.lora_parameters()[0], model.lora_parameters()[0]))


if __name__ == "__main__":
    unittest.main()
