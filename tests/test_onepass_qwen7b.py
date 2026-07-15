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

from onepass_qwen7b.checkpoint import load_checkpoint, save_checkpoint
from onepass_qwen7b.data import OnePass7BDataset, grid_targets, prepare_batch
from onepass_qwen7b.model import OnePassQwen7B, SegMaskQueryBuilder, segmentation_loss
from onepass_stamp.lora import inject_lora


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 9
    unk_token_id = 99
    ids = {"<|seg|>": 5, "<|mask|>": 6, "<|image_pad|>": 7}

    def convert_tokens_to_ids(self, token):
        return self.ids.get(token, self.unk_token_id)


class FakeProcessor:
    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.image_processor = SimpleNamespace(merge_size=2)
        self.last_messages = None
        self.last_generation_prompt = None

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        del tokenize
        self.last_messages = messages
        self.last_generation_prompt = add_generation_prompt
        return "prepared"

    def __call__(self, text, images, padding, return_tensors):
        del text, images, padding, return_tensors
        input_ids = torch.tensor([[10, 7, 7, 11, 5]])
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "pixel_values": torch.zeros(8, 3),
            "image_grid_thw": torch.tensor([[1, 2, 4]]),
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
        return input_ids.eq(7).unsqueeze(-1).expand_as(inputs_embeds), None

    def forward(self, inputs_embeds, seg_mask, **kwargs):
        del kwargs
        self.last_seg_mask = seg_mask
        projected = self.q_proj(inputs_embeds)
        return SimpleNamespace(last_hidden_state=projected + projected.mean(dim=1, keepdim=True))


class FakeBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=4)
        self.model = FakeCore(4)
        self.classifier = nn.Linear(4, 1)


class OnePassQwen7BTests(unittest.TestCase):
    def test_query_builder_matches_mask_plus_visual_at_zero_position_init(self):
        builder = SegMaskQueryBuilder(4, max_grid_height=8, max_grid_width=8)
        visual = torch.randn(6, 4)
        with torch.no_grad():
            builder.mask_embedding.fill_(0.25)
        queries = builder.spatial_queries(visual, [(2, 3)], torch.float32)
        self.assertTrue(torch.allclose(queries, visual + 0.25))

    def test_dataset_batch_and_patch_count_alignment(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(root / "image.png")
            mask = np.zeros((4, 4), dtype=np.uint8)
            mask[:, :2] = 255
            Image.fromarray(mask).save(root / "mask.png")
            values = [
                {
                    "id": "sample",
                    "images": ["image.png"],
                    "masks": ["mask.png"],
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"text": 'Please segment the object "left half".'}],
                        }
                    ],
                }
            ]
            (root / "data.json").write_text(json.dumps(values), encoding="utf-8")
            dataset = OnePass7BDataset([root / "data.json"])
            sample = dataset[0]
            processor = FakeProcessor()
            prepared = prepare_batch([sample], processor, torch.device("cpu"))
            valid = prepared["input_ids"][0][prepared["attention_mask"][0].bool()]
            self.assertEqual(int(valid.eq(5).sum()), 1)
            self.assertEqual(int(valid.eq(6).sum()), 2)
            self.assertEqual(prepared["grid_shapes"], [(1, 2)])
            targets = grid_targets([sample], prepared["grid_shapes"], torch.device("cpu"))
            self.assertEqual(tuple(targets[0].shape), (2,))
            self.assertTrue(processor.last_generation_prompt)
            self.assertEqual(len(processor.last_messages), 1)

    def test_forward_backward_and_mask_attention_marker(self):
        backbone = FakeBackbone()
        inject_lora(backbone.model, rank=2, alpha=4.0, dropout=0.0, target_names=("q_proj",))
        model = OnePassQwen7B(backbone, seg_token_id=5, mask_token_id=6, merge_size=2)
        input_ids = torch.tensor([[7, 7, 5, 6, 6]])
        output = model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            pixel_values=torch.zeros(8, 3),
            image_grid_thw=torch.tensor([[1, 2, 4]]),
        )
        self.assertEqual(output.grid_shapes, [(1, 2)])
        loss, _ = segmentation_loss(output.mask_logits, [torch.tensor([1.0, 0.0])])
        loss.backward()
        self.assertIsNotNone(model.query_builder.seg_embedding.grad)
        self.assertIsNotNone(model.query_builder.mask_embedding.grad)
        self.assertIsNotNone(model.query_builder.row_embedding.weight.grad)
        self.assertIsNotNone(model.mask_classifier.weight.grad)
        self.assertTrue(torch.equal(backbone.model.last_seg_mask, input_ids.eq(6)))

    def test_seg_grounding_starts_as_output_preserving_and_receives_gradients(self):
        backbone = FakeBackbone()
        inject_lora(backbone.model, rank=2, alpha=4.0, dropout=0.0, target_names=("q_proj",))
        model = OnePassQwen7B(
            backbone,
            seg_token_id=5,
            mask_token_id=6,
            merge_size=2,
            use_seg_grounding=True,
            seg_grounding_size=2,
        )
        output = model(
            input_ids=torch.tensor([[7, 7, 5, 6, 6]]),
            attention_mask=torch.ones(1, 5, dtype=torch.long),
            pixel_values=torch.zeros(8, 3),
            image_grid_thw=torch.tensor([[1, 2, 4]]),
        )
        self.assertIsNotNone(output.seg_logits)
        self.assertTrue(torch.allclose(output.mask_logits[0], output.raw_mask_logits[0]))
        targets = [torch.tensor([1.0, 0.0])]
        mask_loss, _ = segmentation_loss(output.mask_logits, targets)
        seg_loss, _ = segmentation_loss(output.seg_logits, targets)
        (mask_loss + 0.1 * seg_loss).backward()
        self.assertIsNotNone(model.seg_grounding_head.seg_projection.weight.grad)
        self.assertIsNotNone(model.seg_grounding_head.mask_projection.weight.grad)
        self.assertIsNotNone(model.seg_grounding_head.fusion_weight.grad)
        self.assertIsNotNone(model.query_builder.seg_embedding.grad)

    def test_checkpoint_round_trip(self):
        backbone = FakeBackbone()
        inject_lora(backbone.model, rank=2, alpha=4.0, dropout=0.0, target_names=("q_proj",))
        model = OnePassQwen7B(backbone, seg_token_id=5, mask_token_id=6, merge_size=2)
        with torch.no_grad():
            model.query_builder.seg_embedding.fill_(0.2)
            model.mask_classifier.bias.fill_(0.4)
            model.lora_parameters()[0].fill_(0.1)
        with tempfile.TemporaryDirectory() as temporary:
            path = save_checkpoint(
                model,
                Path(temporary) / "model.pt",
                config={"base_model": "base"},
                epoch=1,
                batch_in_epoch=0,
                global_step=3,
            )
            restored_backbone = FakeBackbone()
            inject_lora(restored_backbone.model, rank=2, alpha=4.0, dropout=0.0, target_names=("q_proj",))
            restored = OnePassQwen7B(restored_backbone, seg_token_id=5, mask_token_id=6, merge_size=2)
            checkpoint = load_checkpoint(restored, path)
            self.assertEqual(checkpoint["global_step"], 3)
            self.assertTrue(torch.allclose(restored.query_builder.seg_embedding, model.query_builder.seg_embedding))
            self.assertTrue(torch.allclose(restored.mask_classifier.bias, model.mask_classifier.bias))
            self.assertTrue(torch.allclose(restored.lora_parameters()[0], model.lora_parameters()[0]))
            grounding_backbone = FakeBackbone()
            inject_lora(
                grounding_backbone.model,
                rank=2,
                alpha=4.0,
                dropout=0.0,
                target_names=("q_proj",),
            )
            grounding_model = OnePassQwen7B(
                grounding_backbone,
                seg_token_id=5,
                mask_token_id=6,
                merge_size=2,
                use_seg_grounding=True,
                seg_grounding_size=2,
            )
            load_checkpoint(grounding_model, path, allow_missing_seg_grounding=True)

    def test_seg_grounding_checkpoint_round_trip(self):
        backbone = FakeBackbone()
        inject_lora(backbone.model, rank=2, alpha=4.0, dropout=0.0, target_names=("q_proj",))
        model = OnePassQwen7B(
            backbone,
            seg_token_id=5,
            mask_token_id=6,
            merge_size=2,
            use_seg_grounding=True,
            seg_grounding_size=2,
        )
        with torch.no_grad():
            model.seg_grounding_head.fusion_weight.fill_(0.3)
            model.seg_grounding_head.seg_projection.weight.fill_(0.2)
        with tempfile.TemporaryDirectory() as temporary:
            path = save_checkpoint(
                model,
                Path(temporary) / "model.pt",
                config={"use_seg_grounding": True},
                epoch=1,
                batch_in_epoch=0,
                global_step=2,
            )
            restored_backbone = FakeBackbone()
            inject_lora(restored_backbone.model, rank=2, alpha=4.0, dropout=0.0, target_names=("q_proj",))
            restored = OnePassQwen7B(
                restored_backbone,
                seg_token_id=5,
                mask_token_id=6,
                merge_size=2,
                use_seg_grounding=True,
                seg_grounding_size=2,
            )
            load_checkpoint(restored, path)
            self.assertTrue(
                torch.allclose(
                    restored.seg_grounding_head.seg_projection.weight,
                    model.seg_grounding_head.seg_projection.weight,
                )
            )
            self.assertAlmostEqual(restored.seg_fusion_alpha(), model.seg_fusion_alpha(), places=6)


if __name__ == "__main__":
    unittest.main()
