from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from onepass_qwen7b.data import MASK_TOKEN, SEG_TOKEN
from onepass_qwen7b.model import hidden_size
from onepass_qwen7b.runtime import (
    add_stamp_code_to_path,
    configure_cudnn,
    enable_gradient_checkpointing,
    reset_classifier,
    resolve_dtype,
    semantic_average,
)
from onepass_stamp.lora import DEFAULT_TARGETS, inject_lora

from .model import FairSTAMP7B, SpecialTokenInputAdapter, SpecialTokenOutputAdapter


def load_fair_stamp_model(
    *,
    stamp_code_dir: str | Path,
    base_model: str | Path,
    device: str | torch.device,
    dtype: str = "bf16",
    attn_implementation: str = "sdpa",
    min_pixels: int = 802816,
    max_pixels: int = 1003520,
    use_lora: bool = True,
    lora_rank: int = 64,
    lora_alpha: float = 128.0,
    lora_dropout: float = 0.05,
    use_rslora: bool = True,
    allow_existing_segmentation_tokens: bool = False,
    gradient_checkpointing: bool = True,
) -> tuple[FairSTAMP7B, Any]:
    add_stamp_code_to_path(stamp_code_dir)
    from transformers import AutoProcessor
    from model.qwen_changes import SegQwenVL, _create_hybrid_mask_and_dependencies, get_rope_index

    device = torch.device(device)
    torch_dtype = resolve_dtype(dtype, device)
    processor = AutoProcessor.from_pretrained(
        str(base_model),
        trust_remote_code=True,
        min_pixels=int(min_pixels),
        max_pixels=int(max_pixels),
    )
    tokenizer = processor.tokenizer
    old_vocab_size = len(tokenizer)
    existing = []
    for token in (SEG_TOKEN, MASK_TOKEN):
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is not None and token_id >= 0 and token_id != tokenizer.unk_token_id:
            existing.append(token)
    if existing and not allow_existing_segmentation_tokens:
        raise ValueError(
            f"Base tokenizer already contains {existing}. Use plain Qwen2-VL-7B-Instruct for the fair run."
        )
    backbone = SegQwenVL.from_pretrained(
        str(base_model),
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        attn_implementation=attn_implementation,
        trust_remote_code=True,
    )
    tokenizer.add_special_tokens({"additional_special_tokens": [SEG_TOKEN, MASK_TOKEN]})
    backbone.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        backbone.config.pad_token_id = tokenizer.pad_token_id
    backbone.model.get_rope_index = get_rope_index.__get__(backbone.model, type(backbone.model))
    backbone.model._create_hybrid_mask_and_dependencies = _create_hybrid_mask_and_dependencies.__get__(
        backbone.model, type(backbone.model)
    )
    seg_id = int(tokenizer.convert_tokens_to_ids(SEG_TOKEN))
    mask_id = int(tokenizer.convert_tokens_to_ids(MASK_TOKEN))
    backbone.mask_token_id = mask_id
    embeddings = backbone.get_input_embeddings().weight
    seg_vector = semantic_average(tokenizer, embeddings.detach(), "segment target", old_vocab_size)
    mask_vector = semantic_average(tokenizer, embeddings.detach(), "mask region", old_vocab_size)
    with torch.no_grad():
        embeddings[seg_id].copy_(seg_vector.to(embeddings))
        embeddings[mask_id].copy_(mask_vector.to(embeddings))
        output_weight = backbone.get_output_embeddings().weight
        if output_weight.data_ptr() != embeddings.data_ptr():
            output_weight[seg_id].copy_(seg_vector.to(output_weight))
            output_weight[mask_id].copy_(mask_vector.to(output_weight))
    input_adapter = SpecialTokenInputAdapter(backbone.get_input_embeddings(), (seg_id, mask_id))
    output_adapter = SpecialTokenOutputAdapter(
        backbone.get_output_embeddings(), (seg_id, mask_id), hidden_size(backbone.config)
    )
    backbone.set_input_embeddings(input_adapter)
    backbone.set_output_embeddings(output_adapter)
    reset_classifier(backbone.classifier, getattr(backbone.config, "initializer_range", 0.02))
    if use_lora:
        inject_lora(
            backbone.model.language_model,
            rank=int(lora_rank),
            alpha=float(lora_alpha),
            dropout=float(lora_dropout),
            target_names=DEFAULT_TARGETS,
            use_rslora=use_rslora,
        )
    if gradient_checkpointing:
        enable_gradient_checkpointing(backbone)
    model = FairSTAMP7B(backbone, input_adapter, output_adapter, mask_token_id=mask_id)
    model.to(device)
    return model, processor


__all__ = ["configure_cudnn", "load_fair_stamp_model"]
