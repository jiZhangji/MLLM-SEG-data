from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch

from .data import MASK_TOKEN, SEG_TOKEN, TASK_TOKEN
from .lora import DEFAULT_TARGETS, inject_lora
from .model import OnePassSTAMP


def add_stamp_to_path(stamp_code_dir: str | Path) -> Path:
    path = Path(stamp_code_dir).resolve()
    if not (path / "model" / "qwen_changes.py").exists():
        raise FileNotFoundError(f"Not a STAMP source directory: {path}")
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    return path


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    normalized = name.lower()
    if normalized == "auto":
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    values = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if normalized not in values:
        raise ValueError(f"Unsupported dtype {name!r}.")
    if device.type == "cpu" and values[normalized] != torch.float32:
        return torch.float32
    return values[normalized]


def _initialize_added_task_token(backbone: Any, tokenizer: Any, task_token_id: int) -> None:
    source_ids = tokenizer.encode("segmentation task", add_special_tokens=False)
    source_ids = [token_id for token_id in source_ids if 0 <= token_id < task_token_id]
    if not source_ids:
        return
    embeddings = backbone.get_input_embeddings().weight
    with torch.no_grad():
        source = embeddings[torch.tensor(source_ids, device=embeddings.device)].mean(dim=0)
        embeddings[task_token_id].copy_(source)
        output_embeddings = backbone.get_output_embeddings()
        if output_embeddings is not None and output_embeddings.weight.shape[0] > task_token_id:
            output_embeddings.weight[task_token_id].copy_(source)


def load_onepass_model(
    *,
    stamp_code_dir: str | Path,
    model_name: str | Path,
    device: str | torch.device,
    dtype: str = "bf16",
    attn_implementation: str = "sdpa",
    min_pixels: int = 802816,
    max_pixels: int = 1003520,
    use_task_token: bool = True,
    use_lora: bool = True,
    lora_rank: int = 8,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.05,
) -> tuple[OnePassSTAMP, Any]:
    add_stamp_to_path(stamp_code_dir)
    from transformers import AutoProcessor
    from model.qwen_changes import SegQwenVL, _create_hybrid_mask_and_dependencies, get_rope_index

    device = torch.device(device)
    torch_dtype = resolve_dtype(dtype, device)
    model_name = str(model_name)
    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True,
        min_pixels=int(min_pixels),
        max_pixels=int(max_pixels),
    )
    backbone = SegQwenVL.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        attn_implementation=attn_implementation,
        trust_remote_code=True,
    )

    missing_required = []
    for token in (SEG_TOKEN, MASK_TOKEN):
        token_id = processor.tokenizer.convert_tokens_to_ids(token)
        if token_id is None or token_id < 0 or token_id == processor.tokenizer.unk_token_id:
            missing_required.append(token)
    if missing_required:
        raise ValueError(
            f"Model tokenizer is missing STAMP tokens {missing_required}. "
            "Use a trained STAMP checkpoint rather than a base Qwen checkpoint."
        )

    special_tokens = [TASK_TOKEN] if use_task_token else []
    old_size = len(processor.tokenizer)
    if special_tokens:
        processor.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    if len(processor.tokenizer) != backbone.get_input_embeddings().weight.shape[0]:
        backbone.resize_token_embeddings(len(processor.tokenizer))
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
        backbone.config.pad_token_id = processor.tokenizer.pad_token_id

    task_token_id = None
    if use_task_token:
        task_token_id = processor.tokenizer.convert_tokens_to_ids(TASK_TOKEN)
        if len(processor.tokenizer) > old_size:
            _initialize_added_task_token(backbone, processor.tokenizer, int(task_token_id))

    # Bind the official multimodal position function to the actual Qwen model
    # instance. This avoids relying on global monkey-patch order.
    backbone.model.get_rope_index = get_rope_index.__get__(backbone.model, type(backbone.model))
    backbone.model._create_hybrid_mask_and_dependencies = _create_hybrid_mask_and_dependencies.__get__(
        backbone.model, type(backbone.model)
    )
    seg_token_id = processor.tokenizer.convert_tokens_to_ids(SEG_TOKEN)
    mask_token_id = processor.tokenizer.convert_tokens_to_ids(MASK_TOKEN)
    backbone.mask_token_id = int(mask_token_id)
    if use_lora:
        inject_lora(
            backbone.model.language_model,
            rank=int(lora_rank),
            alpha=float(lora_alpha),
            dropout=float(lora_dropout),
            target_names=DEFAULT_TARGETS,
        )
    model = OnePassSTAMP(
        backbone=backbone,
        seg_token_id=int(seg_token_id),
        mask_token_id=int(mask_token_id),
        task_token_id=None if task_token_id is None else int(task_token_id),
        merge_size=int(processor.image_processor.merge_size),
    )
    model.to(device)
    return model, processor
