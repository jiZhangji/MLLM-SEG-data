from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch

from onepass_stamp.lora import DEFAULT_TARGETS, inject_lora

from .data import MASK_TOKEN, SEG_TOKEN
from .model import OnePassQwen7B


def add_stamp_code_to_path(stamp_code_dir: str | Path) -> Path:
    path = Path(stamp_code_dir).resolve()
    if not (path / "model" / "qwen_changes.py").exists():
        raise FileNotFoundError(f"Not a STAMP source directory: {path}")
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    return path


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    values = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    if name not in values:
        raise ValueError(f"Unsupported dtype {name!r}.")
    if device.type == "cpu":
        return torch.float32
    return values[name]


def configure_cudnn(requested_disable: bool | None, device: torch.device) -> bool:
    if device.type != "cuda":
        return False
    if requested_disable is True:
        torch.backends.cudnn.enabled = False
        return True
    if requested_disable is False:
        torch.backends.cudnn.enabled = True
        return False
    try:
        torch.backends.cudnn.enabled = True
        layer = torch.nn.Conv3d(3, 8, kernel_size=(2, 14, 14), stride=(2, 14, 14), bias=False).to(
            device=device, dtype=torch.bfloat16
        )
        inputs = torch.randn(1, 3, 2, 28, 28, device=device, dtype=torch.bfloat16)
        layer(inputs)
        torch.cuda.synchronize(device)
        return False
    except RuntimeError as error:
        torch.backends.cudnn.enabled = False
        print(f"[WARN] cuDNN Conv3D probe failed; using native CUDA kernels: {error}", flush=True)
        return True


def semantic_average(tokenizer: Any, embeddings: torch.Tensor, text: str, old_vocab_size: int) -> torch.Tensor:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    token_ids = [token_id for token_id in token_ids if 0 <= token_id < old_vocab_size]
    if not token_ids:
        return embeddings[:old_vocab_size].mean(dim=0)
    indices = torch.tensor(token_ids, device=embeddings.device)
    return embeddings[indices].mean(dim=0)


def reset_classifier(classifier: torch.nn.Module, initializer_range: float) -> None:
    with torch.no_grad():
        torch.nn.init.normal_(classifier.weight, mean=0.0, std=float(initializer_range))
        if classifier.bias is not None:
            torch.nn.init.zeros_(classifier.bias)


def enable_gradient_checkpointing(model: torch.nn.Module) -> None:
    """Enable DDP-safe checkpointing for trainable adapters on a frozen base."""
    try:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    except TypeError as error:
        raise RuntimeError(
            "This Transformers version cannot request non-reentrant gradient checkpointing. "
            "Upgrade Transformers or train with --no-gradient-checkpointing."
        ) from error
    model.config.use_cache = False


def load_base_onepass_model(
    *,
    stamp_code_dir: str | Path,
    base_model: str | Path,
    device: str | torch.device,
    dtype: str = "bf16",
    attn_implementation: str = "sdpa",
    min_pixels: int = 802816,
    max_pixels: int = 1003520,
    max_grid_height: int = 512,
    max_grid_width: int = 512,
    use_lora: bool = True,
    lora_rank: int = 16,
    lora_alpha: float = 32.0,
    lora_dropout: float = 0.05,
    use_rslora: bool = False,
    train_visual_projection: bool = True,
    train_position_embeddings: bool = True,
    train_classifier: bool = True,
    use_seg_grounding: bool = False,
    seg_grounding_size: int = 256,
    seg_grounding_temperature: float = 0.1,
    use_seg_fusion: bool = True,
    allow_existing_segmentation_tokens: bool = False,
    gradient_checkpointing: bool = False,
) -> tuple[OnePassQwen7B, Any]:
    add_stamp_code_to_path(stamp_code_dir)
    from transformers import AutoProcessor
    from model.qwen_changes import SegQwenVL, _create_hybrid_mask_and_dependencies, get_rope_index

    device = torch.device(device)
    torch_dtype = resolve_dtype(dtype, device)
    model_name = str(base_model)
    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=False,
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
            f"Base tokenizer already contains {existing}. For a fair base-model experiment use an untrained "
            "Qwen2-VL checkpoint, or explicitly pass --allow-existing-segmentation-tokens."
        )

    backbone = SegQwenVL.from_pretrained(
        model_name,
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
    seg_token_id = tokenizer.convert_tokens_to_ids(SEG_TOKEN)
    mask_token_id = tokenizer.convert_tokens_to_ids(MASK_TOKEN)
    backbone.mask_token_id = int(mask_token_id)
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

    model = OnePassQwen7B(
        backbone=backbone,
        seg_token_id=int(seg_token_id),
        mask_token_id=int(mask_token_id),
        merge_size=int(processor.image_processor.merge_size),
        max_grid_height=int(max_grid_height),
        max_grid_width=int(max_grid_width),
        train_visual_projection=train_visual_projection,
        train_position_embeddings=train_position_embeddings,
        train_classifier=train_classifier,
        use_seg_grounding=use_seg_grounding,
        seg_grounding_size=seg_grounding_size,
        seg_grounding_temperature=seg_grounding_temperature,
        use_seg_fusion=use_seg_fusion,
    )
    embeddings = backbone.get_input_embeddings().weight.detach()
    seg_vector = semantic_average(tokenizer, embeddings, "segment target", old_vocab_size)
    mask_vector = semantic_average(tokenizer, embeddings, "mask region", old_vocab_size)
    model.query_builder.initialize_semantic_embeddings(seg_vector, mask_vector)
    model.to(device)
    return model, processor
