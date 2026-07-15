from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from onepass_stamp.lora import load_lora_state_dict, lora_state_dict

from .model import OnePassQwen7B


FORMAT_VERSION = 1


def save_checkpoint(
    model: OnePassQwen7B,
    path: str | Path,
    *,
    config: dict[str, Any],
    epoch: int,
    batch_in_epoch: int,
    global_step: int,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "query_state_dict": model.query_builder.state_dict(),
        "classifier_state_dict": model.mask_classifier.state_dict(),
        "lora_state_dict": lora_state_dict(model.backbone),
        "config": config,
        "epoch": int(epoch),
        "batch_in_epoch": int(batch_in_epoch),
        "global_step": int(global_step),
    }
    if model.seg_grounding_head is not None:
        payload["seg_grounding_state_dict"] = model.seg_grounding_head.state_dict()
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return path


def load_checkpoint(
    model: OnePassQwen7B,
    path: str | Path,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    allow_missing_seg_grounding: bool = False,
) -> dict[str, Any]:
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    if int(checkpoint.get("format_version", 0)) != FORMAT_VERSION:
        raise ValueError(f"Unsupported checkpoint format in {path}.")
    model.query_builder.load_state_dict(checkpoint["query_state_dict"], strict=True)
    model.mask_classifier.load_state_dict(checkpoint["classifier_state_dict"], strict=True)
    saved_lora = checkpoint.get("lora_state_dict", {})
    if saved_lora or model.lora_parameters():
        load_lora_state_dict(model.backbone, saved_lora)
    saved_grounding = checkpoint.get("seg_grounding_state_dict")
    if model.seg_grounding_head is not None:
        if saved_grounding is None:
            if not allow_missing_seg_grounding:
                raise ValueError(f"Checkpoint {path} has no SEG grounding head state.")
        else:
            model.seg_grounding_head.load_state_dict(saved_grounding, strict=True)
    elif saved_grounding is not None:
        raise ValueError("Checkpoint contains a SEG grounding head, but the model was built without it.")
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint


def read_checkpoint_config(path: str | Path) -> dict[str, Any]:
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    return dict(checkpoint.get("config", {}))
