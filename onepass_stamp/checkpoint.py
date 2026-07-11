from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .model import OnePassSTAMP
from .lora import load_lora_state_dict, lora_state_dict


FORMAT_VERSION = 1


def save_checkpoint(
    model: OnePassSTAMP,
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
        "query_state_dict": model.query_module.state_dict(),
        "classifier_state_dict": model.mask_classifier.state_dict(),
        "lora_state_dict": lora_state_dict(model.backbone),
        "config": config,
        "epoch": int(epoch),
        "batch_in_epoch": int(batch_in_epoch),
        "global_step": int(global_step),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return path


def load_checkpoint(
    model: OnePassSTAMP,
    path: str | Path,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
) -> dict[str, Any]:
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    version = int(checkpoint.get("format_version", 0))
    if version != FORMAT_VERSION:
        raise ValueError(f"Unsupported OnePass checkpoint version {version}; expected {FORMAT_VERSION}.")
    model.query_module.load_state_dict(checkpoint["query_state_dict"], strict=True)
    model.mask_classifier.load_state_dict(checkpoint["classifier_state_dict"], strict=True)
    saved_lora = checkpoint.get("lora_state_dict", {})
    if saved_lora or model.lora_parameters():
        load_lora_state_dict(model.backbone, saved_lora)
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint


def read_checkpoint_config(path: str | Path) -> dict[str, Any]:
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    return dict(checkpoint.get("config", {}))
