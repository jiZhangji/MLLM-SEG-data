from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from onepass_stamp.lora import load_lora_state_dict, lora_state_dict

from .model import FairSTAMP7B


FORMAT_VERSION = 1


def save_checkpoint(
    model: FairSTAMP7B,
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
        "token_state_dict": model.token_state_dict(),
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
    model: FairSTAMP7B,
    path: str | Path,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
) -> dict[str, Any]:
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    if int(checkpoint.get("format_version", 0)) != FORMAT_VERSION:
        raise ValueError(f"Unsupported fair STAMP checkpoint: {path}")
    model.load_token_state_dict(checkpoint["token_state_dict"])
    model.mask_classifier.load_state_dict(checkpoint["classifier_state_dict"], strict=True)
    load_lora_state_dict(model.backbone, checkpoint.get("lora_state_dict", {}))
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint

