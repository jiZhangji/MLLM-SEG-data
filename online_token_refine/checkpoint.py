from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from transformers import TrainerCallback

from .model import UncertaintyAwareMaskHead


def _unwrap_model(model: Any) -> Any:
    while hasattr(model, "module"):
        model = model.module
    return model


def save_uncertainty_head(model: Any, path: str | Path, step: int | None = None) -> Path:
    model = _unwrap_model(model)
    if not hasattr(model, "uncertainty_mask_head"):
        raise AttributeError("Model does not have uncertainty_mask_head.")
    head = model.uncertainty_mask_head
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 2,
        "model_state_dict": head.state_dict(),
        "token_dim": int(head.token_dim),
        "hidden_size": int(head.hidden_size),
        "use_uncertainty_gate": bool(head.use_uncertainty_gate),
        "global_step": step,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return path


def load_uncertainty_head(model: Any, path: str | Path) -> dict[str, Any]:
    model = _unwrap_model(model)
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.uncertainty_mask_head.load_state_dict(state_dict, strict=True)
    return checkpoint


def attach_uncertainty_head_from_checkpoint(model: Any, path: str | Path) -> dict[str, Any]:
    """Construct and attach the dynamic head before loading its state."""
    model = _unwrap_model(model)
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    token_dim = int(checkpoint.get("token_dim", model.config.hidden_size))
    head = UncertaintyAwareMaskHead(
        token_dim=token_dim,
        hidden_size=int(checkpoint.get("hidden_size", 128)),
        use_uncertainty_gate=bool(checkpoint.get("use_uncertainty_gate", True)),
    )
    head.initialize_from_classifier(model.classifier)
    head.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=True)
    device = next(model.parameters()).device
    model.uncertainty_mask_head = head.to(device=device)
    return checkpoint


class UncertaintyHeadCheckpointCallback(TrainerCallback):
    """Save only the small dynamic head instead of a full 2B checkpoint."""

    def __init__(self, save_steps: int) -> None:
        self.save_steps = max(0, int(save_steps))

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if (
            self.save_steps > 0
            and state.is_world_process_zero
            and state.global_step > 0
            and state.global_step % self.save_steps == 0
        ):
            path = Path(args.output_dir) / f"uncertainty_mask_head_step_{state.global_step}.pt"
            save_uncertainty_head(model, path, step=state.global_step)
        return control

    def on_train_end(self, args, state, control, model=None, **kwargs):
        if state.is_world_process_zero:
            path = Path(args.output_dir) / "uncertainty_mask_head.pt"
            save_uncertainty_head(model, path, step=state.global_step)
        return control
