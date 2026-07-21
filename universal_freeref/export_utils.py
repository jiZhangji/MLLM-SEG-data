from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def artifact_paths(output_dir: Path, index: int) -> dict[str, Path]:
    sample_id = f"{index:08d}"
    return {
        "logits": output_dir / "pred_logits" / f"{sample_id}.npz",
        "mask": output_dir / "pred_masks" / f"{sample_id}.png",
        "gt": output_dir / "gt_masks" / f"{sample_id}.png",
    }


def artifacts_complete(paths: dict[str, Path], *, require_logits: bool = True) -> bool:
    required = ("mask", "gt", "logits") if require_logits else ("mask", "gt")
    return all(paths[name].is_file() and paths[name].stat().st_size > 0 for name in required)


def atomic_save_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    Image.fromarray(np.asarray(mask, dtype=bool).astype(np.uint8) * 255, mode="L").save(
        temporary, format="PNG"
    )
    os.replace(temporary, path)


def atomic_save_logits(path: Path, logits: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, logits=np.asarray(logits, dtype=np.float16))
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=True), encoding="utf-8")
    os.replace(temporary, path)


def atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def merge_binary_logits(logits: np.ndarray, empty_logit: float = -20.0) -> np.ndarray:
    """Merge multiple foreground logits as a probabilistic union."""
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim == 2:
        return values.astype(np.float32)
    if values.ndim != 3:
        raise ValueError(f"Expected [N,H,W] or [H,W] logits, got {values.shape}.")
    if values.shape[0] == 0:
        raise ValueError("An empty logit stack needs an explicit spatial shape.")
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(values, -60.0, 60.0)))
    union_probability = 1.0 - np.prod(1.0 - probabilities, axis=0)
    epsilon = 1e-6
    union_probability = np.clip(union_probability, epsilon, 1.0 - epsilon)
    merged = np.log(union_probability) - np.log1p(-union_probability)
    merged[~np.isfinite(merged)] = empty_logit
    return merged.astype(np.float32)
