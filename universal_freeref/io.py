from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from .schema import ManifestItem


IMAGE_SUFFIXES = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def _select_mapping_value(value: dict[str, Any], key: str | None) -> Any:
    if key is not None:
        if key not in value:
            raise KeyError(f"Array key {key!r} is absent; available keys: {sorted(value)}")
        return value[key]
    for candidate in ("probability", "prob", "logits", "mask", "prediction", "pred"):
        if candidate in value:
            return value[candidate]
    if len(value) == 1:
        return next(iter(value.values()))
    raise KeyError(f"Cannot select an array from keys: {sorted(value)}; set array_key in the manifest.")


def load_array(path: Path, key: str | None = None) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return np.asarray(Image.open(path).convert("L"))
    if suffix == ".npy":
        return np.asarray(np.load(path, allow_pickle=False))
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            values = {name: archive[name] for name in archive.files}
        return np.asarray(_select_mapping_value(values, key))
    if suffix in {".pt", ".pth"}:
        value = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(value, dict):
            value = _select_mapping_value(value, key)
        if isinstance(value, torch.Tensor):
            value = value.detach().float().cpu().numpy()
        return np.asarray(value)
    raise ValueError(f"Unsupported prediction format {suffix!r}: {path}")


def _spatial_array(values: np.ndarray, foreground_channel: int, kind: str) -> np.ndarray:
    values = np.asarray(values)
    while values.ndim > 2 and 1 in values.shape:
        axis = next(index for index, size in enumerate(values.shape) if size == 1)
        values = np.squeeze(values, axis=axis)
    if values.ndim == 2:
        return values.astype(np.float64)
    if values.ndim != 3:
        raise ValueError(f"Prediction must reduce to [H, W] or [C, H, W], got {values.shape}.")

    if values.shape[0] <= 8 and values.shape[1] > 8 and values.shape[2] > 8:
        channels = values.astype(np.float64)
    elif values.shape[-1] <= 8 and values.shape[0] > 8 and values.shape[1] > 8:
        channels = np.moveaxis(values, -1, 0).astype(np.float64)
    else:
        raise ValueError(f"Cannot infer the channel axis for prediction shape {values.shape}.")
    if channels.shape[0] == 1:
        return channels[0]
    if not 0 <= foreground_channel < channels.shape[0]:
        raise IndexError(
            f"foreground_channel={foreground_channel} is invalid for {channels.shape[0]} channels."
        )
    if kind == "logits":
        shifted = channels - channels.max(axis=0, keepdims=True)
        probabilities = np.exp(shifted)
        probabilities /= np.maximum(probabilities.sum(axis=0, keepdims=True), 1e-12)
        return probabilities[foreground_channel]
    return channels[foreground_channel]


def resize_float(values: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if values.shape == shape:
        return values.astype(np.float64)
    image = Image.fromarray(values.astype(np.float32), mode="F")
    return np.asarray(
        image.resize((shape[1], shape[0]), Image.Resampling.BILINEAR), dtype=np.float64
    )


def resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if mask.shape == shape:
        return mask
    image = Image.fromarray(mask.astype(np.uint8) * 255)
    return np.asarray(image.resize((shape[1], shape[0]), Image.Resampling.NEAREST)) > 0


def load_mask(path: Path, shape: tuple[int, int] | None = None) -> np.ndarray:
    mask = load_array(path)
    mask = np.asarray(mask)
    if mask.ndim == 3:
        if mask.shape[-1] in (3, 4):
            mask = mask[..., 0]
        elif mask.shape[0] == 1:
            mask = mask[0]
    if mask.ndim != 2:
        raise ValueError(f"Mask must have shape [H, W], got {mask.shape} from {path}.")
    result = mask > 0
    return resize_mask(result, shape) if shape is not None else result


def load_probability(item: ManifestItem, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray | None]:
    raw = _spatial_array(
        load_array(item.prediction, item.array_key), item.foreground_channel, item.prediction_kind
    )
    if item.prediction_kind == "mask":
        probability = (raw > 0).astype(np.float64)
    elif item.prediction_kind == "logits":
        if raw.min() < 0.0 or raw.max() > 1.0:
            probability = 1.0 / (1.0 + np.exp(-np.clip(raw, -60.0, 60.0)))
        else:
            probability = raw
    else:
        probability = raw
        if probability.max(initial=0.0) > 1.0 and probability.max(initial=0.0) <= 255.0:
            probability = probability / 255.0
    probability = np.clip(resize_float(probability, shape), 0.0, 1.0)

    uncertainty = None
    if item.uncertainty is not None:
        uncertainty = _spatial_array(
            load_array(item.uncertainty, item.uncertainty_key), 0, "probability"
        )
        if uncertainty.max(initial=0.0) > 1.0 and uncertainty.max(initial=0.0) <= 255.0:
            uncertainty = uncertainty / 255.0
        uncertainty = np.clip(resize_float(uncertainty, shape), 0.0, 1.0)
    return probability, uncertainty
