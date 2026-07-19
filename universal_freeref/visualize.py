from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def overlay(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    base = image.astype(np.float32)
    tint = np.zeros_like(base)
    tint[...] = np.asarray(color, dtype=np.float32)
    alpha = np.asarray(mask, dtype=np.float32)[..., None] * 0.45
    return np.clip(base * (1.0 - alpha) + tint * alpha, 0, 255).astype(np.uint8)


def correction_map(image: np.ndarray, target: np.ndarray, coarse: np.ndarray, refined: np.ndarray) -> np.ndarray:
    result = image.astype(np.float32) * 0.45
    corrected = coarse != target
    corrected &= refined == target
    regressed = coarse == target
    regressed &= refined != target
    unchanged_error = (refined != target) & ~regressed
    result[corrected] = (40, 210, 80)
    result[regressed] = (235, 65, 55)
    result[unchanged_error] = (245, 180, 40)
    return np.clip(result, 0, 255).astype(np.uint8)


def save_panel(
    path: Path,
    image: np.ndarray,
    target: np.ndarray,
    coarse: np.ndarray,
    refined: np.ndarray,
    uncertainty: np.ndarray,
) -> None:
    heat = np.stack(
        [uncertainty, np.sqrt(np.clip(uncertainty, 0.0, 1.0)), 1.0 - uncertainty], axis=-1
    )
    panels = [
        ("Image", image),
        ("Ground truth", overlay(image, target, (40, 210, 80))),
        ("Original", overlay(image, coarse, (235, 65, 55))),
        ("Uncertainty", np.clip(heat * 255.0, 0, 255).astype(np.uint8)),
        ("FreeRef", overlay(image, refined, (45, 120, 235))),
        ("Change: green fix / red regress", correction_map(image, target, coarse, refined)),
    ]
    height, width = image.shape[:2]
    label_height = 26
    canvas = Image.new("RGB", (width * len(panels), height + label_height), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (label, values) in enumerate(panels):
        canvas.paste(Image.fromarray(values), (index * width, label_height))
        draw.text((index * width + 5, 6), label, fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
