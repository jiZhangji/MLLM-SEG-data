from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont


def tensor_to_numpy_2d(tensor: torch.Tensor) -> np.ndarray:
    arr = tensor.detach().float().cpu()
    while arr.ndim > 2:
        arr = arr.squeeze(0)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D tensor after squeeze, got {tuple(arr.shape)}")
    return arr.numpy()


def normalize_to_uint8(array: np.ndarray) -> np.ndarray:
    array = array.astype(np.float32)
    finite = np.isfinite(array)
    if not finite.any():
        return np.zeros_like(array, dtype=np.uint8)
    min_value = float(array[finite].min())
    max_value = float(array[finite].max())
    if max_value <= min_value:
        return np.zeros_like(array, dtype=np.uint8)
    out = (array - min_value) / (max_value - min_value)
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def heatmap_image(tensor: torch.Tensor, size: tuple[int, int]) -> Image.Image:
    arr = normalize_to_uint8(tensor_to_numpy_2d(tensor))
    gray = Image.fromarray(arr, mode="L").resize(size, Image.Resampling.BILINEAR)
    gray_arr = np.asarray(gray).astype(np.float32) / 255.0

    # Simple blue -> cyan -> yellow -> red heatmap without external deps.
    red = np.clip(2.0 * gray_arr - 0.25, 0.0, 1.0)
    green = np.clip(2.0 - np.abs(4.0 * gray_arr - 2.0), 0.0, 1.0)
    blue = np.clip(1.25 - 2.0 * gray_arr, 0.0, 1.0)
    rgb = np.stack([red, green, blue], axis=-1)
    return Image.fromarray((rgb * 255.0).astype(np.uint8), mode="RGB")


def mask_image(mask: torch.Tensor, size: tuple[int, int]) -> Image.Image:
    arr = (tensor_to_numpy_2d(mask) > 0.5).astype(np.uint8) * 255
    return Image.fromarray(arr, mode="L").resize(size, Image.Resampling.NEAREST).convert("RGB")


def overlay_mask(
    image: Image.Image,
    mask: torch.Tensor,
    color: tuple[int, int, int] = (255, 40, 40),
    alpha: float = 0.45,
) -> Image.Image:
    base = image.convert("RGB")
    mask_img = mask_image(mask, base.size).convert("L")
    color_img = Image.new("RGB", base.size, color)
    return Image.composite(Image.blend(base, color_img, alpha), base, mask_img)


def draw_patch_boxes(
    image: Image.Image,
    selected_ids: torch.Tensor,
    grid_hw: tuple[int, int],
    color: tuple[int, int, int] = (255, 220, 0),
    width: int = 3,
) -> Image.Image:
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    grid_h, grid_w = grid_hw
    image_w, image_h = out.size
    patch_w = image_w / grid_w
    patch_h = image_h / grid_h

    ids = selected_ids.detach().cpu().flatten().tolist()
    for patch_id in ids:
        row = int(patch_id) // grid_w
        col = int(patch_id) % grid_w
        x1 = round(col * patch_w)
        y1 = round(row * patch_h)
        x2 = round((col + 1) * patch_w)
        y2 = round((row + 1) * patch_h)
        for offset in range(width):
            draw.rectangle(
                [x1 + offset, y1 + offset, x2 - offset, y2 - offset],
                outline=color,
            )
    return out


def add_title(image: Image.Image, title: str, height: int = 28) -> Image.Image:
    out = Image.new("RGB", (image.width, image.height + height), (255, 255, 255))
    out.paste(image, (0, height))
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    draw.text((8, 7), title, fill=(0, 0, 0), font=font)
    return out


def make_panel(images: list[tuple[str, Image.Image]], columns: int = 3) -> Image.Image:
    if not images:
        raise ValueError("images must be non-empty.")
    titled = [add_title(img, title) for title, img in images]
    cell_w = max(img.width for img in titled)
    cell_h = max(img.height for img in titled)
    rows = int(np.ceil(len(titled) / columns))
    panel = Image.new("RGB", (columns * cell_w, rows * cell_h), (245, 245, 245))
    for index, img in enumerate(titled):
        row = index // columns
        col = index % columns
        panel.paste(img, (col * cell_w, row * cell_h))
    return panel


def upsample_grid_map(tensor: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    return F.interpolate(
        tensor.float(),
        size=(image_size[1], image_size[0]),
        mode="bilinear",
        align_corners=False,
    )


def save_image(path: Path, image: Image.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
