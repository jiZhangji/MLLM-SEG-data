from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def find_dump_paths(input_dir: Path, limit: int = 0) -> list[Path]:
    paths = sorted(input_dir.expanduser().resolve().glob("*.pt"))
    if limit > 0:
        paths = paths[:limit]
    if not paths:
        raise FileNotFoundError(f"No .pt dumps found in {input_dir}")
    return paths


def split_paths(
    paths: list[Path],
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[list[Path], list[Path]]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1).")
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(paths), generator=generator).tolist()
    val_count = max(1, int(round(len(paths) * val_fraction)))
    if len(paths) > 1:
        val_count = min(val_count, len(paths) - 1)
    val_ids = set(indices[:val_count])
    train = [p for i, p in enumerate(paths) if i not in val_ids]
    val = [p for i, p in enumerate(paths) if i in val_ids]
    return train, val


def _load_image_tensor(path: str | Path, image_size: int | None = None) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    if image_size is not None and image_size > 0:
        image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
    array = np.asarray(image).astype(np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def _load_mask_tensor(path: str | Path, image_size: int | None = None) -> torch.Tensor:
    mask = Image.open(path).convert("L")
    if image_size is not None and image_size > 0:
        mask = mask.resize((image_size, image_size), Image.Resampling.NEAREST)
    array = (np.asarray(mask) > 0).astype(np.float32)
    return torch.from_numpy(array).unsqueeze(0)


class RefinementDumpDataset(Dataset):
    def __init__(
        self,
        paths: list[Path],
        image_size: int | None = None,
    ) -> None:
        self.paths = paths
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        path = self.paths[index]
        payload = torch.load(path, map_location="cpu")
        image = _load_image_tensor(payload["image_path"], self.image_size)
        mask = _load_mask_tensor(payload["mask_path"], self.image_size)
        mask_logits = payload["mask_logits"].float()
        mask_hidden = payload["mask_hidden"].float()
        if mask_logits.ndim == 4 and mask_logits.shape[-2] == 1:
            mask_logits = mask_logits.squeeze(-2)
        if mask_hidden.ndim == 4 and mask_hidden.shape[-2] == 1:
            mask_hidden = mask_hidden.squeeze(-2)
        if mask_logits.shape[0] == 1:
            mask_logits = mask_logits[0]
        if mask_hidden.shape[0] == 1:
            mask_hidden = mask_hidden[0]
        return {
            "path": str(path),
            "name": str(payload.get("name") or path.stem),
            "image": image,
            "mask": mask,
            "mask_logits": mask_logits,
            "mask_hidden": mask_hidden,
            "grid_hw": tuple(int(x) for x in payload["grid_hw"]),
        }


def collate_refinement_dumps(batch: list[dict[str, Any]]) -> dict[str, Any]:
    grid_hw = batch[0]["grid_hw"]
    for item in batch:
        if item["grid_hw"] != grid_hw:
            raise ValueError("All items in a batch must share grid_hw. Use batch_size=1 for mixed dynamic grids.")
    return {
        "paths": [item["path"] for item in batch],
        "names": [item["name"] for item in batch],
        "images": torch.stack([item["image"] for item in batch], dim=0),
        "masks": torch.stack([item["mask"] for item in batch], dim=0),
        "mask_logits": torch.stack([item["mask_logits"] for item in batch], dim=0),
        "mask_hidden": torch.stack([item["mask_hidden"] for item in batch], dim=0),
        "grid_hw": grid_hw,
    }
