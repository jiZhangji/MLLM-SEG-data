from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


def resolve_path(path_value: str | Path, dump_path: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute() and path.exists():
        return path
    candidates = [dump_path.parent / path.name, dump_path.parent / path, Path.cwd() / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return path


def load_image(path: Path, image_size: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    tensor = torch.from_numpy(__import__("numpy").array(image)).permute(2, 0, 1).float() / 255.0
    return F.interpolate(tensor.unsqueeze(0), size=(image_size, image_size), mode="bilinear", align_corners=False).squeeze(0)


def load_mask(path: Path, image_size: int) -> torch.Tensor:
    mask = Image.open(path).convert("L")
    tensor = (torch.from_numpy(__import__("numpy").array(mask)).unsqueeze(0).float() / 255.0).clamp(0.0, 1.0)
    return F.interpolate(tensor.unsqueeze(0), size=(image_size, image_size), mode="nearest").squeeze(0)


class DumpDataset(Dataset):
    def __init__(self, dump_paths: list[Path], image_size: int = 896) -> None:
        self.dump_paths = dump_paths
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.dump_paths)

    def __getitem__(self, index: int) -> dict[str, object]:
        dump_path = self.dump_paths[index]
        payload = torch.load(dump_path, map_location="cpu", weights_only=False)
        image_path = resolve_path(payload["image_path"], dump_path)
        mask_path = resolve_path(payload["mask_path"], dump_path)
        return {
            "name": payload.get("name", dump_path.stem),
            "path": str(dump_path),
            "image": load_image(image_path, self.image_size),
            "gt_mask": load_mask(mask_path, self.image_size),
            "mask_logits": payload["mask_logits"].squeeze(0).float(),
            "mask_hidden": payload["mask_hidden"].squeeze(0).float(),
            "grid_hw": tuple(payload["grid_hw"]),
        }


def collate_dump_batch(items: list[dict[str, object]]) -> dict[str, object]:
    grid_hw = items[0]["grid_hw"]
    if any(item["grid_hw"] != grid_hw for item in items):
        raise ValueError("All samples in one batch must share grid_hw.")
    return {
        "name": [item["name"] for item in items],
        "path": [item["path"] for item in items],
        "image": torch.stack([item["image"] for item in items]),
        "gt_mask": torch.stack([item["gt_mask"] for item in items]),
        "mask_logits": torch.stack([item["mask_logits"] for item in items]),
        "mask_hidden": torch.stack([item["mask_hidden"] for item in items]),
        "grid_hw": grid_hw,
    }
