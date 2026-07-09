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


def load_mask(path: Path, image_size: int) -> torch.Tensor:
    mask = Image.open(path).convert("L")
    array = __import__("numpy").array(mask)
    tensor = (torch.from_numpy(array).unsqueeze(0).float() / 255.0).clamp(0.0, 1.0)
    return F.interpolate(tensor.unsqueeze(0), size=(image_size, image_size), mode="nearest").squeeze(0)


def token_targets_from_mask(mask: torch.Tensor, grid_hw: tuple[int, int]) -> torch.Tensor:
    target = F.interpolate(mask.unsqueeze(0), size=grid_hw, mode="area").squeeze(0).squeeze(0)
    return (target >= 0.5).long().flatten()


def token_soft_targets_from_mask(mask: torch.Tensor, grid_hw: tuple[int, int]) -> torch.Tensor:
    return F.interpolate(mask.unsqueeze(0), size=grid_hw, mode="area").squeeze(0).squeeze(0).flatten()


class TokenDumpDataset(Dataset):
    def __init__(self, dump_paths: list[Path], image_size: int = 896) -> None:
        self.dump_paths = dump_paths
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.dump_paths)

    def __getitem__(self, index: int) -> dict[str, object]:
        dump_path = self.dump_paths[index]
        payload = torch.load(dump_path, map_location="cpu", weights_only=False)
        grid_hw = tuple(payload["grid_hw"])
        mask_path = resolve_path(payload["mask_path"], dump_path)
        gt_mask = load_mask(mask_path, self.image_size)
        return {
            "name": payload.get("name", dump_path.stem),
            "path": str(dump_path),
            "mask_logits": payload["mask_logits"].squeeze(0).float(),
            "mask_hidden": payload["mask_hidden"].squeeze(0).float(),
            "gt_mask": gt_mask,
            "target_tokens": token_targets_from_mask(gt_mask, grid_hw),
            "target_soft": token_soft_targets_from_mask(gt_mask, grid_hw),
            "grid_hw": grid_hw,
        }


class CachedTokenDataset(Dataset):
    def __init__(self, items: list[dict[str, object]]) -> None:
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, object]:
        return self.items[index]


def build_token_cache(
    dump_paths: list[Path],
    cache_path: Path,
    image_size: int = 896,
    progress=None,
) -> dict[str, object]:
    iterator = progress(dump_paths) if progress is not None else dump_paths
    items = []
    for dump_path in iterator:
        payload = torch.load(dump_path, map_location="cpu", weights_only=False)
        grid_hw = tuple(payload["grid_hw"])
        mask_path = resolve_path(payload["mask_path"], dump_path)
        gt_mask = load_mask(mask_path, image_size)
        items.append(
            {
                "name": payload.get("name", dump_path.stem),
                "path": str(dump_path),
                "mask_logits": payload["mask_logits"].squeeze(0).float(),
                "mask_hidden": payload["mask_hidden"].squeeze(0).float(),
                "target_tokens": token_targets_from_mask(gt_mask, grid_hw),
                "target_soft": token_soft_targets_from_mask(gt_mask, grid_hw),
                "grid_hw": grid_hw,
            }
        )
    cache = {
        "version": 1,
        "image_size": image_size,
        "num_items": len(items),
        "items": items,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    torch.save(cache, tmp_path)
    tmp_path.replace(cache_path)
    return cache


def load_token_cache(cache_path: Path) -> dict[str, object]:
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    if not isinstance(cache, dict) or "items" not in cache:
        raise ValueError(f"Invalid token cache: {cache_path}")
    return cache


def collate_token_dumps(items: list[dict[str, object]]) -> dict[str, object]:
    grid_hw = items[0]["grid_hw"]
    if any(item["grid_hw"] != grid_hw for item in items):
        raise ValueError("All samples in one batch must share grid_hw.")
    batch = {
        "name": [item["name"] for item in items],
        "path": [item["path"] for item in items],
        "mask_logits": torch.stack([item["mask_logits"] for item in items]),
        "mask_hidden": torch.stack([item["mask_hidden"] for item in items]),
        "target_tokens": torch.stack([item["target_tokens"] for item in items]),
        "target_soft": torch.stack([item["target_soft"] for item in items]),
        "grid_hw": grid_hw,
    }
    if "gt_mask" in items[0]:
        batch["gt_mask"] = torch.stack([item["gt_mask"] for item in items])
    return batch
