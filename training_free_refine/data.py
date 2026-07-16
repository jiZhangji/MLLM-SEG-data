from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def _content_text(message: dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    values = []
    if isinstance(content, list):
        for part in content:
            if isinstance(part, str):
                values.append(part)
            elif isinstance(part, dict) and part.get("text") is not None:
                values.append(str(part["text"]))
    return "\n".join(value for value in values if value)


def _messages(item: dict[str, Any]) -> list[dict[str, Any]]:
    values = item.get("messages") or item.get("conversations") or []
    return values if isinstance(values, list) else []


def extract_user_text(item: dict[str, Any]) -> str:
    for message in _messages(item):
        role = str(message.get("role") or message.get("from") or "").lower()
        if role in {"user", "human"}:
            text = _content_text(message).strip()
            if text:
                return text
    return str(item.get("query") or item.get("text") or item.get("prompt") or "").strip()


def extract_target_text(item: dict[str, Any], user_text: str) -> str:
    label = str(item.get("label") or item.get("target") or "").strip()
    if label:
        return label
    for pattern in [
        r'Please segment the object this sentence describes:\s*"([^"]+)"',
        r'The\s+"([^"]+)"\s+refers to',
        r'"([^"]+)"',
    ]:
        match = re.search(pattern, user_text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return user_text.strip()


def _path_values(item: dict[str, Any], keys: Sequence[str]) -> list[str]:
    for key in keys:
        value = item.get(key)
        if value is None:
            continue
        if isinstance(value, list):
            return [str(entry) for entry in value]
        return [str(value)]
    return []


def resolve_path(raw: str, roots: Sequence[Path]) -> Path:
    candidate = Path(raw)
    if candidate.is_absolute():
        return candidate
    if candidate.exists():
        return candidate.resolve()
    for root in roots:
        resolved = root / raw
        if resolved.exists():
            return resolved.resolve()
    raise FileNotFoundError(f"Could not resolve path {raw!r} below {[str(root) for root in roots]}.")


@dataclass(frozen=True)
class ReferringSegRecord:
    index: int
    name: str
    image_path: Path
    mask_paths: tuple[Path, ...]
    user_text: str


@dataclass
class ReferringSegSample:
    record: ReferringSegRecord
    image: Image.Image
    mask: torch.Tensor


class ReferringSegDataset(Dataset):
    """Read the flat STAMP/Text4Seg evaluation JSON without model dependencies."""

    def __init__(
        self,
        json_path: str | Path,
        data_root: str | Path | None = None,
        limit: int = 0,
        offset: int = 0,
    ) -> None:
        self.json_path = Path(json_path).resolve()
        values = json.loads(self.json_path.read_text(encoding="utf-8"))
        if not isinstance(values, list):
            raise TypeError(f"Expected a JSON list: {self.json_path}")
        if offset < 0:
            raise ValueError("offset must be non-negative.")
        values = values[offset : offset + limit if limit > 0 else None]
        roots = [self.json_path.parent]
        if data_root is not None:
            roots.insert(0, Path(data_root).resolve())

        self.records: list[ReferringSegRecord] = []
        for local_index, raw_item in enumerate(values):
            if not isinstance(raw_item, dict):
                raise TypeError(f"Dataset item {offset + local_index} is not a JSON object.")
            image_values = _path_values(raw_item, ("images", "image", "image_path"))
            mask_values = _path_values(raw_item, ("masks", "mask", "mask_path"))
            if len(image_values) != 1:
                raise ValueError(
                    f"Expected one image for item {offset + local_index}, got {len(image_values)}."
                )
            if not mask_values:
                raise ValueError(f"Dataset item {offset + local_index} has no target mask.")
            user_text = extract_user_text(raw_item)
            if not user_text:
                raise ValueError(f"Dataset item {offset + local_index} has no user prompt.")
            image_path = resolve_path(image_values[0], roots)
            mask_paths = tuple(resolve_path(value, roots) for value in mask_values)
            self.records.append(
                ReferringSegRecord(
                    index=offset + local_index,
                    name=str(raw_item.get("id") or raw_item.get("name") or image_path.stem),
                    image_path=image_path,
                    mask_paths=mask_paths,
                    user_text=user_text,
                )
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> ReferringSegSample:
        record = self.records[index]
        with Image.open(record.image_path) as handle:
            image = handle.convert("RGB").copy()
        merged_mask = None
        for mask_path in record.mask_paths:
            with Image.open(mask_path) as handle:
                current = torch.from_numpy(np.asarray(handle.convert("L"), dtype=np.uint8).copy() > 0)
            if merged_mask is None:
                merged_mask = current
            else:
                if current.shape != merged_mask.shape:
                    raise ValueError(
                        f"Mask shape mismatch for {record.name}: {tuple(current.shape)} vs "
                        f"{tuple(merged_mask.shape)}."
                    )
                merged_mask |= current
        if merged_mask is None:
            raise RuntimeError(f"No mask loaded for {record.name}.")
        return ReferringSegSample(record=record, image=image, mask=merged_mask)
