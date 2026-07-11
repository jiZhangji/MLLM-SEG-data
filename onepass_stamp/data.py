from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


SEG_TOKEN = "<|seg|>"
MASK_TOKEN = "<|mask|>"
TASK_TOKEN = "<|task_seg|>"


def _content_text(message: dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    values = []
    if isinstance(content, list):
        for part in content:
            if isinstance(part, str):
                values.append(part)
            elif isinstance(part, dict) and "text" in part and part.get("text") is not None:
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
    patterns = [
        r'Please segment the object this sentence describes:\s*"([^"]+)"',
        r'The\s+"([^"]+)"\s+refers to',
        r'"([^"]+)"',
    ]
    for pattern in patterns:
        match = re.search(pattern, user_text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return user_text.strip()


def extract_assistant_prefix(item: dict[str, Any], target_text: str) -> str:
    for message in _messages(item):
        role = str(message.get("role") or message.get("from") or "").lower()
        if role in {"assistant", "gpt"}:
            text = _content_text(message).replace("<SEG>", SEG_TOKEN)
            index = text.find(SEG_TOKEN)
            if index >= 0:
                return text[: index + len(SEG_TOKEN)].strip()
    return f'The "{target_text}" refers to {SEG_TOKEN}'


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
    # Full STAMP JSON files already contain absolute paths. Avoid 160k+ remote
    # filesystem stat calls while indexing image/mask records; PIL will still
    # raise a precise FileNotFoundError when a sampled file is opened.
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
class OnePassRecord:
    index: int
    name: str
    image_path: Path
    mask_paths: tuple[Path, ...]
    user_text: str
    assistant_prefix: str


@dataclass
class OnePassSample:
    record: OnePassRecord
    image: Image.Image
    mask: torch.Tensor


class OnePassDataset(Dataset):
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

        self.records: list[OnePassRecord] = []
        for local_index, raw_item in enumerate(values):
            if not isinstance(raw_item, dict):
                raise TypeError(f"Dataset item {offset + local_index} is not a JSON object.")
            image_values = _path_values(raw_item, ("images", "image", "image_path"))
            mask_values = _path_values(raw_item, ("masks", "mask", "mask_path"))
            if len(image_values) != 1:
                raise ValueError(
                    f"OnePass-STAMP currently requires one image per sample; item {offset + local_index} "
                    f"contains {len(image_values)}."
                )
            if not mask_values:
                raise ValueError(f"Dataset item {offset + local_index} has no target mask.")
            user_text = extract_user_text(raw_item)
            if not user_text:
                raise ValueError(f"Dataset item {offset + local_index} has no user prompt.")
            target_text = extract_target_text(raw_item, user_text)
            image_path = resolve_path(image_values[0], roots)
            mask_paths = tuple(resolve_path(value, roots) for value in mask_values)
            name = str(raw_item.get("id") or raw_item.get("name") or image_path.stem)
            self.records.append(
                OnePassRecord(
                    index=offset + local_index,
                    name=name,
                    image_path=image_path,
                    mask_paths=mask_paths,
                    user_text=user_text,
                    assistant_prefix=extract_assistant_prefix(raw_item, target_text),
                )
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> OnePassSample:
        record = self.records[index]
        with Image.open(record.image_path) as handle:
            image = handle.convert("RGB").copy()
        merged_mask = None
        for mask_path in record.mask_paths:
            with Image.open(mask_path) as handle:
                array = np.asarray(handle.convert("L"), dtype=np.uint8).copy()
            current = torch.from_numpy(array > 0)
            if merged_mask is None:
                merged_mask = current
            else:
                if current.shape != merged_mask.shape:
                    raise ValueError(
                        f"Mask shape mismatch for {record.name}: {tuple(current.shape)} vs {tuple(merged_mask.shape)}."
                    )
                merged_mask |= current
        if merged_mask is None:
            raise RuntimeError(f"No mask loaded for {record.name}.")
        return OnePassSample(record=record, image=image, mask=merged_mask)


def collate_samples(samples: list[OnePassSample]) -> list[OnePassSample]:
    return samples


def _chat_text(processor: Any, sample: OnePassSample, use_task_token: bool) -> str:
    user_text = sample.record.user_text
    if use_task_token:
        user_text = f"{TASK_TOKEN}\n{user_text}"
    messages = [
        {"role": "user", "content": [{"image": sample.image}, {"text": user_text}]},
        {"role": "assistant", "content": [{"text": sample.record.assistant_prefix}]},
    ]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def prepare_onepass_batch(
    samples: list[OnePassSample],
    processor: Any,
    device: torch.device,
    use_task_token: bool,
) -> dict[str, Any]:
    if not samples:
        raise ValueError("Cannot prepare an empty batch.")
    tokenizer = processor.tokenizer
    seg_token_id = tokenizer.convert_tokens_to_ids(SEG_TOKEN)
    mask_token_id = tokenizer.convert_tokens_to_ids(MASK_TOKEN)
    image_pad_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    task_token_id = tokenizer.convert_tokens_to_ids(TASK_TOKEN) if use_task_token else None
    for token, token_id in [(SEG_TOKEN, seg_token_id), (MASK_TOKEN, mask_token_id)]:
        if token_id is None or token_id < 0 or token_id == tokenizer.unk_token_id:
            raise ValueError(f"Tokenizer does not contain required token {token}.")

    texts = [_chat_text(processor, sample, use_task_token) for sample in samples]
    encoded = processor(
        text=texts,
        images=[sample.image for sample in samples],
        padding=True,
        return_tensors="pt",
    )
    if "image_grid_thw" not in encoded or "pixel_values" not in encoded:
        raise KeyError("Processor output must contain image_grid_thw and pixel_values.")

    image_grid_thw = encoded["image_grid_thw"].long()
    merge_size = int(processor.image_processor.merge_size)
    if image_grid_thw.shape[0] != len(samples):
        raise ValueError(
            f"Expected one image grid per sample, got {image_grid_thw.shape[0]} for {len(samples)} samples."
        )

    sequences = []
    masks = []
    grid_shapes: list[tuple[int, int]] = []
    for index in range(len(samples)):
        valid = encoded["attention_mask"][index].bool()
        token_ids = encoded["input_ids"][index][valid]
        seg_indices = torch.where(token_ids.eq(seg_token_id))[0]
        if seg_indices.numel() != 1:
            raise ValueError(
                f"Prepared prompt for {samples[index].record.name} must contain exactly one {SEG_TOKEN}; "
                f"got {seg_indices.numel()}."
            )
        prefix = token_ids[: int(seg_indices.item()) + 1]
        temporal, grid_height, grid_width = (int(value) for value in image_grid_thw[index].tolist())
        if temporal != 1 or grid_height % merge_size or grid_width % merge_size:
            raise ValueError(
                f"Unsupported image grid {(temporal, grid_height, grid_width)} for merge_size={merge_size}."
            )
        height, width = grid_height // merge_size, grid_width // merge_size
        query_count = height * width
        image_token_count = int(prefix.eq(image_pad_id).sum().item())
        if image_token_count != query_count:
            raise ValueError(
                f"Image placeholder/query mismatch for {samples[index].record.name}: "
                f"image_tokens={image_token_count}, queries={query_count}."
            )
        if use_task_token and int(prefix.eq(task_token_id).sum().item()) != 1:
            raise ValueError(f"Prepared prompt for {samples[index].record.name} is missing {TASK_TOKEN}.")
        query_ids = torch.full((query_count,), mask_token_id, dtype=torch.long)
        sequence = torch.cat([prefix.cpu(), query_ids], dim=0)
        sequences.append(sequence)
        masks.append(torch.ones_like(sequence))
        grid_shapes.append((height, width))

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    input_ids = pad_sequence(sequences, batch_first=True, padding_value=int(pad_token_id))
    attention_mask = pad_sequence(masks, batch_first=True, padding_value=0)
    return {
        "input_ids": input_ids.to(device, non_blocking=True),
        "attention_mask": attention_mask.to(device, non_blocking=True),
        "pixel_values": encoded["pixel_values"].to(device, non_blocking=True),
        "image_grid_thw": image_grid_thw.to(device, non_blocking=True),
        "grid_shapes": grid_shapes,
        "samples": samples,
    }


def grid_targets(samples: list[OnePassSample], grid_shapes: list[tuple[int, int]], device: torch.device):
    if len(samples) != len(grid_shapes):
        raise ValueError("Sample/grid count mismatch.")
    targets = []
    for sample, (height, width) in zip(samples, grid_shapes):
        mask = sample.mask.to(device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        resized = F.interpolate(mask, size=(height, width), mode="nearest").reshape(-1)
        targets.append(resized)
    return targets
