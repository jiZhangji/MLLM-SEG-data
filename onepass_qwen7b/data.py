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
PROMPT_MODE_STRICT_QUERY = "strict_query"
PROMPT_MODE_SEMANTIC_ANCHOR = "semantic_anchor"
PROMPT_MODES = (PROMPT_MODE_STRICT_QUERY, PROMPT_MODE_SEMANTIC_ANCHOR)


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
    value = str(item.get("label") or item.get("target") or "").strip()
    if value:
        return value
    patterns = (
        r'Please segment the object this sentence describes:\s*"([^"]+)"',
        r'The\s+"([^"]+)"\s+refers to',
        r'"([^"]+)"',
    )
    for pattern in patterns:
        match = re.search(pattern, user_text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return user_text


def strict_instruction(item: dict[str, Any]) -> str:
    user_text = extract_user_text(item)
    if not user_text:
        raise ValueError("Training item has no user instruction.")
    target = extract_target_text(item, user_text)
    if "please segment" in user_text.lower() or "分割" in user_text:
        return user_text
    return f'Please segment the object this sentence describes: "{target}".'


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
        resolved = root / candidate
        if resolved.exists():
            return resolved.resolve()
    raise FileNotFoundError(f"Could not resolve path {raw!r} below {[str(root) for root in roots]}.")


@dataclass(frozen=True)
class OnePass7BRecord:
    index: int
    name: str
    image_path: Path
    mask_paths: tuple[Path, ...]
    instruction: str


@dataclass
class OnePass7BSample:
    record: OnePass7BRecord
    image: Image.Image
    mask: torch.Tensor


class OnePass7BDataset(Dataset):
    def __init__(
        self,
        json_paths: Sequence[str | Path],
        data_root: str | Path | None = None,
        limit: int = 0,
        offset: int = 0,
    ) -> None:
        if offset < 0 or limit < 0:
            raise ValueError("offset and limit must be non-negative.")
        roots = [] if data_root is None else [Path(data_root).resolve()]
        records: list[OnePass7BRecord] = []
        global_index = 0
        for json_value in json_paths:
            json_path = Path(json_value).resolve()
            values = json.loads(json_path.read_text(encoding="utf-8"))
            if not isinstance(values, list):
                raise TypeError(f"Expected a JSON list: {json_path}")
            local_roots = [*roots, json_path.parent]
            for item in values:
                if not isinstance(item, dict):
                    raise TypeError(f"Dataset item {global_index} is not a JSON object.")
                if global_index < offset:
                    global_index += 1
                    continue
                if limit > 0 and len(records) >= limit:
                    break
                image_values = _path_values(item, ("images", "image", "image_path"))
                mask_values = _path_values(item, ("masks", "mask", "mask_path"))
                if len(image_values) != 1:
                    raise ValueError(f"Item {global_index} must contain exactly one image.")
                if not mask_values:
                    raise ValueError(f"Item {global_index} has no mask.")
                image_path = resolve_path(image_values[0], local_roots)
                mask_paths = tuple(resolve_path(value, local_roots) for value in mask_values)
                records.append(
                    OnePass7BRecord(
                        index=global_index,
                        name=str(item.get("id") or item.get("name") or f"sample_{global_index:08d}"),
                        image_path=image_path,
                        mask_paths=mask_paths,
                        instruction=strict_instruction(item),
                    )
                )
                global_index += 1
            if limit > 0 and len(records) >= limit:
                break
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> OnePass7BSample:
        record = self.records[index]
        with Image.open(record.image_path) as handle:
            image = handle.convert("RGB").copy()
        merged = None
        for mask_path in record.mask_paths:
            with Image.open(mask_path) as handle:
                current = torch.from_numpy(np.asarray(handle.convert("L"), dtype=np.uint8).copy() > 0)
            if merged is None:
                merged = current
            else:
                if current.shape != merged.shape:
                    raise ValueError(
                        f"Mask shape mismatch for {record.name}: {tuple(current.shape)} vs {tuple(merged.shape)}"
                    )
                merged |= current
        if merged is None:
            raise RuntimeError(f"No mask loaded for {record.name}.")
        return OnePass7BSample(record=record, image=image, mask=merged)


def collate_samples(samples: list[OnePass7BSample]) -> list[OnePass7BSample]:
    return samples


def _chat_text(processor: Any, sample: OnePass7BSample, prompt_mode: str) -> str:
    user_message = {
        "role": "user",
        "content": [{"image": sample.image}, {"text": sample.record.instruction}],
    }
    if prompt_mode == PROMPT_MODE_STRICT_QUERY:
        prefix = processor.apply_chat_template(
            [user_message], tokenize=False, add_generation_prompt=True
        )
        return f"{prefix}{SEG_TOKEN}"
    if prompt_mode == PROMPT_MODE_SEMANTIC_ANCHOR:
        # Match STAMP's deterministic classification context without an
        # autoregressive generation call.
        return processor.apply_chat_template(
            [
                user_message,
                {
                    "role": "assistant",
                    "content": [{"text": f"The referred object is {SEG_TOKEN}."}],
                },
            ],
            tokenize=False,
        )
    raise ValueError(f"Unsupported OnePass prompt mode {prompt_mode!r}; expected {PROMPT_MODES}.")


def prepare_batch(
    samples: list[OnePass7BSample],
    processor: Any,
    device: torch.device,
    prompt_mode: str = PROMPT_MODE_STRICT_QUERY,
) -> dict[str, Any]:
    if not samples:
        raise ValueError("Cannot prepare an empty batch.")
    tokenizer = processor.tokenizer
    seg_token_id = tokenizer.convert_tokens_to_ids(SEG_TOKEN)
    mask_token_id = tokenizer.convert_tokens_to_ids(MASK_TOKEN)
    image_pad_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    texts = [_chat_text(processor, sample, prompt_mode) for sample in samples]
    encoded = processor(
        text=texts,
        images=[sample.image for sample in samples],
        padding=True,
        return_tensors="pt",
    )
    image_grid_thw = encoded["image_grid_thw"].long()
    merge_size = int(processor.image_processor.merge_size)
    sequences = []
    masks = []
    grid_shapes: list[tuple[int, int]] = []
    for index in range(len(samples)):
        valid = encoded["attention_mask"][index].bool()
        token_ids = encoded["input_ids"][index][valid]
        seg_indices = torch.where(token_ids.eq(seg_token_id))[0]
        if seg_indices.numel() != 1:
            raise ValueError(
                f"Prepared prompt for {samples[index].record.name} must contain one SEG token; "
                f"got {seg_indices.numel()}."
            )
        prefix = token_ids[: int(seg_indices.item()) + 1]
        temporal, raw_height, raw_width = (int(value) for value in image_grid_thw[index].tolist())
        if temporal != 1 or raw_height % merge_size or raw_width % merge_size:
            raise ValueError(
                f"Unsupported image grid {(temporal, raw_height, raw_width)} for merge_size={merge_size}."
            )
        grid_height = raw_height // merge_size
        grid_width = raw_width // merge_size
        query_count = grid_height * grid_width
        image_count = int(prefix.eq(image_pad_id).sum().item())
        if image_count != query_count:
            raise ValueError(
                f"Merged image token count {image_count} does not match mask count {query_count} "
                f"for {samples[index].record.name}."
            )
        query_ids = torch.full((query_count,), int(mask_token_id), dtype=torch.long)
        sequence = torch.cat([prefix.cpu(), query_ids], dim=0)
        sequences.append(sequence)
        masks.append(torch.ones_like(sequence))
        grid_shapes.append((grid_height, grid_width))
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    input_ids = pad_sequence(sequences, batch_first=True, padding_value=int(pad_id))
    attention_mask = pad_sequence(masks, batch_first=True, padding_value=0)
    return {
        "input_ids": input_ids.to(device, non_blocking=True),
        "attention_mask": attention_mask.to(device, non_blocking=True),
        "pixel_values": encoded["pixel_values"].to(device, non_blocking=True),
        "image_grid_thw": image_grid_thw.to(device, non_blocking=True),
        "grid_shapes": grid_shapes,
        "samples": samples,
    }


def grid_targets(
    samples: list[OnePass7BSample],
    grid_shapes: list[tuple[int, int]],
    device: torch.device,
) -> list[torch.Tensor]:
    if len(samples) != len(grid_shapes):
        raise ValueError("Sample/grid count mismatch.")
    targets = []
    for sample, (height, width) in zip(samples, grid_shapes):
        mask = sample.mask.to(device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        target = F.interpolate(mask, size=(height, width), mode="nearest").reshape(-1)
        targets.append(target)
    return targets
