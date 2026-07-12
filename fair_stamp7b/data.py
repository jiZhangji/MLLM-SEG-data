from __future__ import annotations

from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence

from onepass_qwen7b.data import (
    MASK_TOKEN,
    SEG_TOKEN,
    OnePass7BDataset,
    OnePass7BSample,
    collate_samples,
    grid_targets,
)


FairStamp7BDataset = OnePass7BDataset


def _last_subsequence(sequence: torch.Tensor, pattern: list[int]) -> int:
    if not pattern or len(pattern) > sequence.numel():
        return -1
    values = sequence.tolist()
    for start in range(len(values) - len(pattern), -1, -1):
        if values[start : start + len(pattern)] == pattern:
            return start
    return -1


def _messages(sample: OnePass7BSample) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [{"image": sample.image}, {"text": sample.record.instruction}],
        },
        {
            "role": "assistant",
            "content": [{"text": f"The referred object is {SEG_TOKEN}."}],
        },
    ]


def prepare_batch(samples: list[OnePass7BSample], processor: Any, device: torch.device) -> dict[str, Any]:
    """Build both native STAMP passes from one processor batch.

    The language pass teacher-forces SEG. The segmentation pass keeps the
    prefix through SEG and appends one MASK query per merged visual token.
    """
    if not samples:
        raise ValueError("Cannot prepare an empty batch.")
    tokenizer = processor.tokenizer
    seg_id = int(tokenizer.convert_tokens_to_ids(SEG_TOKEN))
    mask_id = int(tokenizer.convert_tokens_to_ids(MASK_TOKEN))
    image_pad_id = int(tokenizer.convert_tokens_to_ids("<|image_pad|>"))
    assistant_header = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
    texts = [processor.apply_chat_template(_messages(sample), tokenize=False) for sample in samples]
    encoded = processor(
        text=texts,
        images=[sample.image for sample in samples],
        padding=True,
        return_tensors="pt",
    )
    labels = encoded["input_ids"].clone()
    labels[~encoded["attention_mask"].bool()] = -100
    labels[labels.eq(image_pad_id)] = -100
    cls_sequences: list[torch.Tensor] = []
    cls_masks: list[torch.Tensor] = []
    grid_shapes: list[tuple[int, int]] = []
    merge_size = int(processor.image_processor.merge_size)
    for index, sample in enumerate(samples):
        valid = encoded["attention_mask"][index].bool()
        sequence = encoded["input_ids"][index][valid]
        header_start = _last_subsequence(sequence, assistant_header)
        if header_start < 0:
            raise ValueError(
                f"Could not locate assistant response in tokenized prompt for {sample.record.name}."
            )
        response_start = header_start + len(assistant_header)
        valid_indices = torch.where(valid)[0]
        labels[index, valid_indices[:response_start]] = -100
        seg_positions = torch.where(sequence.eq(seg_id))[0]
        if seg_positions.numel() != 1:
            raise ValueError(
                f"Teacher-forced response for {sample.record.name} must contain one SEG token; "
                f"got {seg_positions.numel()}."
            )
        temporal, raw_height, raw_width = (int(value) for value in encoded["image_grid_thw"][index])
        if temporal != 1 or raw_height % merge_size or raw_width % merge_size:
            raise ValueError(
                f"Unsupported grid {(temporal, raw_height, raw_width)} for merge_size={merge_size}."
            )
        height, width = raw_height // merge_size, raw_width // merge_size
        query_count = height * width
        prefix = sequence[: int(seg_positions.item()) + 1]
        if int(prefix.eq(image_pad_id).sum()) != query_count:
            raise ValueError(
                f"Image/query count mismatch for {sample.record.name}: "
                f"{int(prefix.eq(image_pad_id).sum())} vs {query_count}."
            )
        cls_sequence = torch.cat(
            [prefix.cpu(), torch.full((query_count,), mask_id, dtype=torch.long)], dim=0
        )
        cls_sequences.append(cls_sequence)
        cls_masks.append(torch.ones_like(cls_sequence))
        grid_shapes.append((height, width))
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    cls_input_ids = pad_sequence(cls_sequences, batch_first=True, padding_value=int(pad_id))
    cls_attention_mask = pad_sequence(cls_masks, batch_first=True, padding_value=0)
    return {
        "lm_input_ids": encoded["input_ids"].to(device, non_blocking=True),
        "lm_attention_mask": encoded["attention_mask"].to(device, non_blocking=True),
        "lm_labels": labels.to(device, non_blocking=True),
        "cls_input_ids": cls_input_ids.to(device, non_blocking=True),
        "cls_attention_mask": cls_attention_mask.to(device, non_blocking=True),
        "pixel_values": encoded["pixel_values"].to(device, non_blocking=True),
        "image_grid_thw": encoded["image_grid_thw"].long().to(device, non_blocking=True),
        "grid_shapes": grid_shapes,
        "samples": samples,
    }


__all__ = [
    "FairStamp7BDataset",
    "collate_samples",
    "grid_targets",
    "prepare_batch",
]
