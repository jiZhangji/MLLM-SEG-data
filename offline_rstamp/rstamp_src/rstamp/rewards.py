"""Mask-aware reward utilities for lightweight R-STAMP RL."""

from __future__ import annotations

import math
from typing import Any


def _to_bool_grid(mask: Any) -> list[list[bool]]:
    """Convert a nested list/tuple mask to a bool grid.

    Keep this dependency-free so it can run in offline preprocessing scripts.
    Training code may replace it with torch/numpy implementations.
    """
    return [[bool(v) for v in row] for row in mask]


def mask_iou(pred_mask: Any, gt_mask: Any, eps: float = 1e-6) -> float:
    pred = _to_bool_grid(pred_mask)
    gt = _to_bool_grid(gt_mask)
    inter = 0
    union = 0
    for prow, grow in zip(pred, gt):
        for p, g in zip(prow, grow):
            inter += int(p and g)
            union += int(p or g)
    return float(inter) / float(union + eps)


def _boundary(mask: list[list[bool]]) -> set[tuple[int, int]]:
    h = len(mask)
    w = len(mask[0]) if h else 0
    out: set[tuple[int, int]] = set()
    for y in range(h):
        for x in range(w):
            if not mask[y][x]:
                continue
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                yy, xx = y + dy, x + dx
                if yy < 0 or yy >= h or xx < 0 or xx >= w or not mask[yy][xx]:
                    out.add((y, x))
                    break
    return out


def boundary_f_score(pred_mask: Any, gt_mask: Any, eps: float = 1e-6) -> float:
    pred_b = _boundary(_to_bool_grid(pred_mask))
    gt_b = _boundary(_to_bool_grid(gt_mask))
    if not pred_b and not gt_b:
        return 1.0
    if not pred_b or not gt_b:
        return 0.0
    hit = len(pred_b & gt_b)
    precision = hit / (len(pred_b) + eps)
    recall = hit / (len(gt_b) + eps)
    return 2 * precision * recall / (precision + recall + eps)


def conciseness_penalty(text: str, max_tokens: int = 96) -> float:
    n = len(text.split())
    if n <= max_tokens:
        return 0.0
    return -math.log1p(n - max_tokens)


def format_reward(text: str) -> float:
    required = ["<TARGET>", "</TARGET>"]
    score = sum(1.0 for token in required if token in text)
    location = ("<BOX>" in text and "</BOX>" in text) or ("<POS>" in text and "</POS>" in text)
    return (score + float(location)) / 3.0


def combined_mask_reward(
    pred_mask: Any,
    gt_mask: Any,
    prior_text: str,
    iou_weight: float = 1.0,
    boundary_weight: float = 0.2,
    format_weight: float = 0.2,
    concise_weight: float = 0.05,
) -> float:
    return (
        iou_weight * mask_iou(pred_mask, gt_mask)
        + boundary_weight * boundary_f_score(pred_mask, gt_mask)
        + format_weight * format_reward(prior_text)
        + concise_weight * conciseness_penalty(prior_text)
    )

