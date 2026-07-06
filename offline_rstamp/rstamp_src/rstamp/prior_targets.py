"""Utilities to derive simple structured prior targets from masks/bboxes."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

from .prior_schema import StructuredPrior, format_prior


def bbox_from_binary_mask(mask: list[list[int | bool]]) -> tuple[int, int, int, int] | None:
    xs: list[int] = []
    ys: list[int] = []
    for y, row in enumerate(mask):
        for x, value in enumerate(row):
            if value:
                xs.append(x)
                ys.append(y)
    if not xs:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def center_point_from_bbox(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def build_prior_from_sample(sample: dict[str, Any]) -> StructuredPrior:
    target = (
        sample.get("target")
        or sample.get("phrase")
        or sample.get("expression")
        or sample.get("query")
        or sample.get("text")
        or ""
    )
    bbox = sample.get("bbox")
    if bbox is not None and len(bbox) >= 4:
        bbox_tuple = tuple(float(x) for x in bbox[:4])
    else:
        bbox_tuple = None
    if bbox_tuple is not None:
        pos = [center_point_from_bbox(tuple(int(x) for x in bbox_tuple))]
    else:
        pos = []
    return StructuredPrior(target=str(target), bbox=bbox_tuple, positive_points=pos)


def enrich_jsonl(input_jsonl: Path, output_jsonl: Path) -> None:
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with input_jsonl.open("r", encoding="utf-8") as src, output_jsonl.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            sample = json.loads(line)
            prior = build_prior_from_sample(sample)
            sample["structured_prior"] = asdict(prior)
            sample["structured_prior_text"] = format_prior(prior)
            dst.write(json.dumps(sample, ensure_ascii=False) + "\n")

