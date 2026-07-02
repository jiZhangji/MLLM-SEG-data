#!/usr/bin/env python3
"""Convert downloaded RefCOCO-family JSONL files into STAMP training JSON.

The downloaded `refcoco_family/*.json` files are JSONL files. Each line already
contains an image filename, conversation, object label, bbox, patches and an RLE
mask. STAMP's current `train/main_uni.py` expects JSON files under:

    playground/data/json_files/

where each item roughly has:

    {
      "images": ["/abs/path/to/image.jpg"],
      "masks": ["/abs/path/to/mask.png"],
      "messages": [...]
    }

This script creates that derived training format and stores binary masks as PNGs.
It can also add a lightweight `structured_prior` field for the future R-STAMP
variant while keeping the baseline fields unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
from tqdm import tqdm


DEFAULT_ROOT = Path("/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG")

DATASET_FILES = {
    "refcoco": "refcoco_train.json",
    "refcoco+": "refcoco+_train.json",
    "refcocog": "refcocog_train.json",
}

OUTPUT_JSON_NAMES = {
    "refcoco": "refcoco_formatted_all_sentences_doubled_mp.json",
    "refcoco+": "refcoco+_formatted_all_sentences_doubled_mp.json",
    "refcocog": "refcocog_formatted_all_sentences_doubled_mp.json",
}


def load_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
            if limit is not None and len(items) >= limit:
                break
    return items


def extract_query(sample: dict[str, Any]) -> str:
    objects = sample.get("objects") or []
    if objects and objects[0].get("label"):
        return str(objects[0]["label"]).strip()
    conversations = sample.get("conversations") or []
    if conversations:
        text = str(conversations[0].get("value", ""))
        match = re.search(r'"([^"]+)"', text)
        if match:
            return match.group(1).strip()
        return text.strip()
    return ""


def decode_rle_to_mask(rle: dict[str, Any]) -> np.ndarray:
    # pycocotools accepts compressed RLE with counts as str/bytes.
    rle_obj = dict(rle)
    counts = rle_obj.get("counts")
    if isinstance(counts, str):
        rle_obj["counts"] = counts.encode("utf-8")
    mask = mask_utils.decode(rle_obj)
    if mask.ndim == 3:
        mask = np.sum(mask, axis=2)
    mask = (mask > 0).astype(np.uint8)
    return mask


def format_prior(query: str, obj: dict[str, Any]) -> str:
    bbox = obj.get("bbox")
    parts = [f"<TARGET> {query} </TARGET>"]
    if bbox and len(bbox) >= 4:
        x1, y1, x2, y2 = [float(x) for x in bbox[:4]]
        parts.append(f"<BOX> {x1:.4f} {y1:.4f} {x2:.4f} {y2:.4f} </BOX>")
        parts.append(f"<POS> {(x1 + x2) / 2:.4f},{(y1 + y2) / 2:.4f} </POS>")
    return " ".join(parts)


def make_stamp_item(
    sample: dict[str, Any],
    image_root: Path,
    mask_dir: Path,
    dataset_name: str,
    item_index: int,
    include_prior: bool,
) -> dict[str, Any] | None:
    objects = sample.get("objects") or []
    if not objects:
        return None
    obj = objects[0]
    rle = obj.get("rle")
    if not rle:
        return None

    image_name = sample.get("image")
    if not image_name:
        return None
    image_path = image_root / image_name
    if not image_path.exists():
        # Keep the item out if image is missing; report counts at the end.
        return None

    query = extract_query(sample)
    if not query:
        return None

    mask_arr = decode_rle_to_mask(rle)
    mask_name = f"{dataset_name}_{sample.get('id', item_index)}_{item_index}.png"
    mask_path = mask_dir / dataset_name / mask_name
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask_arr * 255).save(mask_path)

    question = f'Please segment the object this sentence describes: "{query}".'
    answer = f'The "{query}" refers to <|seg|> in this image.'
    item: dict[str, Any] = {
        "images": [str(image_path)],
        "masks": [str(mask_path)],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": answer},
                ],
            },
        ],
        "source_dataset": dataset_name,
        "source_id": sample.get("id"),
        "label": query,
        "bbox": obj.get("bbox"),
    }
    if include_prior:
        item["structured_prior"] = {
            "target": query,
            "bbox": obj.get("bbox"),
            "positive_points": [],
            "negative_points": [],
        }
        item["structured_prior_text"] = format_prior(query, obj)
    return item


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def convert_dataset(
    dataset_name: str,
    input_path: Path,
    image_root: Path,
    mask_root: Path,
    output_json_dir: Path,
    include_prior: bool,
    limit: int | None,
    duplicate: int,
) -> dict[str, Any]:
    raw_items = load_jsonl(input_path, limit=limit)
    converted: list[dict[str, Any]] = []
    skipped = 0
    for dup_idx in range(duplicate):
        iterable = enumerate(raw_items)
        for idx, sample in tqdm(iterable, total=len(raw_items), desc=f"{dataset_name} x{dup_idx + 1}"):
            item = make_stamp_item(
                sample=sample,
                image_root=image_root,
                mask_dir=mask_root,
                dataset_name=dataset_name.replace("+", "plus"),
                item_index=dup_idx * len(raw_items) + idx,
                include_prior=include_prior,
            )
            if item is None:
                skipped += 1
                continue
            converted.append(item)

    output_name = OUTPUT_JSON_NAMES[dataset_name]
    output_path = output_json_dir / output_name
    write_json(output_path, converted)
    return {
        "dataset": dataset_name,
        "input": str(input_path),
        "output": str(output_path),
        "raw_items": len(raw_items),
        "converted_items": len(converted),
        "skipped": skipped,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--stamp-code-dir", type=Path, default=None)
    parser.add_argument("--output-json-dir", type=Path, default=None)
    parser.add_argument("--mask-root", type=Path, default=None)
    parser.add_argument("--datasets", nargs="+", default=["refcoco", "refcoco+", "refcocog"], choices=sorted(DATASET_FILES))
    parser.add_argument("--limit", type=int, default=None, help="Debug limit per dataset.")
    parser.add_argument("--duplicate", type=int, default=1, help="Duplicate each dataset N times. Use 1 for clean baseline.")
    parser.add_argument("--include-prior", action="store_true", help="Add R-STAMP structured_prior fields.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    root = args.root.expanduser().resolve()
    data_root = (args.data_root or root / "data").expanduser().resolve()
    stamp_code_dir = (args.stamp_code_dir or root / "code" / "STAMP").expanduser().resolve()
    output_json_dir = (args.output_json_dir or stamp_code_dir / "playground" / "data" / "json_files").expanduser().resolve()
    mask_root = (args.mask_root or stamp_code_dir / "playground" / "data" / "masks_new").expanduser().resolve()

    refcoco_dir = data_root / "annotations" / "refcoco_family"
    image_root = data_root / "shared" / "coco" / "train2014"

    if not refcoco_dir.exists():
        raise FileNotFoundError(f"RefCOCO annotation dir not found: {refcoco_dir}")
    if not image_root.exists():
        raise FileNotFoundError(f"COCO train2014 image root not found: {image_root}")

    output_json_dir.mkdir(parents=True, exist_ok=True)
    mask_root.mkdir(parents=True, exist_ok=True)

    # Upstream main_uni.py expects this file name. Keep it empty unless dialogue
    # data is prepared. Separate baseline/R-STAMP dirs each get their own copy.
    llava_placeholder = output_json_dir / "all_valid_llava_data_1000.json"
    if not llava_placeholder.exists():
        write_json(llava_placeholder, [])

    summaries = []
    for dataset_name in args.datasets:
        input_path = refcoco_dir / DATASET_FILES[dataset_name]
        if not input_path.exists():
            raise FileNotFoundError(f"Missing input file: {input_path}")
        summaries.append(
            convert_dataset(
                dataset_name=dataset_name,
                input_path=input_path,
                image_root=image_root,
                mask_root=mask_root,
                output_json_dir=output_json_dir,
                include_prior=args.include_prior,
                limit=args.limit,
                duplicate=args.duplicate,
            )
        )

    summary_path = output_json_dir / ("prepare_summary_with_prior.json" if args.include_prior else "prepare_summary_baseline.json")
    write_json(summary_path, summaries)
    print(json.dumps(summaries, indent=2, ensure_ascii=False))
    print(f"Summary written to: {summary_path}")
    print(f"JSON dir: {output_json_dir}")
    print(f"Mask root: {mask_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
