#!/usr/bin/env python3
"""Prepare RefCOCO-family official split evaluation JSONs for STAMP.

Unlike the training-data helper, this script supports val/test splits so we can
evaluate on splits that are closer to the numbers reported in STAMP/Text4Seg
papers. It can create:

- baseline JSONs: original query only;
- text-only prior JSONs: no GT bbox/center in `structured_prior_text`;
- oracle prior JSONs: optional upper-bound with GT bbox/center.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
from tqdm import tqdm


DEFAULT_ROOT = Path("/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG")

SPLIT_FILES = {
    "refcoco_val": ("refcoco", "refcoco_val.json"),
    "refcoco_testA": ("refcoco", "refcoco_testA.json"),
    "refcoco_testB": ("refcoco", "refcoco_testB.json"),
    "refcoco+_val": ("refcoco+", "refcoco+_val.json"),
    "refcoco+_testA": ("refcoco+", "refcoco+_testA.json"),
    "refcoco+_testB": ("refcoco+", "refcoco+_testB.json"),
    "refcocog_val": ("refcocog", "refcocog_val.json"),
    "refcocog_test": ("refcocog", "refcocog_test.json"),
}


def load_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
            if limit and len(items) >= limit:
                break
    return items


def extract_expression(sample: dict[str, Any]) -> str:
    conversations = sample.get("conversations") or []
    if conversations:
        text = str(conversations[0].get("value", ""))
        match = re.search(r'"([^"]+)"', text)
        if match:
            return match.group(1).strip()
        return text.strip()
    return ""


def extract_query(sample: dict[str, Any]) -> str:
    objects = sample.get("objects") or []
    if objects and objects[0].get("label"):
        return str(objects[0]["label"]).strip()
    return extract_expression(sample)


def decode_rle_to_mask(rle: dict[str, Any]) -> np.ndarray:
    rle_obj = dict(rle)
    counts = rle_obj.get("counts")
    if isinstance(counts, str):
        rle_obj["counts"] = counts.encode("utf-8")
    mask = mask_utils.decode(rle_obj)
    if mask.ndim == 3:
        mask = np.sum(mask, axis=2)
    return (mask > 0).astype(np.uint8)


def format_prior(sample: dict[str, Any], query: str, obj: dict[str, Any], prior_mode: str) -> tuple[dict[str, Any], str] | None:
    if prior_mode == "none":
        return None
    if prior_mode == "text_only":
        expression = extract_expression(sample) or query
        return (
            {
                "mode": "text_only",
                "target": expression,
                "bbox": None,
                "uses_gt_geometry": False,
            },
            f"<TARGET> {expression} </TARGET>",
        )
    if prior_mode == "oracle":
        bbox = obj.get("bbox")
        parts = [f"<TARGET> {query} </TARGET>"]
        if bbox and len(bbox) >= 4:
            x1, y1, x2, y2 = [float(x) for x in bbox[:4]]
            parts.append(f"<BOX> {x1:.4f} {y1:.4f} {x2:.4f} {y2:.4f} </BOX>")
            parts.append(f"<POS> {(x1 + x2) / 2:.4f},{(y1 + y2) / 2:.4f} </POS>")
        return (
            {
                "mode": "oracle",
                "target": query,
                "bbox": bbox,
                "uses_gt_geometry": True,
            },
            " ".join(parts),
        )
    raise ValueError(f"Unsupported prior_mode: {prior_mode}")


def make_item(
    sample: dict[str, Any],
    image_root: Path,
    mask_root: Path,
    split_name: str,
    idx: int,
    prior_mode: str,
) -> dict[str, Any] | None:
    objects = sample.get("objects") or []
    if not objects:
        return None
    obj = objects[0]
    rle = obj.get("rle")
    image_name = sample.get("image")
    if not rle or not image_name:
        return None
    image_path = image_root / image_name
    if not image_path.exists():
        return None
    query = extract_query(sample)
    if not query:
        return None

    mask_arr = decode_rle_to_mask(rle)
    safe_split = split_name.replace("+", "plus")
    mask_path = mask_root / safe_split / f"{safe_split}_{sample.get('id', idx)}_{idx}.png"
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask_arr * 255).save(mask_path)

    question = f'Please segment the object this sentence describes: "{query}".'
    answer = f'The "{query}" refers to <|seg|> in this image.'
    item: dict[str, Any] = {
        "images": [str(image_path)],
        "masks": [str(mask_path)],
        "messages": [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]},
            {"role": "assistant", "content": [{"type": "text", "text": answer}]},
        ],
        "source_split": split_name,
        "source_id": sample.get("id"),
        "label": query,
        "bbox": obj.get("bbox"),
    }
    prior = format_prior(sample, query, obj, prior_mode)
    if prior:
        prior_dict, prior_text = prior
        item["structured_prior"] = prior_dict
        item["structured_prior_text"] = prior_text
    return item


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-json-dir", type=Path, required=True)
    parser.add_argument("--mask-root", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["refcocog_val", "refcocog_test"], choices=sorted(SPLIT_FILES))
    parser.add_argument("--prior-mode", choices=["none", "text_only", "oracle"], default="none")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    root = args.root.expanduser().resolve()
    ann_root = root / "data" / "annotations" / "refcoco_family"
    image_root = root / "data" / "shared" / "coco" / "train2014"
    output_json_dir = args.output_json_dir.expanduser().resolve()
    mask_root = args.mask_root.expanduser().resolve()
    output_json_dir.mkdir(parents=True, exist_ok=True)
    mask_root.mkdir(parents=True, exist_ok=True)

    summaries = []
    for split in args.splits:
        _dataset_name, file_name = SPLIT_FILES[split]
        input_path = ann_root / file_name
        raw = load_jsonl(input_path, limit=args.limit)
        items = []
        skipped = 0
        for idx, sample in tqdm(list(enumerate(raw)), desc=split):
            item = make_item(sample, image_root, mask_root, split, idx, args.prior_mode)
            if item is None:
                skipped += 1
                continue
            items.append(item)
        out_path = output_json_dir / f"{split}.json"
        write_json(out_path, items)
        summaries.append(
            {
                "split": split,
                "prior_mode": args.prior_mode,
                "input": str(input_path),
                "output": str(out_path),
                "raw_items": len(raw),
                "converted_items": len(items),
                "skipped": skipped,
            }
        )

    write_json(output_json_dir / f"prepare_eval_summary_{args.prior_mode}.json", summaries)
    print(json.dumps(summaries, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
