#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm


DEFAULT_ROOT = Path("/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_mask(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.asarray(values) > 0).astype(np.uint8) * 255, mode="L").save(path)


def annotation_mask(annotation: dict[str, Any], height: int, width: int) -> np.ndarray:
    segmentation = annotation.get("segmentation")
    if not segmentation:
        return np.zeros((height, width), dtype=bool)
    if isinstance(segmentation, list):
        image = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(image)
        for polygon in segmentation:
            points = [tuple(float(value) for value in polygon[index : index + 2]) for index in range(0, len(polygon), 2)]
            if len(points) >= 3:
                draw.polygon(points, fill=255)
        return np.asarray(image) > 0
    from pycocotools import mask as mask_utils

    rle = dict(segmentation)
    if isinstance(rle.get("counts"), str):
        rle["counts"] = rle["counts"].encode("utf-8")
    decoded = mask_utils.decode(rle)
    if decoded.ndim == 3:
        decoded = decoded.any(axis=2)
    return np.asarray(decoded, dtype=bool)


def stamp_item(
    image_path: Path,
    mask_path: Path,
    query: str,
    dataset: str,
    split: str,
    source_id: str | int,
    *,
    ignore_path: Path | None = None,
    no_target: bool = False,
) -> dict[str, Any]:
    item: dict[str, Any] = {
        "images": [str(image_path.resolve())],
        "masks": [str(mask_path.resolve())],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f'Please segment the object this sentence describes: "{query}".'},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": f'The "{query}" refers to <|seg|> in this image.'}],
            },
        ],
        "source_dataset": dataset,
        "source_split": split,
        "source_id": source_id,
        "label": query,
        "no_target": bool(no_target),
    }
    if ignore_path is not None:
        item["ignore_masks"] = [str(ignore_path.resolve())]
    return item


def convert_grefcoco(root: Path, splits: list[str], output_dir: Path, mask_root: Path) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_root.mkdir(parents=True, exist_ok=True)
    raw_root = root / "data" / "annotations" / "grefcoco"
    refs_path = raw_root / "grefs(unc).json"
    instances_path = raw_root / "instances.json"
    image_root = root / "data" / "shared" / "coco" / "train2014"
    if not refs_path.exists() or not instances_path.exists():
        raise FileNotFoundError(f"gRefCOCO files missing below {raw_root}")
    refs = load_json(refs_path)
    instances = load_json(instances_path)
    annotations = {int(value["id"]): value for value in instances["annotations"]}
    images = {int(value["id"]): value for value in instances["images"]}
    summaries = []
    for split in splits:
        items = []
        selected = [value for value in refs if str(value.get("split")) == split]
        for ref in tqdm(selected, desc=f"gRefCOCO {split}"):
            image_info = images[int(ref["image_id"])]
            image_path = image_root / str(image_info["file_name"])
            height, width = int(image_info["height"]), int(image_info["width"])
            target = np.zeros((height, width), dtype=bool)
            no_target = bool(ref.get("no_target"))
            if not no_target:
                for ann_id in ref.get("ann_id") or []:
                    if int(ann_id) >= 0:
                        target |= annotation_mask(annotations[int(ann_id)], height, width)
            mask_path = mask_root / "grefcoco" / split / f"ref_{int(ref['ref_id']):08d}.png"
            save_mask(mask_path, target)
            for sentence in ref.get("sentences") or []:
                query = str(sentence.get("sent") or sentence.get("raw") or "").strip()
                if not query:
                    continue
                source_id = f"{ref['ref_id']}_{sentence.get('sent_id', len(items))}"
                items.append(
                    stamp_item(
                        image_path,
                        mask_path,
                        query,
                        "grefcoco",
                        split,
                        source_id,
                        no_target=no_target,
                    )
                )
        output_path = output_dir / f"grefcoco_{split}.json"
        output_path.write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")
        summaries.append({"dataset": "grefcoco", "split": split, "samples": len(items), "output": str(output_path)})
    return summaries


def draw_reason_shape(draw: ImageDraw.ImageDraw, shape: dict[str, Any], fill: int) -> None:
    points = [tuple(float(v) for v in point[:2]) for point in shape.get("points") or []]
    if len(points) < 2:
        return
    if shape.get("shape_type") == "rectangle":
        draw.rectangle([points[0], points[1]], fill=fill)
    elif len(points) >= 3:
        draw.polygon(points, fill=fill)


def convert_reasonseg(root: Path, splits: list[str], output_dir: Path, mask_root: Path) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_root.mkdir(parents=True, exist_ok=True)
    raw_root = root / "data" / "datasets" / "reasonseg"
    summaries = []
    for split in splits:
        split_root = raw_root / split
        if not split_root.exists():
            raise FileNotFoundError(f"ReasonSeg split missing: {split_root}")
        items = []
        for json_path in tqdm(sorted(split_root.glob("*.json")), desc=f"ReasonSeg {split}"):
            value = load_json(json_path)
            image_path = next(
                (json_path.with_suffix(suffix) for suffix in (".jpg", ".jpeg", ".png") if json_path.with_suffix(suffix).exists()),
                None,
            )
            if image_path is None:
                continue
            with Image.open(image_path) as image_handle:
                width, height = image_handle.size
            target_image = Image.new("L", (width, height), 0)
            ignore_image = Image.new("L", (width, height), 0)
            target_draw = ImageDraw.Draw(target_image)
            ignore_draw = ImageDraw.Draw(ignore_image)
            for shape in value.get("shapes") or []:
                label = str(shape.get("label") or "").lower()
                if label == "target":
                    draw_reason_shape(target_draw, shape, 255)
                elif label == "ignore":
                    draw_reason_shape(ignore_draw, shape, 255)
            mask_path = mask_root / "reasonseg" / split / f"{json_path.stem}_target.png"
            ignore_path = mask_root / "reasonseg" / split / f"{json_path.stem}_ignore.png"
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            target_image.save(mask_path)
            ignore_image.save(ignore_path)
            texts = value.get("text") or []
            if isinstance(texts, str):
                texts = [texts]
            for text_index, query_value in enumerate(texts):
                query = str(query_value).strip()
                if query:
                    items.append(
                        stamp_item(
                            image_path,
                            mask_path,
                            query,
                            "reasonseg",
                            split,
                            f"{json_path.stem}_{text_index}",
                            ignore_path=ignore_path,
                        )
                    )
        output_path = output_dir / f"reasonseg_{split}.json"
        output_path.write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")
        summaries.append({"dataset": "reasonseg", "split": split, "samples": len(items), "output": str(output_path)})
    return summaries


def find_refclef_layout(raw_root: Path) -> tuple[Path, Path, Path]:
    refs_paths = list(raw_root.rglob("refs(unc).p"))
    if not refs_paths:
        raise FileNotFoundError("RefCLEF refs(unc).p not found; the HF yiqun/referit snapshot contains code only.")
    refs_path = refs_paths[0]
    instances_path = refs_path.parent / "instances.json"
    if not instances_path.exists():
        raise FileNotFoundError(f"RefCLEF instances.json missing beside {refs_path}")
    image_candidates = [path for path in raw_root.rglob("*") if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    if not image_candidates:
        raise FileNotFoundError("RefCLEF saiapr_tc-12 images are missing.")
    return refs_path, instances_path, raw_root


def convert_refclef(root: Path, splits: list[str], output_dir: Path, mask_root: Path) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_root.mkdir(parents=True, exist_ok=True)
    raw_root = root / "data" / "datasets" / "refclef_referit"
    refs_path, instances_path, image_root = find_refclef_layout(raw_root)
    with refs_path.open("rb") as handle:
        refs = pickle.load(handle, encoding="latin1")
    instances = load_json(instances_path)
    annotations = {int(value["id"]): value for value in instances["annotations"]}
    images = {int(value["id"]): value for value in instances["images"]}
    image_index = {
        path.name: path for path in image_root.rglob("*") if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    }
    summaries = []
    for split in splits:
        items = []
        selected = [value for value in refs if str(value.get("split")) == split]
        for ref in tqdm(selected, desc=f"RefCLEF {split}"):
            image_info = images[int(ref["image_id"])]
            image_path = image_index.get(Path(str(image_info["file_name"])).name)
            if image_path is None:
                continue
            with Image.open(image_path) as image_handle:
                width, height = image_handle.size
            target = annotation_mask(annotations[int(ref["ann_id"])], height, width)
            mask_path = mask_root / "refclef" / split / f"ref_{int(ref['ref_id']):08d}.png"
            save_mask(mask_path, target)
            for sentence in ref.get("sentences") or []:
                query = str(sentence.get("sent") or sentence.get("raw") or "").strip()
                if query:
                    items.append(
                        stamp_item(
                            image_path,
                            mask_path,
                            query,
                            "refclef",
                            split,
                            f"{ref['ref_id']}_{sentence.get('sent_id', len(items))}",
                        )
                    )
        output_path = output_dir / f"refclef_{split}.json"
        output_path.write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")
        summaries.append({"dataset": "refclef", "split": split, "samples": len(items), "output": str(output_path)})
    return summaries


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare STAMP evaluation JSONs for special segmentation datasets.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--dataset", choices=["grefcoco", "reasonseg", "refclef"], required=True)
    parser.add_argument("--splits", nargs="+", required=True)
    parser.add_argument("--output-json-dir", type=Path, required=True)
    parser.add_argument("--mask-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    args.output_json_dir.mkdir(parents=True, exist_ok=True)
    args.mask_root.mkdir(parents=True, exist_ok=True)
    converters = {
        "grefcoco": convert_grefcoco,
        "reasonseg": convert_reasonseg,
        "refclef": convert_refclef,
    }
    summaries = converters[args.dataset](root, args.splits, args.output_json_dir.resolve(), args.mask_root.resolve())
    summary_path = args.output_json_dir / f"prepare_{args.dataset}_summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summaries, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
