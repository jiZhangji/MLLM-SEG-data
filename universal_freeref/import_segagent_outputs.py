from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from .export_utils import artifact_paths, atomic_save_mask, atomic_write_json, atomic_write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert SegAgent's official trajectory JSON to a paired FreeRef mask manifest."
    )
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--method", default="SegAgent-Qwen7B-SimpleClick")
    parser.add_argument(
        "--selection",
        choices=("final", "official-reported-best", "click-index"),
        default="final",
        help="final matches the mask returned by SegAgent's main inference loop.",
    )
    parser.add_argument("--click-index", type=int, default=-1)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--allow-missing-images", action="store_true")
    return parser.parse_args()


def decode_rle(value: Any, height: int, width: int) -> np.ndarray:
    from pycocotools import mask as mask_utils  # type: ignore

    if value is None:
        return np.zeros((height, width), dtype=bool)
    if isinstance(value, list):
        rles = mask_utils.frPyObjects(value, height, width)
        encoded = mask_utils.merge(rles)
    elif isinstance(value, dict) and isinstance(value.get("counts"), list):
        encoded = mask_utils.frPyObjects(value, height, width)
    else:
        encoded = value
    result = np.asarray(mask_utils.decode(encoded))
    if result.ndim == 3:
        result = result.any(axis=2)
    if result.shape != (height, width):
        raise ValueError(f"Decoded RLE has shape {result.shape}, expected {(height, width)}.")
    return result.astype(bool)


def reported_iou(output: Any) -> float:
    if output is None:
        return 0.0
    match = re.search(r"Current IOU:\s*([-+]?[0-9]*\.?[0-9]+)", str(output))
    return float(match.group(1)) if match else 0.0


def select_prediction(sample: dict[str, Any], selection: str, click_index: int) -> tuple[Any, int]:
    predictions = sample.get("pred_list") or []
    if not predictions:
        return None, -1
    if selection == "final":
        selected = len(predictions) - 1
    elif selection == "click-index":
        selected = click_index if click_index >= 0 else len(predictions) + click_index
        if not 0 <= selected < len(predictions):
            raise IndexError(f"click-index {click_index} is invalid for {len(predictions)} predictions.")
    else:
        best = int(np.argmax([reported_iou(value.get("outputs")) for value in predictions]))
        # Preserve the released eval_result_iou.py convention: the reported IoU
        # describes the action after the preceding mask state.
        selected = best - 1 if best > 0 else 0
    return predictions[selected].get("mask"), selected


def resolve_image(raw_path: str, image_root: Path | None) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_file():
        return path.resolve()
    if image_root is not None:
        candidates = [image_root / raw_path, image_root / path.name]
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
    return path


def main() -> int:
    args = parse_args()
    input_path = args.input_json.expanduser().resolve()
    values = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(values, list):
        raise TypeError(f"Expected a JSON list from SegAgent: {input_path}")
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_root = args.image_root.expanduser().resolve() if args.image_root else None

    rows: list[dict[str, Any]] = []
    missing_images: list[str] = []
    empty_predictions = 0
    selected_clicks: list[int] = []
    for index, sample in enumerate(values):
        if not isinstance(sample, dict):
            raise TypeError(f"SegAgent item {index} is not an object.")
        height = int(sample["height"])
        width = int(sample["width"])
        image = resolve_image(str(sample["img_path"]), image_root)
        if not image.is_file():
            missing_images.append(str(image))
            continue
        prediction_rle, selected_click = select_prediction(sample, args.selection, args.click_index)
        prediction = decode_rle(prediction_rle, height, width)
        target = decode_rle(sample.get("gt_mask"), height, width)
        paths = artifact_paths(args.output_dir, index)
        atomic_save_mask(paths["mask"], prediction)
        atomic_save_mask(paths["gt"], target)
        empty_predictions += int(not prediction.any())
        selected_clicks.append(selected_click)
        caption_value = sample.get("caption") or ""
        caption = str(caption_value[0] if isinstance(caption_value, list) and caption_value else caption_value)
        rows.append(
            {
                "name": f"segagent_{index:08d}",
                "method": args.method,
                "split": args.split,
                "instance_id": str(index),
                "image": str(image),
                "gt_mask": str(paths["gt"]),
                "prediction": str(paths["mask"]),
                "prediction_kind": "mask",
                "threshold": 0.5,
                "query": caption,
                "selected_click": selected_click,
                "protocol": "segagent_official_simpleclick_final_mask_import",
            }
        )
    if missing_images and not args.allow_missing_images:
        raise FileNotFoundError(
            f"{len(missing_images)} SegAgent image paths cannot be resolved; examples: {missing_images[:5]}. "
            "Pass --image-root for relocated COCO images."
        )
    if not rows:
        raise ValueError("No SegAgent predictions were converted.")
    manifest = args.output_dir / "manifest.jsonl"
    atomic_write_jsonl(manifest, rows)
    report = {
        "source": "official_segagent_trajectory_json",
        "samples": len(rows),
        "split": args.split,
        "method": args.method,
        "selection": args.selection,
        "empty_predictions": empty_predictions,
        "missing_images": len(missing_images),
        "selected_click_min": min(selected_clicks),
        "selected_click_max": max(selected_clicks),
        "manifest": str(manifest.resolve()),
    }
    atomic_write_json(args.output_dir / "import_summary.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
