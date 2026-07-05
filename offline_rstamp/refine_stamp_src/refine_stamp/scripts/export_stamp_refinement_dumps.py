from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


DEFAULT_ROOT = Path("/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG")


def setup_stamp_imports(stamp_code_dir: Path) -> None:
    stamp_code_dir = stamp_code_dir.resolve()
    if str(stamp_code_dir) not in sys.path:
        sys.path.insert(0, str(stamp_code_dir))


def load_segmenter_class(stamp_code_dir: Path):
    setup_stamp_imports(stamp_code_dir)
    try:
        from segment_predictor import GenerativeSegmenter  # type: ignore
    except Exception:
        from segment_predictor_cache import GenerativeSegmenter  # type: ignore
    return GenerativeSegmenter


def load_official_question_templates(stamp_code_dir: Path) -> list[str]:
    setup_stamp_imports(stamp_code_dir)
    try:
        from data.question_answer_list import QUESTION_PARTIAL  # type: ignore

        templates = [str(x) for x in QUESTION_PARTIAL if "[class_name]" in str(x)]
        if templates:
            return templates
    except Exception:
        pass
    return ["Please segment the [class_name] in this image."]


def load_items(path: Path, limit: int) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise TypeError(f"Expected a JSON list: {path}")
    if limit > 0:
        data = data[:limit]
    return data


def extract_text_from_messages(item: dict[str, Any]) -> str:
    messages = item.get("messages") or item.get("conversations") or []
    if not messages:
        return item.get("query") or item.get("text") or ""
    content = messages[0].get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                texts.append(str(part.get("text", "")))
            elif isinstance(part, str):
                texts.append(part)
        return "\n".join(x for x in texts if x)
    return str(content)


def extract_target_text(item: dict[str, Any]) -> str:
    label = str(item.get("label") or "").strip()
    if label:
        return label
    text = extract_text_from_messages(item).strip()
    import re

    for pattern in [
        r'Please segment the object this sentence describes:\s*"([^"]+)"',
        r'The "([^"]+)" refers to',
        r'"([^"]+)"',
    ]:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
    return text


def build_query(item: dict[str, Any], prompt_mode: str, official_templates: list[str]) -> str:
    if prompt_mode == "official":
        target = extract_target_text(item)
        return official_templates[0].replace("[class_name]", target).strip()
    if prompt_mode == "target_only":
        return extract_target_text(item)
    return extract_text_from_messages(item).strip()


def first_existing_path(paths: list[str], root: Path | None = None) -> Path:
    for raw in paths:
        p = Path(raw)
        if p.exists():
            return p
        if root is not None:
            q = root / raw
            if q.exists():
                return q
    raise FileNotFoundError(f"No existing path found among: {paths}")


def get_image_path(item: dict[str, Any], root: Path) -> Path:
    values = item.get("images") or item.get("image") or item.get("image_path")
    candidates = [str(x) for x in values] if isinstance(values, list) else [str(values)]
    return first_existing_path(candidates, root)


def get_mask_path(item: dict[str, Any], root: Path) -> Path:
    values = item.get("masks") or item.get("mask") or item.get("mask_path")
    candidates = [str(x) for x in values] if isinstance(values, list) else [str(values)]
    return first_existing_path(candidates, root)


def normalize_refinement_outputs(outputs: Any) -> dict[str, Any]:
    if isinstance(outputs, tuple):
        # Allow segmenter APIs that return (masks, response_text, refinement_dict)
        for value in reversed(outputs):
            if isinstance(value, dict) and {"mask_logits", "mask_hidden", "grid_hw"} <= set(value):
                outputs = value
                break
    if not isinstance(outputs, dict):
        raise TypeError(f"Refinement output must be a dict, got {type(outputs).__name__}")

    required = {"mask_logits", "mask_hidden", "grid_hw"}
    missing = required - set(outputs)
    if missing:
        raise KeyError(f"Missing refinement keys: {sorted(missing)}")

    mask_logits = outputs["mask_logits"].detach().float().cpu()
    mask_hidden = outputs["mask_hidden"].detach().float().cpu()
    grid_hw = tuple(int(x) for x in outputs["grid_hw"])
    if mask_logits.ndim != 3 or mask_logits.shape[-1] != 2:
        raise ValueError(f"mask_logits must be [B, N, 2], got {tuple(mask_logits.shape)}")
    if mask_hidden.ndim != 3:
        raise ValueError(f"mask_hidden must be [B, N, D], got {tuple(mask_hidden.shape)}")
    if mask_logits.shape[:2] != mask_hidden.shape[:2]:
        raise ValueError("mask_logits and mask_hidden do not share [B, N].")
    if grid_hw[0] * grid_hw[1] != mask_logits.shape[1]:
        raise ValueError(f"grid_hw={grid_hw} does not match N={mask_logits.shape[1]}")

    out = dict(outputs)
    out["mask_logits"] = mask_logits
    out["mask_hidden"] = mask_hidden
    out["grid_hw"] = grid_hw
    return out


def call_refinement_export(segmenter: Any, image: Image.Image, query: str) -> dict[str, Any]:
    """Call the first available STAMP refinement export method.

    The local tool repo cannot know the exact STAMP checkout internals, so this
    supports several likely adapter names. If none exists, patch STAMP Phase 2
    to add one of these methods.
    """

    candidates = [
        "generate_with_refinement_outputs",
        "generate_with_refinement",
        "export_refinement_outputs",
        "forward_for_refinement",
    ]
    for name in candidates:
        fn = getattr(segmenter, name, None)
        if callable(fn):
            try:
                return normalize_refinement_outputs(fn(image, query))
            except TypeError:
                return normalize_refinement_outputs(fn(images=[image], texts=[query]))

    raise AttributeError(
        "GenerativeSegmenter does not expose a refinement export method. "
        "Add one of: generate_with_refinement_outputs(image, query), "
        "generate_with_refinement(image, query), export_refinement_outputs(image, query), "
        "or forward_for_refinement(images=[...], texts=[...]). It must return "
        "mask_logits [B,N,2], mask_hidden [B,N,D], and grid_hw."
    )


def export_dumps(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root.expanduser().resolve()
    stamp_code_dir = args.stamp_code_dir.expanduser().resolve()
    model_path = args.model.expanduser().resolve()
    json_path = args.json.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if os.environ.get("STAMP_DISABLE_CUDNN", "0") == "1":
        torch.backends.cudnn.enabled = False

    GenerativeSegmenter = load_segmenter_class(stamp_code_dir)
    official_templates = load_official_question_templates(stamp_code_dir)
    items = load_items(json_path, args.limit)
    output_dir.mkdir(parents=True, exist_ok=True)

    segmenter = GenerativeSegmenter(
        str(model_path),
        device_map=args.device_map,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )

    exported = []
    failed = []
    for index, item in enumerate(tqdm(items, desc="export_refinement_dumps")):
        image_path = get_image_path(item, root)
        mask_path = get_mask_path(item, root)
        query = build_query(item, args.prompt_mode, official_templates)
        image = Image.open(image_path).convert("RGB")
        try:
            with torch.inference_mode():
                outputs = call_refinement_export(segmenter, image, query)
            dump = {
                "name": f"{args.split_name}_{index:06d}",
                "index": index,
                "query": query,
                "image_path": str(image_path),
                "mask_path": str(mask_path),
                "source_item": {
                    "label": item.get("label"),
                    "source_id": item.get("source_id"),
                    "source_split": item.get("source_split"),
                },
                "mask_logits": outputs["mask_logits"],
                "mask_hidden": outputs["mask_hidden"],
                "grid_hw": outputs["grid_hw"],
            }
            out_path = output_dir / f"{args.split_name}_{index:06d}.pt"
            torch.save(dump, out_path)
            exported.append(str(out_path))
        except Exception as exc:
            failed.append({"index": index, "image": str(image_path), "error": f"{type(exc).__name__}: {exc}"})
            if args.fail_fast:
                raise

    return {
        "model": str(model_path),
        "json": str(json_path),
        "output_dir": str(output_dir),
        "num_items": len(items),
        "num_exported": len(exported),
        "num_failed": len(failed),
        "exported": exported,
        "failed": failed,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--stamp-code-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-name", default="refcocog_val")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--min-pixels", type=int, default=802816)
    parser.add_argument("--max-pixels", type=int, default=1003520)
    parser.add_argument("--prompt-mode", choices=["prepared", "official", "target_only"], default="official")
    parser.add_argument("--device-map", default="cuda")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    report = export_dumps(args)
    report_path = args.output_dir / "export_refinement_dumps_summary.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["num_exported"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
