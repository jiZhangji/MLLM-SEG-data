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


def load_items(path: Path, limit: int, offset: int = 0) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise TypeError(f"Expected a JSON list: {path}")
    if offset < 0:
        raise ValueError("offset must be non-negative.")
    if offset > 0:
        data = data[offset:]
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


def get_ignore_path(item: dict[str, Any], root: Path) -> Path | None:
    values = item.get("ignore_masks") or item.get("ignore_mask") or item.get("ignore_path")
    if values is None:
        return None
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
    if mask_logits.ndim == 4 and mask_logits.shape[-2] == 1 and mask_logits.shape[-1] == 2:
        mask_logits = mask_logits.squeeze(-2)
    if mask_hidden.ndim == 4 and mask_hidden.shape[-2] == 1:
        mask_hidden = mask_hidden.squeeze(-2)
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


def call_batch_refinement_export(
    segmenter: Any,
    images: list[Image.Image],
    queries: list[str],
) -> list[dict[str, Any] | Exception]:
    fn = getattr(segmenter, "generate_batch_with_refinement_outputs", None)
    if not callable(fn):
        return [call_refinement_export(segmenter, image, query) for image, query in zip(images, queries)]

    outputs = fn(images, queries)
    if not isinstance(outputs, (list, tuple)) or len(outputs) != len(images):
        raise TypeError(f"Batch refinement export returned {type(outputs).__name__}, expected {len(images)} items.")

    normalized: list[dict[str, Any] | Exception] = []
    for output in outputs:
        if isinstance(output, dict) and output.get("batch_export_error"):
            normalized.append(RuntimeError(str(output["batch_export_error"])))
            continue
        try:
            normalized.append(normalize_refinement_outputs(output))
        except Exception as exc:
            normalized.append(exc)
    return normalized


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
    items = load_items(json_path, args.limit, args.offset)
    output_dir.mkdir(parents=True, exist_ok=True)
    segmenter = GenerativeSegmenter(
        str(model_path),
        device_map=args.device_map,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )

    exported: list[str] = []
    skipped: list[str] = []
    failed: list[dict[str, Any]] = []
    empty_predictions: list[dict[str, Any]] = []
    batch_calls = 0
    batch_splits = 0
    single_retries = 0
    adaptive_batch_size = args.batch_size

    def infer_prepared(prepared: list[dict[str, Any]]) -> list[dict[str, Any] | Exception]:
        nonlocal adaptive_batch_size, batch_calls, batch_splits
        if len(prepared) == 1 or args.batch_size == 1:
            try:
                return [call_refinement_export(segmenter, prepared[0]["image"], prepared[0]["query"])]
            except Exception as exc:
                return [exc]
        try:
            batch_calls += 1
            return call_batch_refinement_export(
                segmenter,
                [entry["image"] for entry in prepared],
                [entry["query"] for entry in prepared],
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            batch_splits += 1
            midpoint = len(prepared) // 2
            adaptive_batch_size = min(adaptive_batch_size, max(1, midpoint))
            print(f"WARNING: CUDA OOM at batch {len(prepared)}; retrying as {midpoint}+{len(prepared)-midpoint}.")
            return infer_prepared(prepared[:midpoint]) + infer_prepared(prepared[midpoint:])
        except Exception as exc:
            batch_splits += 1
            midpoint = len(prepared) // 2
            print(f"WARNING: batch export failed at size {len(prepared)} ({type(exc).__name__}: {exc}); splitting.")
            return infer_prepared(prepared[:midpoint]) + infer_prepared(prepared[midpoint:])

    progress = tqdm(total=len(items), desc="export_refinement_dumps")
    batch_start = 0
    while batch_start < len(items):
        prepared: list[dict[str, Any]] = []
        batch_items = items[batch_start : batch_start + adaptive_batch_size]
        for batch_offset, item in enumerate(batch_items):
            local_index = batch_start + batch_offset
            index = args.offset + local_index
            out_path = output_dir / f"{args.split_name}_{index:06d}.pt"
            if out_path.exists() and not args.overwrite:
                skipped.append(str(out_path))
                progress.update(1)
                continue
            try:
                image_path = get_image_path(item, root)
                mask_path = get_mask_path(item, root)
                ignore_path = get_ignore_path(item, root)
                query = build_query(item, args.prompt_mode, official_templates)
                image = Image.open(image_path).convert("RGB")
                dump = {
                    "name": f"{args.split_name}_{index:06d}",
                    "index": index,
                    "query": query,
                    "image_path": str(image_path),
                    "mask_path": str(mask_path),
                    "ignore_path": str(ignore_path) if ignore_path is not None else None,
                    "source_item": {
                        "label": item.get("label"),
                        "source_id": item.get("source_id"),
                        "source_split": item.get("source_split"),
                        "source_dataset": item.get("source_dataset"),
                        "no_target": bool(item.get("no_target", False)),
                    },
                }
                prepared.append(
                    {
                        "index": index,
                        "out_path": out_path,
                        "image_path": image_path,
                        "image": image,
                        "query": query,
                        "dump": dump,
                    }
                )
            except Exception as exc:
                failed.append(
                    {"index": index, "image": str(item.get("image", "")), "error": f"{type(exc).__name__}: {exc}"}
                )
                progress.update(1)
                if args.fail_fast:
                    progress.close()
                    raise

        if not prepared:
            batch_start += len(batch_items)
            continue
        with torch.inference_mode():
            results = infer_prepared(prepared)

        for entry, result in zip(prepared, results):
            if isinstance(result, Exception) and args.batch_size > 1:
                single_retries += 1
                try:
                    with torch.inference_mode():
                        result = call_refinement_export(segmenter, entry["image"], entry["query"])
                except Exception as exc:
                    result = exc

            dump = entry["dump"]
            out_path = entry["out_path"]
            if not isinstance(result, Exception):
                dump.update(
                    {
                        "mask_logits": result["mask_logits"],
                        "mask_hidden": result["mask_hidden"],
                        "grid_hw": result["grid_hw"],
                        "response_text": result.get("response_text"),
                        "requested_batch_size": args.batch_size,
                    }
                )
                torch.save(dump, out_path)
                exported.append(str(out_path))
                progress.update(1)
                continue

            error = f"{type(result).__name__}: {result}"
            if args.empty_on_failure:
                dump.update(
                    {
                        "mask_logits": torch.tensor([[[20.0, -20.0]]], dtype=torch.float32),
                        "mask_hidden": torch.zeros((1, 1, 1), dtype=torch.float32),
                        "grid_hw": (1, 1),
                        "prediction_error": error,
                        "empty_prediction": True,
                    }
                )
                torch.save(dump, out_path)
                exported.append(str(out_path))
                empty_predictions.append({"index": entry["index"], "image": str(entry["image_path"]), "error": error})
            else:
                failed.append({"index": entry["index"], "image": str(entry["image_path"]), "error": error})
                if args.fail_fast:
                    progress.close()
                    raise result
            progress.update(1)
        batch_start += len(batch_items)
    progress.close()

    return {
        "model": str(model_path),
        "json": str(json_path),
        "output_dir": str(output_dir),
        "num_items": len(items),
        "offset": args.offset,
        "limit": args.limit,
        "batch_size": args.batch_size,
        "effective_batch_size": adaptive_batch_size,
        "num_batch_calls": batch_calls,
        "num_batch_splits": batch_splits,
        "num_single_retries": single_retries,
        "num_exported": len(exported),
        "num_skipped": len(skipped),
        "num_failed": len(failed),
        "num_empty_predictions": len(empty_predictions),
        "exported": exported,
        "skipped": skipped,
        "failed": failed,
        "empty_predictions": empty_predictions,
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
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--min-pixels", type=int, default=802816)
    parser.add_argument("--max-pixels", type=int, default=1003520)
    parser.add_argument("--prompt-mode", choices=["prepared", "official", "target_only"], default="official")
    parser.add_argument("--device-map", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--empty-on-failure",
        action="store_true",
        help="Record a valid empty prediction when STAMP emits no mask; required for generalized/no-target evaluation.",
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1.")

    report = export_dumps(args)
    report_path = args.output_dir / "export_refinement_dumps_summary.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    completed = report["num_exported"] + report["num_skipped"]
    if completed == 0:
        print("")
        print("No refinement dumps were exported.")
        print("Most likely STAMP Phase-2 export still needs a small compatibility fix.")
        print("Run this inspection helper on the server and send the generated source snippets back:")
        print("")
        print("  bash offline_rstamp/run/77_inspect_stamp_refinement_points.sh")
        print("")
        print("Key files to send:")
        print("  $ROOT/outputs/refine_stamp_phase2_inspection/GenerativeSegmenter_source.py.txt")
        print("  $ROOT/outputs/refine_stamp_phase2_inspection/generate_with_segmentation_source.py.txt")
        print("  $ROOT/outputs/refine_stamp_phase2_inspection/stamp_refinement_inspection.json")
    if report["num_exported"] == 0 and report["num_skipped"] > 0:
        print(f"All {report['num_skipped']} existing dumps were reused.")
    return 0 if completed > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
