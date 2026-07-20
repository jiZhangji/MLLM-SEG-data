from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


PAPER_CIOU = {
    "base": {
        "refcoco|unc|val": 74.1,
        "refcoco|unc|testA": 76.5,
        "refcoco|unc|testB": 71.1,
        "refcoco+|unc|val": 62.4,
        "refcoco+|unc|testA": 67.4,
        "refcoco+|unc|testB": 56.5,
        "refcocog|umd|val": 66.4,
        "refcocog|umd|test": 68.5,
    },
    "finetuned_referseg": {
        "refcoco|unc|val": 74.9,
        "refcoco|unc|testA": 79.1,
        "refcoco|unc|testB": 72.3,
        "refcoco+|unc|val": 65.1,
        "refcoco+|unc|testA": 70.8,
        "refcoco+|unc|testB": 58.1,
        "refcocog|umd|val": 67.9,
        "refcocog|umd|test": 70.6,
    },
}

EXPECTED_EXPRESSIONS = {
    "refcoco|unc|val": 10834,
    "refcoco|unc|testA": 5657,
    "refcoco|unc|testB": 5095,
    "refcoco+|unc|val": 10758,
    "refcoco+|unc|testA": 5726,
    "refcoco+|unc|testB": 4889,
    "refcocog|umd|val": 4896,
    "refcocog|umd|test": 9602,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce LISA referring-segmentation results with the official "
            "ValDataset, REFER annotations, prompt, threshold, and cIoU/gIoU metrics."
        )
    )
    parser.add_argument("--lisa-code-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--vision-tower", type=Path, required=True)
    parser.add_argument("--sam-path", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--val-dataset", default="refcoco|unc|testA")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--paper-row", choices=tuple(PAPER_CIOU), default="finetuned_referseg")
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    parser.add_argument("--conv-type", choices=("llava_v1", "llava_llama_2"), default="llava_v1")
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--model-max-length", type=int, default=512)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit-images", type=int, default=0)
    parser.add_argument("--offset-images", type=int, default=0)
    parser.add_argument("--tolerance-points", type=float, default=0.5)
    parser.add_argument("--require-paper-match", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _dtype(precision: str) -> torch.dtype:
    return {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[precision]


def _validate_layout(args: argparse.Namespace) -> tuple[str, str, str]:
    parts = args.val_dataset.split("|")
    if len(parts) != 3:
        raise ValueError("--val-dataset must look like refcoco|unc|testA.")
    dataset, split_by, split = parts
    if args.val_dataset not in EXPECTED_EXPRESSIONS:
        raise ValueError(
            f"Unsupported paper split {args.val_dataset!r}; expected one of "
            f"{sorted(EXPECTED_EXPRESSIONS)}."
        )

    code_dir = args.lisa_code_dir.expanduser().resolve()
    model_path = args.model_path.expanduser().resolve()
    vision_tower = args.vision_tower.expanduser().resolve()
    sam_path = args.sam_path.expanduser().resolve()
    dataset_dir = args.dataset_dir.expanduser().resolve()
    required = {
        "LISA model code": code_dir / "model" / "LISA.py",
        "LISA official dataset code": code_dir / "utils" / "dataset.py",
        "LISA checkpoint config": model_path / "config.json",
        "CLIP config": vision_tower / "config.json",
        "SAM-H checkpoint": sam_path,
        "REFER split annotations": dataset_dir / dataset / f"refs({split_by}).p",
        "REFER instances": dataset_dir / dataset / "instances.json",
        "COCO train2014 images": dataset_dir / "images" / "mscoco" / "images" / "train2014",
    }
    missing = [f"{label}: {path}" for label, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Official LISA paper-protocol inputs are incomplete:\n" + "\n".join(missing))
    if not any(model_path.glob("*.bin")) and not any(model_path.glob("*.safetensors")):
        raise FileNotFoundError(f"No LISA weight shards were found below {model_path}.")
    if sam_path.stat().st_size < 2_000_000_000:
        raise ValueError(f"SAM-H checkpoint appears incomplete: {sam_path}")
    return dataset, split_by, split


def _artifact_paths(output_dir: Path, image_index: int, expression_index: int) -> dict[str, Path]:
    stem = f"image_{image_index:06d}_expr_{expression_index:04d}"
    return {
        "logits": output_dir / "pred_logits" / f"{stem}.npz",
        "mask": output_dir / "pred_masks" / f"{stem}.png",
        "gt": output_dir / "gt_masks" / f"{stem}.png",
    }


def _complete(paths: dict[str, Path]) -> bool:
    return all(path.is_file() and path.stat().st_size > 0 for path in paths.values())


def _save_mask(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    Image.fromarray(np.asarray(value, dtype=np.uint8) * 255, mode="L").save(
        temporary, format="PNG"
    )
    os.replace(temporary, path)


def _save_logits(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, logits=np.asarray(value, dtype=np.float16))
    os.replace(temporary, path)


def _load_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8) > 0


def official_lisa_metric_totals(
    predictions: list[np.ndarray], targets: list[np.ndarray]
) -> dict[str, float]:
    """Compute the foreground statistics used by LISA's official validate()."""
    if len(predictions) != len(targets):
        raise ValueError("predictions and targets must have the same length.")
    intersection = 0
    union = 0
    giou_sum = 0.0
    for prediction, target in zip(predictions, targets):
        prediction = np.asarray(prediction, dtype=bool)
        target = np.asarray(target, dtype=bool)
        if prediction.shape != target.shape:
            raise ValueError(f"Mask shape mismatch: {prediction.shape} vs {target.shape}.")
        item_intersection = int(np.logical_and(prediction, target).sum())
        item_union = int(np.logical_or(prediction, target).sum())
        intersection += item_intersection
        union += item_union
        giou_sum += item_intersection / item_union if item_union else 1.0
    count = len(predictions)
    return {
        "intersection": float(intersection),
        "union": float(union),
        "giou_sum": giou_sum,
        "count": float(count),
    }


def _git_revision(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    args = parse_args()
    if min(args.limit_images, args.offset_images, args.workers) < 0:
        raise ValueError("limits, offsets, and workers must be non-negative.")
    if args.tolerance_points < 0:
        raise ValueError("--tolerance-points must be non-negative.")
    if not torch.cuda.is_available():
        raise RuntimeError("Official LISA reproduction requires a CUDA GPU.")
    dataset_name, split_by, split = _validate_layout(args)

    code_dir = args.lisa_code_dir.expanduser().resolve()
    model_path = args.model_path.expanduser().resolve()
    vision_tower_path = args.vision_tower.expanduser().resolve()
    sam_path = args.sam_path.expanduser().resolve()
    dataset_dir = args.dataset_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(code_dir))
    from transformers import AutoTokenizer  # type: ignore

    from model.LISA import LISAForCausalLM  # type: ignore[import-not-found]
    from model.llava import conversation as conversation_lib  # type: ignore[import-not-found]
    from model.llava.model.language_model.llava_llama import (  # type: ignore[import-not-found]
        LlavaConfig,
    )
    from utils.dataset import ValDataset, collate_fn  # type: ignore[import-not-found]

    torch.set_grad_enabled(False)
    dtype = _dtype(args.precision)
    device = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
        local_files_only=True,
    )
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.add_tokens("[SEG]")
    seg_token_ids = tokenizer("[SEG]", add_special_tokens=False).input_ids
    if len(seg_token_ids) != 1:
        raise RuntimeError(f"Expected [SEG] to map to one token, got {seg_token_ids}.")

    from utils.utils import (  # type: ignore[import-not-found]
        DEFAULT_IM_END_TOKEN,
        DEFAULT_IM_START_TOKEN,
    )

    tokenizer.add_tokens(
        [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
    )
    config = LlavaConfig.from_pretrained(model_path, local_files_only=True)
    config.vision_tower = str(vision_tower_path)
    config.mm_vision_tower = str(vision_tower_path)
    model, loading_info = LISAForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        train_mask_decoder=True,
        out_dim=256,
        seg_token_idx=seg_token_ids[0],
        vision_pretrained=str(sam_path),
        vision_tower=str(vision_tower_path),
        use_mm_start_end=True,
        local_files_only=True,
        output_loading_info=True,
    )
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.resize_token_embeddings(len(tokenizer))
    model.get_model().initialize_vision_modules(model.get_model().config)
    model.to(device=device, dtype=dtype)
    model.get_model().get_vision_tower().to(device=device, dtype=dtype)
    model.eval()

    conversation_lib.default_conversation = conversation_lib.conv_templates[args.conv_type]
    dataset = ValDataset(
        str(dataset_dir),
        tokenizer,
        str(vision_tower_path),
        args.val_dataset,
        args.image_size,
    )
    start_index = min(args.offset_images, len(dataset))
    end_index = len(dataset)
    if args.limit_images:
        end_index = min(start_index + args.limit_images, end_index)
    image_indices = list(range(start_index, end_index))
    selected = Subset(dataset, image_indices)
    loader = DataLoader(
        selected,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False,
        collate_fn=partial(
            collate_fn,
            tokenizer=tokenizer,
            conv_type=args.conv_type,
            use_mm_start_end=True,
            local_rank=0,
        ),
    )

    totals = {"intersection": 0.0, "union": 0.0, "giou_sum": 0.0, "count": 0.0}
    rows: list[dict[str, Any]] = []
    model_seconds = 0.0
    model_calls = 0
    reused_expressions = 0
    exported_expressions = 0

    for image_index, batch in tqdm(
        zip(image_indices, loader),
        total=len(image_indices),
        desc=f"LISA official {dataset_name} {split}",
        dynamic_ncols=True,
    ):
        image_path = Path(batch["image_paths"][0]).resolve()
        expression_count = len(batch["conversation_list"])
        paths_list = [
            _artifact_paths(output_dir, image_index, expression_index)
            for expression_index in range(expression_count)
        ]
        can_reuse = not args.overwrite and all(_complete(paths) for paths in paths_list)

        if can_reuse:
            predictions = [_load_mask(paths["mask"]) for paths in paths_list]
            targets = [_load_mask(paths["gt"]) for paths in paths_list]
            reused_expressions += expression_count
        else:
            for key, value in list(batch.items()):
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(device, non_blocking=True)
                elif value and isinstance(value, list) and isinstance(value[0], torch.Tensor):
                    batch[key] = [item.to(device, non_blocking=True) for item in value]
            batch["images"] = batch["images"].to(dtype=dtype)
            batch["images_clip"] = batch["images_clip"].to(dtype=dtype)
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.inference_mode():
                output = model(**batch)
            torch.cuda.synchronize()
            model_seconds += time.perf_counter() - started
            model_calls += 1

            if len(output["pred_masks"]) != 1:
                raise RuntimeError("Official validation batch must return one image mask group.")
            logits_batch = output["pred_masks"][0].detach().float().cpu().numpy()
            target_batch = output["gt_masks"][0].detach().cpu().numpy()
            if len(logits_batch) != expression_count or len(target_batch) != expression_count:
                raise RuntimeError(
                    f"Expected {expression_count} expression masks, got "
                    f"{len(logits_batch)} predictions and {len(target_batch)} targets."
                )
            predictions = []
            targets = []
            for paths, logits, target in zip(paths_list, logits_batch, target_batch):
                prediction = np.asarray(logits > 0.0, dtype=bool)
                target_mask = np.asarray(target > 0, dtype=bool)
                _save_logits(paths["logits"], logits)
                _save_mask(paths["mask"], prediction)
                _save_mask(paths["gt"], target_mask)
                predictions.append(prediction)
                targets.append(target_mask)
            exported_expressions += expression_count

        item_totals = official_lisa_metric_totals(predictions, targets)
        for key in totals:
            totals[key] += item_totals[key]
        for expression_index, paths in enumerate(paths_list):
            rows.append(
                {
                    "name": f"image_{image_index:06d}_expr_{expression_index:04d}",
                    "method": "LISA-7B",
                    "split": args.val_dataset,
                    "instance_id": f"{image_index}:{expression_index}",
                    "image": str(image_path),
                    "gt_mask": str(paths["gt"]),
                    "prediction": str(paths["logits"]),
                    "prediction_kind": "logits",
                    "array_key": "logits",
                    "threshold": 0.5,
                    "pred_mask": str(paths["mask"]),
                    "protocol": "lisa_official_refer_valdataset",
                }
            )

    manifest_path = output_dir / "manifest.jsonl"
    _write_jsonl(manifest_path, rows)
    samples = int(totals["count"])
    giou = totals["giou_sum"] / max(samples, 1)
    ciou = totals["intersection"] / max(totals["union"], 1.0)
    full_run = args.limit_images == 0 and args.offset_images == 0
    expected_samples = EXPECTED_EXPRESSIONS[args.val_dataset]
    expected_ciou = PAPER_CIOU[args.paper_row][args.val_dataset]
    gap_points = ciou * 100.0 - expected_ciou
    sample_count_matches = samples == expected_samples if full_run else None
    paper_match = (
        bool(sample_count_matches) and abs(gap_points) <= args.tolerance_points
        if full_run
        else None
    )
    summary = {
        "source": "lisa_official_paper_protocol",
        "paper_row": args.paper_row,
        "val_dataset": args.val_dataset,
        "dataset": dataset_name,
        "split_by": split_by,
        "split": split,
        "images": len(image_indices),
        "samples": samples,
        "expected_full_samples": expected_samples,
        "sample_count_matches": sample_count_matches,
        "gIoU": giou,
        "cIoU": ciou,
        "gIoU_percent": giou * 100.0,
        "cIoU_percent": ciou * 100.0,
        "paper_cIoU_percent": expected_ciou,
        "cIoU_gap_points": gap_points,
        "tolerance_points": args.tolerance_points,
        "paper_match": paper_match,
        "model": str(model_path),
        "vision_tower": str(vision_tower_path),
        "sam_path": str(sam_path),
        "dataset_dir": str(dataset_dir),
        "lisa_code_revision": _git_revision(code_dir),
        "precision": args.precision,
        "model_calls": model_calls,
        "exported_expressions": exported_expressions,
        "reused_expressions": reused_expressions,
        "model_seconds": model_seconds,
        "model_seconds_per_new_expression": model_seconds / max(exported_expressions, 1),
        "loading_info": {
            key: [str(item) for item in loading_info.get(key, [])]
            for key in ("missing_keys", "unexpected_keys", "mismatched_keys")
        },
        "manifest": str(manifest_path),
    }
    summary_path = output_dir / "paper_reproduction_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

    if full_run and not sample_count_matches:
        print(
            f"ERROR: official split produced {samples} expressions; expected {expected_samples}.",
            file=sys.stderr,
        )
        return 2
    if full_run and args.require_paper_match and not paper_match:
        print(
            f"ERROR: cIoU is {ciou * 100.0:.2f}, paper target is {expected_ciou:.2f} "
            f"(gap {gap_points:+.2f} points). FreeRef must not be compared yet.",
            file=sys.stderr,
        )
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
