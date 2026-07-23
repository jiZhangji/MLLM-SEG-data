from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from .eval_stamp_dumps import boundary_iou, mask_iou
from .export_text4seg_masks import compute_sam_logits, sample_mask_points
from .refiner import TrainingFreeRefineConfig, TrainingFreeUncertaintyRefiner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen SAM-H after public Text4Seg coarse and FreeRef masks."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--text4seg-code-dir", type=Path, required=True)
    parser.add_argument("--sam-path", type=Path, required=True)
    parser.add_argument(
        "--sam-model-type",
        choices=("vit_b", "vit_l", "vit_h"),
        default="vit_h",
    )
    parser.add_argument("--model-label", default="Text4Seg-7B-p24")
    parser.add_argument("--split-name", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--point-count", type=int, default=10)
    parser.add_argument("--cascade-steps", type=int, default=2)
    parser.add_argument("--boundary-sigma", type=float, default=8.0)
    parser.add_argument("--boundary-tolerance", type=int, default=2)
    parser.add_argument("--save-visualizations", type=int, default=8)
    parser.add_argument("--n-segments", type=int, default=1024)
    parser.add_argument("--compactness", type=float, default=10.0)
    parser.add_argument("--slic-sigma", type=float, default=1.0)
    parser.add_argument("--graph-lambda", type=float, default=1.0)
    parser.add_argument("--confidence-power", type=float, default=2.0)
    parser.add_argument("--fusion-power", type=float, default=1.0)
    parser.add_argument("--foreground-seed", type=float, default=0.9)
    parser.add_argument("--background-seed", type=float, default=0.1)
    parser.add_argument("--seed-strength", type=float, default=50.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def stable_sample_seed(base_seed: int, name: str) -> int:
    digest = hashlib.blake2b(name.encode("utf-8"), digest_size=8).digest()
    return (base_seed + int.from_bytes(digest, "little")) % (2**32)


def save_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(path)


def load_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > 0


def resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if mask.shape == shape:
        return mask.astype(bool)
    return np.asarray(
        Image.fromarray(mask.astype(np.uint8) * 255).resize(
            (shape[1], shape[0]), Image.Resampling.NEAREST
        )
    ) > 0


def resolve_path(value: str, manifest: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else manifest.parent / path


def load_manifest(path: Path, limit: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"Manifest line {line_number} is not an object.")
        rows.append(
            {
                "name": str(value.get("name") or f"sample_{line_number:08d}"),
                "image": resolve_path(str(value["image"]), path),
                "coarse": resolve_path(str(value["pred_mask"]), path),
                "target": resolve_path(str(value["gt_mask"]), path),
            }
        )
    return rows[:limit] if limit else rows


def sam_protocol_name(model_type: str) -> str:
    return (
        "text4seg_public_p24_paired_frozen_sam_h_v1"
        if model_type == "vit_h"
        else f"text4seg_public_p24_paired_frozen_sam_{model_type}_v1"
    )


def load_sam(text4seg_code_dir: Path, sam_path: Path, model_type: str = "vit_h"):
    code_dir = str(text4seg_code_dir.resolve())
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)
    from llava.model.segment_anything import SamPredictor, sam_model_registry  # type: ignore[import-not-found]
    from llava.model.segment_anything.utils.transforms import ResizeLongestSide  # type: ignore[import-not-found]

    sam = sam_model_registry[model_type](checkpoint=str(sam_path))
    sam = sam.to(dtype=torch.float32, device="cuda")
    sam.eval()
    return SamPredictor(sam), ResizeLongestSide


def sam_refine(
    predictor,
    resize_longest_side,
    mask: np.ndarray,
    cascade_steps: int,
    point_count: int,
    seed: int,
) -> np.ndarray:
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)

    logits = compute_sam_logits(mask.astype(np.uint8), resize_longest_side)
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    try:
        np.random.seed(seed)
        torch.manual_seed(seed)
        point_coords, point_labels = sample_mask_points(mask, count=point_count)
    finally:
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)

    prediction, _, low_res_logits = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        mask_input=logits,
        multimask_output=False,
    )
    for _ in range(cascade_steps):
        prediction, _, low_res_logits = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            mask_input=low_res_logits,
            multimask_output=False,
        )
    return prediction[0].astype(bool)


def overlay(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> Image.Image:
    base = image.astype(np.float32)
    tint = np.empty_like(base)
    tint[...] = np.asarray(color, dtype=np.float32)
    alpha = mask.astype(np.float32)[..., None] * 0.45
    return Image.fromarray(np.clip(base * (1.0 - alpha) + tint * alpha, 0, 255).astype(np.uint8))


def save_visualization(
    path: Path,
    image: np.ndarray,
    target: np.ndarray,
    coarse: np.ndarray,
    freeref: np.ndarray,
    coarse_sam: np.ndarray,
    freeref_sam: np.ndarray,
) -> None:
    panels = [
        Image.fromarray(image),
        overlay(image, target, (40, 220, 80)),
        overlay(image, coarse, (230, 80, 60)),
        overlay(image, freeref, (60, 130, 240)),
        overlay(image, coarse_sam, (220, 150, 40)),
        overlay(image, freeref_sam, (130, 80, 220)),
    ]
    width, height = panels[0].size
    canvas = Image.new("RGB", (width * len(panels), height))
    for index, panel in enumerate(panels):
        canvas.paste(panel, (index * width, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def main() -> int:
    args = parse_args()
    if args.limit < 0 or args.cascade_steps < 0 or args.save_visualizations < 0:
        raise ValueError("limit, cascade-steps, and save-visualizations must be non-negative.")
    if args.point_count <= 0:
        raise ValueError("point-count must be positive.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for frozen-SAM evaluation.")
    minimum_checkpoint_bytes = {"vit_b": 300_000_000, "vit_l": 900_000_000, "vit_h": 2_000_000_000}
    if (
        not args.sam_path.is_file()
        or args.sam_path.stat().st_size < minimum_checkpoint_bytes[args.sam_model_type]
    ):
        raise FileNotFoundError(
            f"Complete SAM {args.sam_model_type} checkpoint not found: {args.sam_path}"
        )

    items = load_manifest(args.manifest, args.limit)
    if not items:
        raise FileNotFoundError(f"No samples found in {args.manifest}")

    config = TrainingFreeRefineConfig(
        n_segments=args.n_segments,
        compactness=args.compactness,
        slic_sigma=args.slic_sigma,
        graph_lambda=args.graph_lambda,
        confidence_power=args.confidence_power,
        fusion_power=args.fusion_power,
        foreground_seed=args.foreground_seed,
        background_seed=args.background_seed,
        seed_strength=args.seed_strength,
        threshold=args.threshold,
    )
    refiner = TrainingFreeUncertaintyRefiner(config)
    predictor, resize_longest_side = load_sam(
        args.text4seg_code_dir, args.sam_path, args.sam_model_type
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    variants = ("coarse", "freeref", "coarse_sam", "freeref_sam")
    mask_dirs = {name: args.output_dir / f"{name}_masks" for name in variants}
    metadata_dir = args.output_dir / "metadata"
    totals = {
        name: {"intersection": 0, "union": 0, "mean_iou": 0.0, "boundary": 0.0}
        for name in variants
    }
    rows: list[dict[str, object]] = []
    failure_count = 0

    for index, item in enumerate(
        tqdm(items, desc=f"Text4Seg SAM-H {args.split_name}", dynamic_ncols=True)
    ):
        stem = f"{index:08d}"
        image_path = Path(item["image"])
        target_path = Path(item["target"])
        coarse_path = Path(item["coarse"])
        image = np.asarray(Image.open(image_path).convert("RGB"))
        target = load_mask(target_path)
        coarse = resize_mask(load_mask(coarse_path), image.shape[:2])
        paths = {name: mask_dirs[name] / f"{stem}.png" for name in variants}
        metadata_path = metadata_dir / f"{stem}.json"
        timings = {"freeref_seconds": 0.0, "sam_image_seconds": 0.0, "sam_prompt_seconds": 0.0}

        if paths["freeref"].exists():
            freeref = resize_mask(load_mask(paths["freeref"]), image.shape[:2])
        else:
            start = time.perf_counter()
            output = refiner.refine_hard_mask(image, coarse, boundary_sigma=args.boundary_sigma)
            freeref = output["refined_probability"].numpy() >= config.threshold
            timings["freeref_seconds"] = time.perf_counter() - start
            save_mask(paths["coarse"], coarse)
            save_mask(paths["freeref"], freeref)

        need_coarse_sam = not paths["coarse_sam"].exists()
        need_freeref_sam = not paths["freeref_sam"].exists()
        if need_coarse_sam or need_freeref_sam:
            try:
                torch.cuda.synchronize()
                start = time.perf_counter()
                predictor.set_image(image)
                torch.cuda.synchronize()
                timings["sam_image_seconds"] = time.perf_counter() - start
                start = time.perf_counter()
                prompt_seed = stable_sample_seed(args.seed, str(item["name"]))
                if need_coarse_sam:
                    save_mask(
                        paths["coarse_sam"],
                        sam_refine(
                            predictor,
                            resize_longest_side,
                            coarse,
                            args.cascade_steps,
                            args.point_count,
                            prompt_seed,
                        ),
                    )
                if need_freeref_sam:
                    save_mask(
                        paths["freeref_sam"],
                        sam_refine(
                            predictor,
                            resize_longest_side,
                            freeref,
                            args.cascade_steps,
                            args.point_count,
                            prompt_seed,
                        ),
                    )
                torch.cuda.synchronize()
                timings["sam_prompt_seconds"] = time.perf_counter() - start
            except Exception as error:  # preserve paired full-split metrics with disclosed fallback
                failure_count += 1
                if need_coarse_sam:
                    save_mask(paths["coarse_sam"], coarse)
                if need_freeref_sam:
                    save_mask(paths["freeref_sam"], freeref)
                timings["sam_error"] = repr(error)

        if metadata_path.exists():
            cached = json.loads(metadata_path.read_text(encoding="utf-8"))
            for key in ("freeref_seconds", "sam_image_seconds", "sam_prompt_seconds"):
                if not timings[key]:
                    timings[key] = float(cached.get(key, 0.0))
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(timings, indent=2), encoding="utf-8")

        masks_at_image = {
            "coarse": coarse,
            "freeref": freeref,
            "coarse_sam": resize_mask(load_mask(paths["coarse_sam"]), image.shape[:2]),
            "freeref_sam": resize_mask(load_mask(paths["freeref_sam"]), image.shape[:2]),
        }
        masks = {name: resize_mask(mask, target.shape) for name, mask in masks_at_image.items()}
        row: dict[str, object] = {
            "name": item["name"],
            "image": str(image_path),
            "target": str(target_path),
            "coarse_mask": str(coarse_path),
            **timings,
        }
        for name, mask in masks.items():
            iou, intersection, union = mask_iou(mask, target)
            boundary = boundary_iou(mask, target, args.boundary_tolerance)
            totals[name]["intersection"] += intersection
            totals[name]["union"] += union
            totals[name]["mean_iou"] += iou
            totals[name]["boundary"] += boundary
            row[f"{name}_iou"] = iou
            row[f"{name}_boundary_iou"] = boundary
        row["freeref_delta"] = float(row["freeref_iou"]) - float(row["coarse_iou"])
        row["coarse_sam_delta"] = float(row["coarse_sam_iou"]) - float(row["coarse_iou"])
        row["freeref_sam_delta"] = float(row["freeref_sam_iou"]) - float(row["freeref_iou"])
        row["combined_delta"] = float(row["freeref_sam_iou"]) - float(row["coarse_iou"])
        rows.append(row)

        if index < args.save_visualizations:
            target_at_image = resize_mask(target, image.shape[:2])
            save_visualization(
                args.output_dir / "visualizations" / f"{stem}.png",
                image,
                target_at_image,
                masks_at_image["coarse"],
                masks_at_image["freeref"],
                masks_at_image["coarse_sam"],
                masks_at_image["freeref_sam"],
            )

    rows_path = args.output_dir / "eval_rows.csv"
    with rows_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames: list[str] = []
        for row in rows:
            fieldnames.extend(key for key in row if key not in fieldnames)
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    count = len(rows)
    summary: dict[str, object] = {
        "model": args.model_label,
        "split": args.split_name,
        "samples": count,
        "protocol": sam_protocol_name(args.sam_model_type),
        "note": (
            f"Coarse and FreeRef masks prompt the same frozen SAM {args.sam_model_type} "
            "with paired per-sample random seeds, 10 points per class, and two cascade passes."
        ),
        "sam_checkpoint": str(args.sam_path),
        "sam_model_type": args.sam_model_type,
        "sam_failures_with_input_fallback": failure_count,
        "config": asdict(config),
        "sam_config": {
            "point_count_per_class": args.point_count,
            "cascade_steps": args.cascade_steps,
            "seed": args.seed,
            "model_type": args.sam_model_type,
            "paired_seed_between_branches": True,
            "mask_logit_implementation": "training_free_refine.export_text4seg_masks:compute_sam_logits",
            "point_implementation": "training_free_refine.export_text4seg_masks:sample_mask_points",
        },
        "rows_csv": str(rows_path),
    }
    for name in variants:
        summary[f"{name}_mean_iou"] = totals[name]["mean_iou"] / count
        summary[f"{name}_cIoU"] = totals[name]["intersection"] / max(totals[name]["union"], 1)
        summary[f"{name}_boundary_iou"] = totals[name]["boundary"] / count
    for branch, baseline in (
        ("freeref", "coarse"),
        ("coarse_sam", "coarse"),
        ("freeref_sam", "freeref"),
    ):
        summary[f"{branch}_mean_iou_delta"] = float(summary[f"{branch}_mean_iou"]) - float(
            summary[f"{baseline}_mean_iou"]
        )
        summary[f"{branch}_cIoU_delta"] = float(summary[f"{branch}_cIoU"]) - float(
            summary[f"{baseline}_cIoU"]
        )
        summary[f"{branch}_improved_samples"] = sum(
            float(row[f"{branch}_iou"]) > float(row[f"{baseline}_iou"]) + 1e-12 for row in rows
        )
        summary[f"{branch}_degraded_samples"] = sum(
            float(row[f"{branch}_iou"]) < float(row[f"{baseline}_iou"]) - 1e-12 for row in rows
        )
    summary["combined_mean_iou_delta_vs_coarse"] = float(summary["freeref_sam_mean_iou"]) - float(
        summary["coarse_mean_iou"]
    )
    summary["combined_cIoU_delta_vs_coarse"] = float(summary["freeref_sam_cIoU"]) - float(
        summary["coarse_cIoU"]
    )
    summary["target_ordering_mean_iou"] = (
        float(summary["coarse_mean_iou"])
        < float(summary["freeref_mean_iou"])
        < float(summary["freeref_sam_mean_iou"])
    )
    summary["target_ordering_cIoU"] = (
        float(summary["coarse_cIoU"])
        < float(summary["freeref_cIoU"])
        < float(summary["freeref_sam_cIoU"])
    )
    summary["seconds_per_sample"] = sum(
        float(row["freeref_seconds"])
        + float(row["sam_image_seconds"])
        + float(row["sam_prompt_seconds"])
        for row in rows
    ) / count
    summary_path = args.output_dir / "eval_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
