from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from .eval_stamp_dumps import boundary_iou, mask_iou, resolve_asset
from .refiner import TrainingFreeRefineConfig, TrainingFreeUncertaintyRefiner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen SAM-H after STAMP coarse masks and FreeRef masks."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stamp-code-dir", type=Path, required=True)
    parser.add_argument("--sam-path", type=Path, required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--split-name", required=True)
    parser.add_argument("--pattern", default="*.pt")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--point-count", type=int, default=10)
    parser.add_argument("--cascade-steps", type=int, default=2)
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


def save_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255).save(path)


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


def stable_sample_seed(base_seed: int, name: str) -> int:
    """Return one paired seed for coarse and FreeRef prompts of the same sample."""
    digest = hashlib.blake2b(name.encode("utf-8"), digest_size=8).digest()
    return (base_seed + int.from_bytes(digest, "little")) % (2**32)


def sam_refine(
    predictor,
    compute_logits_from_mask,
    masks_sample_points,
    mask: np.ndarray,
    cascade_steps: int,
    seed: int,
) -> np.ndarray:
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)

    # Match STAMP's released SAM-H protocol exactly. In particular, STAMP first
    # resizes/pads the hard mask and only then converts it to SAM mask logits.
    logits = compute_logits_from_mask(mask.astype(np.float32))
    if logits.shape != (1, 256, 256) or not np.isfinite(logits).all():
        raise ValueError(f"Official STAMP SAM mask prompt is invalid: {logits.shape}")

    # The official helper uses torch.randperm. Preserve global RNG state while
    # making the paired coarse/FreeRef comparison reproducible.
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    try:
        np.random.seed(seed)
        torch.manual_seed(seed)
        point_coords, point_labels = masks_sample_points(
            torch.from_numpy(mask.astype(np.float32))
        )
    finally:
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)

    sam_mask, _, low_res_logits = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        mask_input=logits,
        multimask_output=False,
    )
    for _ in range(cascade_steps):
        sam_mask, _, low_res_logits = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            mask_input=low_res_logits,
            multimask_output=False,
        )
    return sam_mask[0].astype(bool)


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


def load_sam(stamp_code_dir: Path, sam_path: Path):
    resolved_code_dir = stamp_code_dir.resolve()
    code_dir = str(resolved_code_dir)
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)
    from model.segment_anything import SamPredictor, sam_model_registry  # type: ignore[import-not-found]

    official_utils = importlib.import_module("eval.utils")
    utils_path = Path(official_utils.__file__).resolve()
    if resolved_code_dir not in utils_path.parents:
        raise ImportError(
            f"Expected STAMP eval.utils under {resolved_code_dir}, imported {utils_path} instead."
        )
    compute_logits_from_mask = official_utils.compute_logits_from_mask
    masks_sample_points = official_utils.masks_sample_points

    sam = sam_model_registry["vit_h"](checkpoint=str(sam_path))
    sam = sam.to(dtype=torch.float32, device="cuda")
    sam.eval()
    return SamPredictor(sam), compute_logits_from_mask, masks_sample_points, utils_path


def main() -> int:
    args = parse_args()
    if args.limit < 0 or args.cascade_steps < 0:
        raise ValueError("limit and cascade-steps must be non-negative.")
    if args.point_count != 10:
        raise ValueError("The released STAMP SAM-H protocol requires exactly 10 points per class.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for SAM-H evaluation.")
    if not args.sam_path.is_file() or args.sam_path.stat().st_size < 2_000_000_000:
        raise FileNotFoundError(f"A complete SAM-H checkpoint was not found at {args.sam_path}.")

    dump_paths = sorted(args.input_dir.glob(args.pattern))
    if args.limit:
        dump_paths = dump_paths[: args.limit]
    if not dump_paths:
        raise FileNotFoundError(f"No STAMP dumps found in {args.input_dir}.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    mask_dirs = {
        name: args.output_dir / name
        for name in ("coarse_masks", "freeref_masks", "coarse_sam_masks", "freeref_sam_masks")
    }
    metadata_dir = args.output_dir / "metadata"

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
    predictor, compute_logits_from_mask, masks_sample_points, official_utils_path = load_sam(
        args.stamp_code_dir, args.sam_path
    )

    variants = ("coarse", "freeref", "coarse_sam", "freeref_sam")
    totals = {name: {"intersection": 0, "union": 0, "mean_iou": 0.0, "boundary": 0.0} for name in variants}
    rows: list[dict[str, object]] = []
    failure_count = 0

    for index, dump_path in enumerate(tqdm(dump_paths, desc=f"SAM-H {args.model_label} {args.split_name}", dynamic_ncols=True)):
        payload = torch.load(dump_path, map_location="cpu", weights_only=False)
        image_path = resolve_asset(payload["image_path"], dump_path)
        target_path = resolve_asset(payload["mask_path"], dump_path)
        image = np.asarray(Image.open(image_path).convert("RGB"))
        target = load_mask(target_path)
        if image.shape[:2] != target.shape:
            image = np.asarray(
                Image.fromarray(image).resize((target.shape[1], target.shape[0]), Image.Resampling.BILINEAR)
            )
        ignore = np.zeros(target.shape, dtype=bool)
        if payload.get("ignore_path"):
            ignore_path = resolve_asset(payload["ignore_path"], dump_path)
            ignore = resize_mask(load_mask(ignore_path), target.shape)
            target &= ~ignore

        stem = dump_path.stem
        paths = {name: mask_dirs[f"{name}_masks"] / f"{stem}.png" for name in variants}
        metadata_path = metadata_dir / f"{stem}.json"
        timings = {"freeref_seconds": 0.0, "sam_image_seconds": 0.0, "sam_prompt_seconds": 0.0}

        if paths["coarse"].exists() and paths["freeref"].exists():
            coarse = resize_mask(load_mask(paths["coarse"]), target.shape)
            freeref = resize_mask(load_mask(paths["freeref"]), target.shape)
        else:
            start = time.perf_counter()
            output = refiner.refine(image, payload["mask_logits"], tuple(payload["grid_hw"]))
            timings["freeref_seconds"] = time.perf_counter() - start
            coarse = output["coarse_probability"].numpy() >= config.threshold
            freeref = output["refined_probability"].numpy() >= config.threshold
            coarse &= ~ignore
            freeref &= ~ignore
            save_mask(paths["coarse"], coarse)
            save_mask(paths["freeref"], freeref)

        need_coarse_sam = not paths["coarse_sam"].exists()
        need_freeref_sam = not paths["freeref_sam"].exists()
        if need_coarse_sam or need_freeref_sam:
            try:
                if (need_coarse_sam and coarse.any()) or (need_freeref_sam and freeref.any()):
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    predictor.set_image(image)
                    torch.cuda.synchronize()
                    timings["sam_image_seconds"] = time.perf_counter() - start
                torch.cuda.synchronize()
                start = time.perf_counter()
                prompt_seed = stable_sample_seed(args.seed, stem)
                if need_coarse_sam:
                    coarse_sam = sam_refine(
                        predictor,
                        compute_logits_from_mask,
                        masks_sample_points,
                        coarse,
                        args.cascade_steps,
                        prompt_seed,
                    )
                    save_mask(paths["coarse_sam"], coarse_sam)
                if need_freeref_sam:
                    freeref_sam = sam_refine(
                        predictor,
                        compute_logits_from_mask,
                        masks_sample_points,
                        freeref,
                        args.cascade_steps,
                        prompt_seed,
                    )
                    save_mask(paths["freeref_sam"], freeref_sam)
                torch.cuda.synchronize()
                timings["sam_prompt_seconds"] = time.perf_counter() - start
            except Exception as error:  # preserve full-split pairing with a disclosed fallback
                failure_count += 1
                if need_coarse_sam:
                    save_mask(paths["coarse_sam"], coarse)
                if need_freeref_sam:
                    save_mask(paths["freeref_sam"], freeref)
                timings["sam_error"] = repr(error)

        coarse_sam = resize_mask(load_mask(paths["coarse_sam"]), target.shape) & ~ignore
        freeref_sam = resize_mask(load_mask(paths["freeref_sam"]), target.shape) & ~ignore
        if metadata_path.exists():
            cached = json.loads(metadata_path.read_text(encoding="utf-8"))
            for key in ("freeref_seconds", "sam_image_seconds", "sam_prompt_seconds"):
                if not timings[key]:
                    timings[key] = float(cached.get(key, 0.0))
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(timings, indent=2), encoding="utf-8")

        masks = {"coarse": coarse, "freeref": freeref, "coarse_sam": coarse_sam, "freeref_sam": freeref_sam}
        row: dict[str, object] = {
            "name": payload.get("name", stem),
            "dump": str(dump_path),
            "image": str(image_path),
            "target": str(target_path),
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
            save_visualization(
                args.output_dir / "visualizations" / f"{stem}.png",
                image,
                target,
                coarse,
                freeref,
                coarse_sam,
                freeref_sam,
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
        "protocol": "stamp_official_frozen_sam_h_v1",
        "note": (
            "Uses STAMP eval.utils.compute_logits_from_mask and masks_sample_points, "
            "followed by the released initial SAM prediction and two cascade passes. "
            "The STAMP and FreeRef branches use the same per-sample RNG seed."
        ),
        "sam_checkpoint": str(args.sam_path),
        "official_stamp_utils": str(official_utils_path),
        "sam_failures_with_input_fallback": failure_count,
        "config": asdict(config),
        "sam_config": {
            "point_count_per_class": 10,
            "cascade_steps": args.cascade_steps,
            "seed": args.seed,
            "paired_seed_between_branches": True,
            "mask_logit_implementation": "STAMP/eval/utils.py:compute_logits_from_mask",
            "point_implementation": "STAMP/eval/utils.py:masks_sample_points",
        },
        "rows_csv": str(rows_path),
    }
    for name in variants:
        summary[f"{name}_mean_iou"] = totals[name]["mean_iou"] / count
        summary[f"{name}_cIoU"] = totals[name]["intersection"] / max(totals[name]["union"], 1)
        summary[f"{name}_boundary_iou"] = totals[name]["boundary"] / count
    for branch, baseline in (("freeref", "coarse"), ("coarse_sam", "coarse"), ("freeref_sam", "freeref")):
        summary[f"{branch}_mean_iou_delta"] = float(summary[f"{branch}_mean_iou"]) - float(summary[f"{baseline}_mean_iou"])
        summary[f"{branch}_cIoU_delta"] = float(summary[f"{branch}_cIoU"]) - float(summary[f"{baseline}_cIoU"])
        summary[f"{branch}_improved_samples"] = sum(float(row[f"{branch}_iou"]) > float(row[f"{baseline}_iou"]) + 1e-12 for row in rows)
        summary[f"{branch}_degraded_samples"] = sum(float(row[f"{branch}_iou"]) < float(row[f"{baseline}_iou"]) - 1e-12 for row in rows)
    summary["combined_mean_iou_delta_vs_coarse"] = float(summary["freeref_sam_mean_iou"]) - float(summary["coarse_mean_iou"])
    summary["combined_cIoU_delta_vs_coarse"] = float(summary["freeref_sam_cIoU"]) - float(summary["coarse_cIoU"])
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
        float(row["freeref_seconds"]) + float(row["sam_image_seconds"]) + float(row["sam_prompt_seconds"])
        for row in rows
    ) / count
    summary_path = args.output_dir / "eval_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
