from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from efficiency_benchmark.common import (
    ProcessGpuMemoryMonitor,
    assert_rtx_4090,
    cuda_elapsed,
    select_indices,
    write_report,
)
from training_free_refine import (
    GpuTrainingFreeUncertaintyRefiner,
    TrainingFreeRefineConfig,
    stamp_probability_gpu,
)
from training_free_refine.eval_stamp_sam_h import load_sam, sam_refine, stable_sample_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified RTX 4090 STAMP end-to-end benchmark.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stamp-code-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--eval-json", type=Path, required=True)
    parser.add_argument("--sam-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variant", choices=("base", "freeref_gpu", "sam_h"), required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-pixels", type=int, default=802816)
    parser.add_argument("--max-pixels", type=int, default=1003520)
    parser.add_argument("--allow-other-gpu", action="store_true")
    return parser.parse_args()


def _raw_outputs(segmenter: Any, image: Image.Image, query: str) -> dict[str, Any]:
    for name in (
        "generate_with_refinement_outputs",
        "generate_with_refinement",
        "export_refinement_outputs",
        "forward_for_refinement",
    ):
        function = getattr(segmenter, name, None)
        if not callable(function):
            continue
        try:
            output = function(image, query)
        except TypeError:
            output = function(images=[image], texts=[query])
        if isinstance(output, tuple):
            output = next(
                (value for value in reversed(output) if isinstance(value, dict) and "mask_logits" in value),
                output,
            )
        if not isinstance(output, dict) or not {"mask_logits", "grid_hw"} <= set(output):
            raise TypeError(f"Unexpected STAMP refinement output from {name}.")
        return output
    raise AttributeError("STAMP does not expose generate_with_refinement_outputs; run patch 78 first.")


def main() -> int:
    args = parse_args()
    gpu_name = assert_rtx_4090(args.allow_other_gpu)
    root = args.root.resolve()
    code_dir = args.stamp_code_dir.resolve()
    tools_root = Path(__file__).resolve().parents[1]
    refine_stamp_source = tools_root / "offline_rstamp" / "refine_stamp_src"
    if str(refine_stamp_source) not in sys.path:
        sys.path.insert(0, str(refine_stamp_source))
    sys.path.insert(0, str(code_dir))
    try:
        from segment_predictor import GenerativeSegmenter  # type: ignore[import-not-found]
    except ImportError:
        from segment_predictor_cache import GenerativeSegmenter  # type: ignore[import-not-found]
    from offline_rstamp.refine_stamp_src.refine_stamp.scripts.export_stamp_refinement_dumps import (
        build_query,
        get_image_path,
        load_official_question_templates,
    )

    items = json.loads(args.eval_json.read_text(encoding="utf-8"))
    warmup_indices, measured_indices = select_indices(
        len(items), args.warmup, args.samples, args.seed
    )
    segmenter = GenerativeSegmenter(
        str(args.model.resolve()),
        device_map="cuda",
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    question_templates = load_official_question_templates(code_dir)
    config = TrainingFreeRefineConfig()
    refiner = GpuTrainingFreeUncertaintyRefiner(config) if args.variant == "freeref_gpu" else None
    predictor = compute_logits = sample_points = official_utils = None
    if args.variant == "sam_h":
        predictor, compute_logits, sample_points, official_utils = load_sam(
            code_dir, args.sam_path.resolve()
        )

    def run(index: int) -> dict[str, float | int]:
        item = items[index]
        image_path = get_image_path(item, root)
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        image_array = np.asarray(image)
        query = build_query(item, "official", question_templates)

        def infer():
            with torch.inference_mode():
                output = _raw_outputs(segmenter, image, query)
                coarse = stamp_probability_gpu(
                    output["mask_logits"], tuple(output["grid_hw"]), image_array.shape[:2]
                )
            return output, coarse

        (output, coarse), model_seconds = cuda_elapsed(infer)
        post_seconds = 0.0
        if args.variant == "base":
            _, post_seconds = cuda_elapsed(lambda: coarse >= config.threshold)
        elif args.variant == "freeref_gpu":
            assert refiner is not None
            _, post_seconds = cuda_elapsed(
                lambda: refiner.refine_probability(image_array, coarse)["refined_mask"]
            )
        else:
            assert predictor is not None and compute_logits is not None and sample_points is not None

            def apply_sam():
                coarse_mask = (coarse >= config.threshold).cpu().numpy()
                predictor.set_image(image_array)
                return sam_refine(
                    predictor,
                    compute_logits,
                    sample_points,
                    coarse_mask,
                    cascade_steps=2,
                    seed=stable_sample_seed(args.seed, str(index)),
                )

            _, post_seconds = cuda_elapsed(apply_sam)
        return {
            "index": index,
            "model_and_reconstruction_seconds": model_seconds,
            "postprocess_seconds": post_seconds,
            "total_seconds": model_seconds + post_seconds,
        }

    for index in tqdm(warmup_indices, desc=f"STAMP warmup {args.variant}"):
        run(index)
    monitor = ProcessGpuMemoryMonitor()
    monitor.start()
    rows = [run(index) for index in tqdm(measured_indices, desc=f"STAMP measured {args.variant}")]
    peak_gpu_gib, memory_backend = monitor.finish()
    write_report(
        args.output_dir,
        {
            "method": "STAMP-7B",
            "variant": args.variant,
            "device": gpu_name,
            "protocol": (
                f"e2e_refcoco_testA_batch1_warmup{args.warmup}_random_seed{args.seed}"
            ),
            "warmup": args.warmup,
            "seed": args.seed,
            "model": str(args.model.resolve()),
            "eval_json": str(args.eval_json.resolve()),
            "sam_path": str(args.sam_path.resolve()) if args.variant == "sam_h" else None,
            "sam_official_utils": str(official_utils) if official_utils else None,
            "freeref_backend": (
                "cucim-cupy-local-slic" if args.variant == "freeref_gpu" else None
            ),
        },
        rows,
        peak_gpu_gib,
        memory_backend,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
