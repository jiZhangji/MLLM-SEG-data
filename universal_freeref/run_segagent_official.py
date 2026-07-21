from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
from PIL import Image


class FullEvaluationList(list[Any]):
    """Neutralize the released evaluator's hard-coded `self.coco[:5000]` cap."""

    def __init__(self, values: list[Any], offset: int = 0, limit: int = 0) -> None:
        super().__init__(values)
        self.offset = offset
        self.limit = limit

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, slice) and key.start is None and key.stop == 5000 and key.step is None:
            stop = self.offset + self.limit if self.limit > 0 else None
            return list.__getitem__(self, slice(self.offset, stop, None))
        return list.__getitem__(self, key)


@contextmanager
def trusted_legacy_checkpoint_loading(torch_module: Any) -> Iterator[None]:
    """Restore the pre-2.6 torch.load default for trusted official checkpoints."""

    original_load = torch_module.load

    def load(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch_module.load = load
    try:
        yield
    finally:
        torch_module.load = original_load


@contextmanager
def freeref_guided_click_generation(
    grounding_model: Any,
    refiner: Any,
    torch_module: Any,
    image_loader: Callable[[str], np.ndarray],
    boundary_sigma: float,
) -> Iterator[dict[str, float]]:
    """Show a FreeRef-refined intermediate mask to the click-generating MLLM."""

    had_instance_method = "generate_response" in getattr(grounding_model, "__dict__", {})
    original_instance_method = getattr(grounding_model, "__dict__", {}).get("generate_response")
    original_generate_response = grounding_model.generate_response
    cached_path = ""
    cached_image: np.ndarray | None = None
    stats = {"guided_masks": 0.0, "freeref_seconds": 0.0}

    def generate_response(prompt: Any, img_path: str, mask: Any, conv: Any) -> Any:
        nonlocal cached_path, cached_image
        guided_mask = mask
        if mask is not None:
            if img_path != cached_path:
                cached_image = image_loader(img_path)
                cached_path = img_path
            mask_array = mask.detach().cpu().numpy() if hasattr(mask, "detach") else np.asarray(mask)
            if mask_array.ndim == 3 and mask_array.shape[0] == 1:
                mask_array = mask_array[0]
            start = time.perf_counter()
            output = refiner.refine_hard_mask(
                cached_image,
                mask_array,
                boundary_sigma=boundary_sigma,
            )
            stats["freeref_seconds"] += time.perf_counter() - start
            stats["guided_masks"] += 1.0
            guided_mask = output["refined_mask"]
            if not isinstance(guided_mask, torch_module.Tensor):
                guided_mask = torch_module.as_tensor(guided_mask)
            if getattr(mask, "ndim", 0) == 3:
                guided_mask = guided_mask.unsqueeze(0)
            guided_mask = guided_mask.to(device=mask.device, dtype=mask.dtype)
        return original_generate_response(prompt, img_path, guided_mask, conv)

    grounding_model.generate_response = generate_response
    try:
        yield stats
    finally:
        if had_instance_method:
            grounding_model.generate_response = original_instance_method
        else:
            del grounding_model.generate_response


def parse_wrapper_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--segagent-code-dir", type=Path, required=True)
    parser.add_argument("--limit-items", type=int, default=0)
    parser.add_argument("--offset-items", type=int, default=0)
    parser.add_argument("--freeref-click-guidance", action="store_true")
    parser.add_argument("--freeref-boundary-sigma", type=float, default=8.0)
    parser.add_argument("--freeref-n-segments", type=int, default=1024)
    parser.add_argument("--freeref-graph-lambda", type=float, default=1.0)
    return parser.parse_known_args()


def load_rgb(path: str) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def main() -> int:
    wrapper, official_args = parse_wrapper_args()
    if min(wrapper.limit_items, wrapper.offset_items) < 0:
        raise ValueError("limit-items and offset-items must be non-negative.")
    code_dir = wrapper.segagent_code_dir.expanduser().resolve()
    eval_dir = code_dir / "evaltools"
    if not (eval_dir / "main.py").is_file():
        raise FileNotFoundError(f"SegAgent official evaluator is missing below {eval_dir}.")

    os.chdir(eval_dir)
    sys.path.insert(0, str(code_dir))
    sys.path.insert(0, str(eval_dir))
    sys.argv = ["segagent-official"] + official_args

    import torch
    import torch.distributed as dist

    from config import get_config  # type: ignore[import-not-found]
    from model_loader import load_model  # type: ignore[import-not-found]
    from refcocog_eval import REFCOCOG_EVAL  # type: ignore[import-not-found]

    args = get_config()
    args.device = torch.device("cpu" if args.cpu else f"cuda:{args.gpus.split(',')[0]}")
    if (args.iou_analysis or args.print_ious) and args.min_n_clicks <= 1:
        args.target_iou = 1.01
    else:
        args.target_iou = max(0.8, args.target_iou)

    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "localhost")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",", 1)[0]
    try:
        port_offset = int(visible)
    except ValueError:
        port_offset = 0
    os.environ.setdefault("MASTER_PORT", str(12399 + port_offset))
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo" if args.cpu else "nccl", init_method="env://")

    original_load_annotations = REFCOCOG_EVAL.load_annotations

    def load_all_annotations(self: Any, annotation_file: str) -> FullEvaluationList:
        values = original_load_annotations(self, annotation_file)
        return FullEvaluationList(values, wrapper.offset_items, wrapper.limit_items)

    REFCOCOG_EVAL.load_annotations = load_all_annotations
    try:
        # SimpleClick checkpoints serialize model classes and predate PyTorch
        # 2.6's weights_only=True default. These files come from the verified
        # upstream release and need the legacy loader only during model setup.
        with trusted_legacy_checkpoint_loading(torch):
            segmentation_model, grounding_model = load_model(args)
        evaluator = REFCOCOG_EVAL(grounding_model, segmentation_model, args)
        guidance_stats: dict[str, float] = {"guided_masks": 0.0, "freeref_seconds": 0.0}
        if wrapper.freeref_click_guidance:
            from training_free_refine import TrainingFreeRefineConfig, TrainingFreeUncertaintyRefiner

            config = TrainingFreeRefineConfig(
                n_segments=wrapper.freeref_n_segments,
                graph_lambda=wrapper.freeref_graph_lambda,
            )
            refiner = TrainingFreeUncertaintyRefiner(config)
            with freeref_guided_click_generation(
                grounding_model,
                refiner,
                torch,
                load_rgb,
                wrapper.freeref_boundary_sigma,
            ) as guidance_stats:
                evaluator.forward(args.img, args.json)
            report = {
                "protocol": "segagent_freeref_guided_click_generation_v1",
                "placement": "intermediate SimpleClick mask -> FreeRef -> next SegAgent click -> SimpleClick",
                "final_mask_postprocessed": False,
                "boundary_sigma": wrapper.freeref_boundary_sigma,
                "config": {
                    "n_segments": wrapper.freeref_n_segments,
                    "graph_lambda": wrapper.freeref_graph_lambda,
                },
                **guidance_stats,
            }
            output_root = Path(os.environ.get("VIS_DIR", os.getcwd()))
            (output_root / "freeref_click_guidance.json").write_text(
                json.dumps(report, indent=2, ensure_ascii=True),
                encoding="utf-8",
            )
            print(json.dumps(report, indent=2, ensure_ascii=True))
        else:
            evaluator.forward(args.img, args.json)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
