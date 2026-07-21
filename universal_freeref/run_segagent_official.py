from __future__ import annotations

import argparse
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


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


def parse_wrapper_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--segagent-code-dir", type=Path, required=True)
    parser.add_argument("--limit-items", type=int, default=0)
    parser.add_argument("--offset-items", type=int, default=0)
    return parser.parse_known_args()


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
        evaluator.forward(args.img, args.json)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
