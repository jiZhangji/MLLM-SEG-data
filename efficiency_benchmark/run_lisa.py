from __future__ import annotations

import argparse
import random
import sys
from functools import partial
from pathlib import Path

import torch
from tqdm import tqdm

from efficiency_benchmark.common import (
    ProcessGpuMemoryMonitor,
    assert_rtx_4090,
    cuda_elapsed,
    write_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified RTX 4090 LISA end-to-end benchmark.")
    parser.add_argument("--lisa-code-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--vision-tower", type=Path, required=True)
    parser.add_argument("--sam-path", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--val-dataset", default="refcoco|unc|testA")
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    parser.add_argument("--conv-type", choices=("llava_v1", "llava_llama_2"), default="llava_v1")
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--model-max-length", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow-other-gpu", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    gpu_name = assert_rtx_4090(args.allow_other_gpu)
    code_dir = args.lisa_code_dir.resolve()
    model_path = args.model_path.resolve()
    vision_tower = args.vision_tower.resolve()
    sam_path = args.sam_path.resolve()
    dataset_dir = args.dataset_dir.resolve()
    for path in (code_dir, model_path, vision_tower, dataset_dir):
        if not path.exists():
            raise FileNotFoundError(path)
    if not sam_path.is_file():
        raise FileNotFoundError(sam_path)
    sys.path.insert(0, str(code_dir))
    from transformers import AutoTokenizer  # type: ignore
    from model.LISA import LISAForCausalLM  # type: ignore[import-not-found]
    from model.llava import conversation as conversation_lib  # type: ignore[import-not-found]
    from model.llava.model.language_model.llava_llama import LlavaConfig  # type: ignore[import-not-found]
    from utils.dataset import ValDataset, collate_fn  # type: ignore[import-not-found]
    from utils.utils import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN  # type: ignore[import-not-found]

    dtype = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[args.precision]
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
        raise RuntimeError(f"Expected one [SEG] token, got {seg_token_ids}.")
    tokenizer.add_tokens(
        [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
    )
    config = LlavaConfig.from_pretrained(model_path, local_files_only=True)
    config.vision_tower = str(vision_tower)
    config.mm_vision_tower = str(vision_tower)
    model = LISAForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        train_mask_decoder=True,
        out_dim=256,
        seg_token_idx=seg_token_ids[0],
        vision_pretrained=str(sam_path),
        vision_tower=str(vision_tower),
        use_mm_start_end=True,
        local_files_only=True,
    )
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.resize_token_embeddings(len(tokenizer))
    model.get_model().initialize_vision_modules(model.get_model().config)
    model.to(device="cuda", dtype=dtype)
    model.get_model().get_vision_tower().to(device="cuda", dtype=dtype)
    model.eval()
    conversation_lib.default_conversation = conversation_lib.conv_templates[args.conv_type]
    dataset = ValDataset(
        str(dataset_dir),
        tokenizer,
        str(vision_tower),
        args.val_dataset,
        args.image_size,
    )
    collate = partial(
        collate_fn,
        tokenizer=tokenizer,
        conv_type=args.conv_type,
        use_mm_start_end=True,
        local_rank=0,
    )
    indices = list(range(len(dataset)))
    random.Random(args.seed).shuffle(indices)

    def run(image_index: int) -> tuple[list[dict[str, float | int]], int]:
        # Image decoding is excluded; collation, tensor preparation, model forward,
        # native learned SAM decoder, and mask thresholding are included.
        item = dataset[image_index]

        def infer():
            batch = collate([item])
            for key, value in list(batch.items()):
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to("cuda", non_blocking=False)
                elif value and isinstance(value, list) and isinstance(value[0], torch.Tensor):
                    batch[key] = [part.to("cuda", non_blocking=False) for part in value]
            batch["images"] = batch["images"].to(dtype=dtype)
            batch["images_clip"] = batch["images_clip"].to(dtype=dtype)
            with torch.inference_mode():
                output = model(**batch)
                masks = [group > 0.0 for group in output["pred_masks"]]
            return batch, masks

        (batch, masks), elapsed = cuda_elapsed(infer)
        expression_count = len(batch["conversation_list"])
        if len(masks) != 1 or len(masks[0]) != expression_count:
            raise RuntimeError("LISA expression count does not match its native mask output.")
        per_expression = elapsed / expression_count
        return [
            {
                "image_index": image_index,
                "expression_index": expression_index,
                "native_pipeline_seconds": per_expression,
                "total_seconds": per_expression,
            }
            for expression_index in range(expression_count)
        ], expression_count

    cursor = 0
    warmed = 0
    with tqdm(total=args.warmup, desc="LISA warmup expressions") as progress:
        while warmed < args.warmup:
            _, count = run(indices[cursor])
            cursor += 1
            step = min(count, args.warmup - warmed)
            warmed += step
            progress.update(step)
    monitor = ProcessGpuMemoryMonitor()
    monitor.start()
    rows: list[dict[str, float | int]] = []
    with tqdm(total=args.samples, desc="LISA measured expressions") as progress:
        while len(rows) < args.samples:
            image_rows, _ = run(indices[cursor])
            cursor += 1
            needed = args.samples - len(rows)
            rows.extend(image_rows[:needed])
            progress.update(min(len(image_rows), needed))
    peak_gpu_gib, memory_backend = monitor.finish()
    write_report(
        args.output_dir,
        {
            "method": "LISA-7B",
            "variant": "original",
            "device": gpu_name,
            "protocol": (
                "e2e_official_refer_grouped_expressions_batch1_image_"
                f"warmup{args.warmup}_seed{args.seed}"
            ),
            "warmup": args.warmup,
            "seed": args.seed,
            "model": str(model_path),
            "vision_tower": str(vision_tower),
            "sam_path": str(sam_path),
            "val_dataset": args.val_dataset,
            "note": "LISA natively evaluates all referring expressions for one image together; image-call latency is evenly amortized across those expressions.",
        },
        rows,
        peak_gpu_gib,
        memory_backend,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
