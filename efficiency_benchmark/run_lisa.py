from __future__ import annotations

import argparse
import sys
from functools import partial
from pathlib import Path

import torch
from tqdm import tqdm

from efficiency_benchmark.common import (
    ProcessGpuMemoryMonitor,
    assert_rtx_4090,
    cuda_elapsed,
    select_indices,
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
    if getattr(dataset, "data_type", None) != "refer_seg":
        raise RuntimeError("The efficiency table requires an official REFER segmentation split.")
    refer_data = dataset.refer_seg_ds
    expression_locations: list[tuple[int, int]] = []
    for image_index, image_info in enumerate(refer_data["images"]):
        expression_count = sum(
            len(ref["sentences"]) for ref in refer_data["img2refs"][image_info["id"]]
        )
        expression_locations.extend(
            (image_index, expression_index) for expression_index in range(expression_count)
        )
    warmup_indices, measured_indices = select_indices(
        len(expression_locations), args.warmup, args.samples, args.seed
    )

    def single_expression_item(item, expression_index: int):
        values = list(item)
        conversations = values[3]
        masks = values[4]
        if not 0 <= expression_index < len(conversations):
            raise IndexError(
                f"Expression {expression_index} is outside an item with {len(conversations)} prompts."
            )
        values[3] = [conversations[expression_index]]
        values[4] = masks[expression_index : expression_index + 1]
        for field_index in (7, 8):
            if isinstance(values[field_index], list):
                values[field_index] = [values[field_index][expression_index]]
        return tuple(values)

    def run(location_index: int) -> dict[str, float | int]:
        image_index, expression_index = expression_locations[location_index]
        # Image decoding is excluded; collation, tensor preparation, model forward,
        # native learned SAM decoder, and mask thresholding are included.
        item = single_expression_item(dataset[image_index], expression_index)

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
        if len(batch["conversation_list"]) != 1 or len(masks) != 1 or len(masks[0]) != 1:
            raise RuntimeError("LISA single-expression benchmark produced a grouped mask output.")
        return {
            "location_index": location_index,
            "image_index": image_index,
            "expression_index": expression_index,
            "native_pipeline_seconds": elapsed,
            "total_seconds": elapsed,
        }

    for location_index in tqdm(warmup_indices, desc="LISA warmup expressions"):
        run(location_index)
    monitor = ProcessGpuMemoryMonitor()
    monitor.start()
    rows = [
        run(location_index)
        for location_index in tqdm(measured_indices, desc="LISA measured expressions")
    ]
    peak_gpu_gib, memory_backend = monitor.finish()
    write_report(
        args.output_dir,
        {
            "method": "LISA-7B",
            "variant": "original",
            "device": gpu_name,
            "protocol": (
                "e2e_official_refer_single_expression_batch1_"
                f"warmup{args.warmup}_seed{args.seed}"
            ),
            "warmup": args.warmup,
            "seed": args.seed,
            "model": str(model_path),
            "vision_tower": str(vision_tower),
            "sam_path": str(sam_path),
            "val_dataset": args.val_dataset,
            "note": "Each timed call contains exactly one referring expression and one native LISA mask decode.",
        },
        rows,
        peak_gpu_gib,
        memory_backend,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
