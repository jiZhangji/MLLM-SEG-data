from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from training_free_refine.data import ReferringSegDataset, extract_target_text

from .export_utils import (
    artifact_paths,
    artifacts_complete,
    atomic_save_logits,
    atomic_save_mask,
    atomic_write_json,
    atomic_write_jsonl,
    merge_binary_logits,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export logits from the public PixelLM-7B checkpoint and build a paired FreeRef manifest. "
            "This uses PixelLM's public chat inference on the flat STAMP JSON protocol; it is not a "
            "claim that the paper table has been reproduced."
        )
    )
    parser.add_argument("--pixellm-code-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--vision-tower", type=Path, required=True)
    parser.add_argument("--preprocessor-config", type=Path, required=True)
    parser.add_argument("--eval-json", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--method", default="PixelLM-7B-public")
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    parser.add_argument("--conv-type", choices=("llava_v1", "llava_llama_2"), default="llava_v1")
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--model-max-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--seg-token-num", type=int, default=3)
    parser.add_argument("--image-feature-scale-num", type=int, default=2)
    parser.add_argument(
        "--question-template",
        default="Can you segment {target} in this image? Please output a segmentation mask.",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _dtype(precision: str) -> torch.dtype:
    return {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[precision]


def preprocess_square(
    image: torch.Tensor,
    image_size: int,
    pixel_mean: tuple[float, float, float] = (123.675, 116.28, 103.53),
    pixel_std: tuple[float, float, float] = (58.395, 57.12, 57.375),
) -> torch.Tensor:
    mean = torch.tensor(pixel_mean, dtype=torch.float32).view(-1, 1, 1)
    std = torch.tensor(pixel_std, dtype=torch.float32).view(-1, 1, 1)
    result = (image.float() - mean) / std
    height, width = result.shape[-2:]
    if height > image_size or width > image_size:
        raise ValueError(f"Resized image {(height, width)} exceeds square size {image_size}.")
    return F.pad(result, (0, image_size - width, 0, image_size - height))


def build_prompt(
    target_text: str,
    question_template: str,
    conversation_lib: Any,
    conv_type: str,
    image_token: str,
    image_start_token: str,
    image_end_token: str,
) -> str:
    question = question_template.format(target=target_text.strip().lower())
    prompt = image_token + "\n" + question
    prompt = prompt.replace(image_token, image_start_token + image_token + image_end_token)
    conversation = conversation_lib.conv_templates[conv_type].copy()
    conversation.messages = []
    conversation.append_message(conversation.roles[0], prompt)
    conversation.append_message(conversation.roles[1], "")
    return conversation.get_prompt()


def _manifest_row(
    dataset: ReferringSegDataset,
    position: int,
    output_dir: Path,
    method: str,
    split: str,
) -> dict[str, Any]:
    record = dataset.records[position]
    paths = artifact_paths(output_dir, record.index)
    return {
        "name": f"{record.name}_{record.index}",
        "method": method,
        "split": split,
        "instance_id": str(record.index),
        "image": str(record.image_path),
        "gt_mask": str(paths["gt"]),
        "prediction": str(paths["logits"]),
        "prediction_kind": "logits",
        "array_key": "logits",
        "threshold": 0.5,
        "pred_mask": str(paths["mask"]),
        "query": extract_target_text({}, record.user_text),
        "protocol": "pixellm_public_chat_autoregressive",
        "data_protocol": "stamp_flat_json_not_paper_reproduction",
    }


def main() -> int:
    args = parse_args()
    integer_values = (
        args.image_size,
        args.model_max_length,
        args.max_new_tokens,
        args.seg_token_num,
        args.image_feature_scale_num,
    )
    if any(value <= 0 for value in integer_values):
        raise ValueError("image/model/token dimensions must be positive.")
    if min(args.limit, args.offset) < 0:
        raise ValueError("limit and offset must be non-negative.")
    if not torch.cuda.is_available():
        raise RuntimeError("PixelLM export requires a CUDA GPU.")

    code_dir = args.pixellm_code_dir.expanduser().resolve()
    model_path = args.model_path.expanduser().resolve()
    vision_tower = args.vision_tower.expanduser().resolve()
    preprocessor_config = args.preprocessor_config.expanduser().resolve()
    required = {
        "PixelLM code": code_dir / "model" / "PixelLM.py",
        "PixelLM checkpoint": model_path / "config.json",
        "CLIP vision tower": vision_tower / "config.json",
        "PixelLM preprocessor": preprocessor_config,
    }
    missing = [f"{label}: {path}" for label, path in required.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("PixelLM inputs are incomplete:\n" + "\n".join(missing))

    sys.path.insert(0, str(code_dir))
    from transformers import AutoTokenizer, CLIPImageProcessor  # type: ignore

    from model.PixelLM import PixelLMForCausalLM  # type: ignore[import-not-found]
    from model.llava import conversation as conversation_lib  # type: ignore[import-not-found]
    from model.llava.mm_utils import tokenizer_image_token  # type: ignore[import-not-found]
    from model.llava.model.language_model.llava_llama import (  # type: ignore[import-not-found]
        LlavaConfig,
    )
    from model.segment_anything.utils.transforms import (  # type: ignore[import-not-found]
        ResizeLongestSide,
    )
    from utils.utils import (  # type: ignore[import-not-found]
        DEFAULT_IM_END_TOKEN,
        DEFAULT_IM_START_TOKEN,
        DEFAULT_IMAGE_TOKEN,
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_grad_enabled(False)
    device = torch.device("cuda:0")
    dtype = _dtype(args.precision)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
        local_files_only=True,
    )
    tokenizer.pad_token = tokenizer.unk_token
    total_seg_tokens = args.seg_token_num * args.image_feature_scale_num
    token_names = ["[SEG]"] if total_seg_tokens == 1 else [f"[SEG{i}]" for i in range(total_seg_tokens)]
    tokenizer.add_tokens(token_names)
    token_ids = [tokenizer(token, add_special_tokens=False).input_ids[0] for token in token_names]
    seg_token_idx: int | list[int] = token_ids[0] if len(token_ids) == 1 else token_ids

    config = LlavaConfig.from_pretrained(model_path, local_files_only=True)
    config.vision_tower = str(vision_tower)
    config.mm_vision_tower = str(vision_tower)
    model = PixelLMForCausalLM.from_pretrained(
        model_path,
        config=config,
        low_cpu_mem_usage=True,
        torch_dtype=dtype,
        vision_tower=str(vision_tower),
        seg_token_idx=seg_token_idx,
        seg_token_num=args.seg_token_num,
        image_feature_scale_num=args.image_feature_scale_num,
        pad_train_clip_images=True,
        resize_vision_tower=True,
        resize_vision_tower_size=448,
        vision_tower_for_mask=True,
        separate_mm_projector=True,
        local_files_only=True,
    )
    model.config.vision_tower = str(vision_tower)
    model.config.mm_vision_tower = str(vision_tower)
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.get_model().initialize_vision_modules(model.get_model().config)
    model.to(device=device, dtype=dtype)
    model.get_model().get_vision_tower().to(device=device, dtype=dtype)
    model.eval()

    clip_processor = CLIPImageProcessor.from_json_file(str(preprocessor_config))
    clip_size_value = clip_processor.size
    if isinstance(clip_size_value, dict):
        clip_size = int(clip_size_value.get("shortest_edge") or clip_size_value.get("height") or 448)
    else:
        clip_size = int(clip_size_value)
    clip_transform = ResizeLongestSide(clip_size)
    mask_transform = ResizeLongestSide(args.image_size)

    dataset = ReferringSegDataset(
        args.eval_json, data_root=args.data_root, limit=args.limit, offset=args.offset
    )
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    exported = reused = empty_predictions = model_calls = 0
    model_seconds = 0.0

    for position in tqdm(range(len(dataset)), desc="PixelLM expressions", dynamic_ncols=True):
        sample = dataset[position]
        paths = artifact_paths(args.output_dir, sample.record.index)
        if not args.overwrite and artifacts_complete(paths):
            reused += 1
            continue

        image_np = np.asarray(sample.image, dtype=np.uint8)
        original_size = tuple(int(value) for value in image_np.shape[:2])
        target_text = extract_target_text({}, sample.record.user_text)
        prompt = build_prompt(
            target_text,
            args.question_template,
            conversation_lib,
            args.conv_type,
            DEFAULT_IMAGE_TOKEN,
            DEFAULT_IM_START_TOKEN,
            DEFAULT_IM_END_TOKEN,
        )
        input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt").unsqueeze(0).to(device)

        clip_image_np = clip_transform.apply_image(image_np)
        clip_resize = tuple(int(value) for value in clip_image_np.shape[:2])
        clip_image = preprocess_square(
            torch.from_numpy(clip_image_np).permute(2, 0, 1).contiguous(), clip_size
        ).unsqueeze(0).to(device=device, dtype=dtype)

        mask_image_np = mask_transform.apply_image(image_np)
        mask_resize = tuple(int(value) for value in mask_image_np.shape[:2])
        mask_image = preprocess_square(
            torch.from_numpy(mask_image_np).permute(2, 0, 1).contiguous(), args.image_size
        ).unsqueeze(0).to(device=device, dtype=dtype)

        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            _, pred_masks, _, _ = model.evaluate(
                clip_image,
                mask_image,
                input_ids,
                [mask_resize],
                clip_resize_list=[clip_resize],
                original_size_list=[original_size],
                max_new_tokens=args.max_new_tokens,
                tokenizer=tokenizer,
            )
        torch.cuda.synchronize()
        model_seconds += time.perf_counter() - start
        model_calls += 1

        if not pred_masks or pred_masks[0].numel() == 0:
            logits = np.full(original_size, -20.0, dtype=np.float32)
            empty_predictions += 1
        else:
            raw_logits = pred_masks[0].detach().float().cpu().numpy()
            logits = merge_binary_logits(raw_logits)
            empty_predictions += int(not (logits > 0.0).any())
        atomic_save_logits(paths["logits"], logits)
        atomic_save_mask(paths["mask"], logits > 0.0)
        atomic_save_mask(paths["gt"], sample.mask.numpy())
        exported += 1

    rows = [
        _manifest_row(dataset, position, args.output_dir, args.method, args.split)
        for position in range(len(dataset))
    ]
    incomplete = [
        row["name"]
        for position, row in enumerate(rows)
        if not artifacts_complete(artifact_paths(args.output_dir, dataset.records[position].index))
    ]
    if incomplete:
        raise RuntimeError(f"PixelLM export has {len(incomplete)} incomplete samples: {incomplete[:5]}")
    manifest = args.output_dir / "manifest.jsonl"
    atomic_write_jsonl(manifest, rows)
    report = {
        "source": "official_pixellm_public_chat_inference",
        "data_protocol": "stamp_flat_json_not_paper_reproduction",
        "paper_reproduction": False,
        "samples": len(rows),
        "model": str(model_path),
        "vision_tower": str(vision_tower),
        "eval_json": str(args.eval_json.resolve()),
        "split": args.split,
        "precision": args.precision,
        "seg_token_num": args.seg_token_num,
        "image_feature_scale_num": args.image_feature_scale_num,
        "prediction_kind": "logits",
        "multiple_mask_reduction": "probabilistic_union",
        "model_calls": model_calls,
        "exported_samples": exported,
        "reused_samples": reused,
        "empty_predictions_in_new_exports": empty_predictions,
        "model_seconds": model_seconds,
        "model_seconds_per_new_sample": model_seconds / max(exported, 1),
        "manifest": str(manifest.resolve()),
    }
    atomic_write_json(args.output_dir / "export_summary.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
