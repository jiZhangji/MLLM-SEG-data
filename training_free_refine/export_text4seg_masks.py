from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from onepass_stamp.data import OnePassDataset, extract_target_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Text4Seg on the same flat JSON used by STAMP evaluation.")
    parser.add_argument("--text4seg-code-dir", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--vision-tower", default="openai/clip-vit-large-patch14-336")
    parser.add_argument("--eval-json", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--sam-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--visual-tokens", type=int, default=24)
    parser.add_argument("--conv-mode", default="vicuna_v1")
    parser.add_argument("--max-new-tokens", type=int, default=3069)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def decode_prediction(text: str, grid_size: int, decode_mask, translate_sequence) -> np.ndarray:
    count = grid_size * grid_size
    try:
        labels = text.split("<seg>", 1)[1].split("</seg>", 1)[0]
        values = translate_sequence(decode_mask(labels))
    except (IndexError, KeyError, TypeError, ValueError):
        values = []
    if not values:
        values = [0] * count
    elif len(values) < count:
        values = values + [values[-1]] * (count - len(values))
    else:
        values = values[:count]
    return (np.asarray(values).reshape(grid_size, grid_size) > 0).astype(np.uint8)


def decode_rle_mask(encoded: str) -> str:
    values = []
    for row in encoded.strip("\n").split("\n "):
        for token in row.split("| "):
            label, count = token.split(" *")
            values.extend([label] * int(count))
    return "|".join(values)


def translate_labels(sequence: str) -> list[int]:
    categories = [value.strip() for value in sequence.split("|")]
    unique = list(dict.fromkeys(categories))
    if "others" in unique:
        unique.remove("others")
        unique.insert(0, "others")
    mapping = {category: index for index, category in enumerate(unique)}
    return [mapping[value] for value in categories]


def compute_sam_logits(mask: np.ndarray, resize_longest_side, eps: float = 1e-3) -> np.ndarray:
    probability = np.where(np.asarray(mask) > 0, 1.0 - eps, eps).astype(np.float32)
    logits = np.log(probability / (1.0 - probability))
    transform = resize_longest_side(256)
    logits = transform.apply_image(logits[..., None])
    logits = np.asarray(logits).squeeze(-1) if np.asarray(logits).ndim == 3 else np.asarray(logits)
    pad_height = 256 - logits.shape[0]
    pad_width = 256 - logits.shape[1]
    logits = np.pad(logits, ((0, pad_height), (0, pad_width)), mode="constant", constant_values=0)
    return logits[None].astype(np.float32)


def sample_mask_points(mask: np.ndarray, count: int = 10) -> tuple[np.ndarray, np.ndarray]:
    hard_mask = torch.from_numpy(np.asarray(mask) > 0)
    height, width = hard_mask.shape
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    points = []
    labels = []
    for value, label in [(True, 1), (False, 0)]:
        selected_x = x[hard_mask == value]
        selected_y = y[hard_mask == value]
        indices = torch.randperm(selected_x.numel())[:count]
        points.append(torch.stack([selected_x[indices], selected_y[indices]], dim=1))
        labels.extend([label] * len(indices))
    return torch.cat(points).numpy().astype(np.float64), np.asarray(labels, dtype=np.uint8)


def save_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(path)


def main() -> int:
    args = parse_args()
    if args.visual_tokens <= 1 or args.limit < 0 or args.offset < 0:
        raise ValueError("visual-tokens must exceed one; limit and offset must be non-negative.")
    code_dir = args.text4seg_code_dir.resolve()
    if not (code_dir / "llava").is_dir():
        raise FileNotFoundError(f"Text4Seg llava package was not found below {code_dir}.")
    sys.path.insert(0, str(code_dir))

    from llava.constants import (  # type: ignore[import-not-found]
        DEFAULT_IMAGE_TOKEN,
        DEFAULT_IM_END_TOKEN,
        DEFAULT_IM_START_TOKEN,
        IMAGE_TOKEN_INDEX,
    )
    from llava.conversation import conv_templates  # type: ignore[import-not-found]
    from llava.eval.question_answer_list import QUESTION_PARTIAL  # type: ignore[import-not-found]
    from llava.mm_utils import (  # type: ignore[import-not-found]
        get_model_name_from_path,
        process_images,
        tokenizer_image_token,
    )
    from llava.model.builder import load_pretrained_model  # type: ignore[import-not-found]
    from llava.model.language_model.llava_llama import LlavaConfig  # type: ignore[import-not-found]
    from llava.model.segment_anything import SamPredictor, sam_model_registry  # type: ignore[import-not-found]
    from llava.model.segment_anything.utils.transforms import ResizeLongestSide  # type: ignore[import-not-found]
    from llava.utils import disable_torch_init  # type: ignore[import-not-found]

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    model_config = LlavaConfig.from_pretrained(args.model_path)
    checkpoint_vision_tower = getattr(model_config, "mm_vision_tower", None)
    model_config.mm_vision_tower = args.vision_tower
    print(
        f"Text4Seg vision tower: {checkpoint_vision_tower!r} -> {args.vision_tower!r}",
        flush=True,
    )
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path,
        None,
        model_name,
        config=model_config,
    )
    sam = sam_model_registry["vit_h"](checkpoint=str(args.sam_path))
    sam = sam.to(dtype=torch.float32, device="cuda")
    predictor = SamPredictor(sam)

    dataset = OnePassDataset(args.eval_json, data_root=args.data_root, limit=args.limit, offset=args.offset)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = args.output_dir / "pred_masks"
    sam_dir = args.output_dir / "sam_masks"
    gt_dir = args.output_dir / "gt_masks"
    manifest_rows = []
    model_seconds = 0.0
    model_calls = 0

    for sample in tqdm(dataset, desc="Text4Seg inference", dynamic_ncols=True):
        record = sample.record
        sample_id = f"{record.index:08d}"
        pred_path = pred_dir / f"{sample_id}.png"
        sam_path = sam_dir / f"{sample_id}.png"
        gt_path = gt_dir / f"{sample_id}.png"
        target_text = extract_target_text({}, record.user_text)

        if not (pred_path.exists() and sam_path.exists() and gt_path.exists()):
            question = QUESTION_PARTIAL[0].replace("[class_name]", target_text).replace(",", "")
            if model.config.mm_use_im_start_end:
                question = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + question
            else:
                question = DEFAULT_IMAGE_TOKEN + "\n" + question
            conversation = conv_templates[args.conv_mode].copy()
            conversation.append_message(conversation.roles[0], question)
            conversation.append_message(conversation.roles[1], None)
            prompt = conversation.get_prompt()

            resized = sample.image.resize((336, 336), Image.Resampling.BILINEAR)
            image_tensor = process_images([resized], image_processor, model.config)[0]
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            predictor.set_image(np.asarray(sample.image))
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids.unsqueeze(0).to(device="cuda", non_blocking=True),
                    images=[image_tensor.to(dtype=torch.float16, device="cuda", non_blocking=True)],
                    image_sizes=(336, 336),
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                )
            torch.cuda.synchronize()
            model_seconds += time.perf_counter() - start
            model_calls += 1
            response = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            coarse_grid = decode_prediction(response, args.visual_tokens, decode_rle_mask, translate_labels)
            coarse = F.interpolate(
                torch.from_numpy(coarse_grid).reshape(1, 1, args.visual_tokens, args.visual_tokens).double(),
                size=(sample.image.height, sample.image.width),
                mode="nearest",
            )[0, 0].numpy().astype(np.uint8)

            if coarse.any():
                logits = compute_sam_logits(coarse_grid, ResizeLongestSide)
                point_coords, point_labels = sample_mask_points(coarse)
                sam_mask, _, sam_logits = predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    mask_input=logits,
                    multimask_output=False,
                )
                for _ in range(2):
                    sam_mask, _, sam_logits = predictor.predict(
                        point_coords=point_coords,
                        point_labels=point_labels,
                        mask_input=sam_logits,
                        multimask_output=False,
                    )
                sam_prediction = sam_mask[0].astype(np.uint8)
            else:
                sam_prediction = np.zeros((sample.image.height, sample.image.width), dtype=np.uint8)

            save_mask(pred_path, coarse)
            save_mask(sam_path, sam_prediction)
            save_mask(gt_path, sample.mask.numpy().astype(np.uint8))

        manifest_rows.append(
            {
                "name": f"{record.name}_{record.index}",
                "index": record.index,
                "image": str(record.image_path),
                "pred_mask": str(pred_path),
                "sam_mask": str(sam_path),
                "gt_mask": str(gt_path),
                "query": target_text,
            }
        )

    manifest_path = args.output_dir / "manifest.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in manifest_rows),
        encoding="utf-8",
    )
    report = {
        "samples": len(manifest_rows),
        "model": args.model_path,
        "eval_json": str(args.eval_json),
        "visual_tokens": args.visual_tokens,
        "model_calls": model_calls,
        "reused_samples": len(manifest_rows) - model_calls,
        "model_seconds_per_call": model_seconds / max(model_calls, 1),
        "manifest": str(manifest_path),
    }
    (args.output_dir / "export_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
