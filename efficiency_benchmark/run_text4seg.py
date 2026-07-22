from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from efficiency_benchmark.common import (
    ProcessGpuMemoryMonitor,
    assert_rtx_4090,
    cuda_elapsed,
    select_indices,
    write_report,
)
from training_free_refine import GpuTrainingFreeUncertaintyRefiner, TrainingFreeRefineConfig
from training_free_refine.data import ReferringSegDataset, extract_target_text
from training_free_refine.export_text4seg_masks import (
    compute_sam_logits,
    decode_prediction,
    decode_rle_mask,
    resolve_descriptor_grid_size,
    sample_mask_points,
    translate_labels,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified RTX 4090 Text4Seg end-to-end benchmark.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--text4seg-code-dir", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--vision-tower", required=True)
    parser.add_argument("--eval-json", type=Path, required=True)
    parser.add_argument("--sam-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variant", choices=("base", "freeref_gpu", "sam_h"), required=True)
    parser.add_argument("--descriptor-grid-size", type=int, default=24)
    parser.add_argument("--conv-mode", default="vicuna_v1")
    parser.add_argument("--max-new-tokens", type=int, default=3069)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow-other-gpu", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    gpu_name = assert_rtx_4090(args.allow_other_gpu)
    code_dir = args.text4seg_code_dir.resolve()
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

    descriptor_grid_size = resolve_descriptor_grid_size(
        args.model_path, args.descriptor_grid_size, None
    )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    model_config = LlavaConfig.from_pretrained(args.model_path, local_files_only=True)
    model_config.mm_vision_tower = args.vision_tower
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path, None, model_name, config=model_config
    )
    model.eval()
    predictor = None
    if args.variant == "sam_h":
        sam = sam_model_registry["vit_h"](checkpoint=str(args.sam_path.resolve()))
        predictor = SamPredictor(sam.to(dtype=torch.float32, device="cuda").eval())
    config = TrainingFreeRefineConfig()
    refiner = GpuTrainingFreeUncertaintyRefiner(config) if args.variant == "freeref_gpu" else None
    dataset = ReferringSegDataset(args.eval_json, data_root=args.root, limit=0, offset=0)
    warmup_indices, measured_indices = select_indices(
        len(dataset), args.warmup, args.samples, args.seed
    )

    def run(index: int) -> dict[str, float | int]:
        # Dataset/image I/O is intentionally outside the timed region.
        sample = dataset[index]
        image_array = np.asarray(sample.image)

        def infer_and_finish():
            target_text = extract_target_text({}, sample.record.user_text)
            question = QUESTION_PARTIAL[0].replace("[class_name]", target_text).replace(",", "")
            if model.config.mm_use_im_start_end:
                question = (
                    DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + question
                )
            else:
                question = DEFAULT_IMAGE_TOKEN + "\n" + question
            conversation = conv_templates[args.conv_mode].copy()
            conversation.append_message(conversation.roles[0], question)
            conversation.append_message(conversation.roles[1], None)
            prompt = conversation.get_prompt()
            resized = sample.image.resize((336, 336), Image.Resampling.BILINEAR)
            image_tensor = process_images([resized], image_processor, model.config)[0]
            input_ids = tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
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
            response = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            coarse_grid = decode_prediction(
                response, descriptor_grid_size, decode_rle_mask, translate_labels
            )
            coarse = F.interpolate(
                torch.as_tensor(coarse_grid, device="cuda", dtype=torch.float32).reshape(
                    1, 1, descriptor_grid_size, descriptor_grid_size
                ),
                size=image_array.shape[:2],
                mode="nearest",
            )[0, 0]
            return coarse_grid, coarse

        (coarse_grid, coarse), model_seconds = cuda_elapsed(infer_and_finish)
        if args.variant == "base":
            _, post_seconds = cuda_elapsed(lambda: coarse >= config.threshold)
        elif args.variant == "freeref_gpu":
            assert refiner is not None
            _, post_seconds = cuda_elapsed(
                lambda: refiner.refine_hard_mask(image_array, coarse)["refined_mask"]
            )
        else:
            assert predictor is not None

            def apply_sam():
                coarse_mask = (coarse >= config.threshold).cpu().numpy()
                if not coarse_mask.any():
                    return np.zeros_like(coarse_mask)
                predictor.set_image(image_array)
                logits = compute_sam_logits(coarse_grid, ResizeLongestSide)
                torch.manual_seed(args.seed + index)
                point_coords, point_labels = sample_mask_points(coarse_mask)
                sam_mask, _, low_res_logits = predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    mask_input=logits,
                    multimask_output=False,
                )
                for _ in range(2):
                    sam_mask, _, low_res_logits = predictor.predict(
                        point_coords=point_coords,
                        point_labels=point_labels,
                        mask_input=low_res_logits,
                        multimask_output=False,
                    )
                return sam_mask[0]

            _, post_seconds = cuda_elapsed(apply_sam)
        return {
            "index": index,
            "model_and_reconstruction_seconds": model_seconds,
            "postprocess_seconds": post_seconds,
            "total_seconds": model_seconds + post_seconds,
        }

    for index in tqdm(warmup_indices, desc=f"Text4Seg warmup {args.variant}"):
        run(index)
    monitor = ProcessGpuMemoryMonitor()
    monitor.start()
    rows = [
        run(index)
        for index in tqdm(measured_indices, desc=f"Text4Seg measured {args.variant}")
    ]
    peak_gpu_gib, memory_backend = monitor.finish()
    write_report(
        args.output_dir,
        {
            "method": "Text4Seg-p24",
            "variant": args.variant,
            "device": gpu_name,
            "protocol": (
                f"e2e_refcoco_testA_batch1_warmup{args.warmup}_random_seed{args.seed}"
            ),
            "warmup": args.warmup,
            "seed": args.seed,
            "model": args.model_path,
            "vision_tower": args.vision_tower,
            "eval_json": str(args.eval_json.resolve()),
            "sam_path": str(args.sam_path.resolve()) if args.variant == "sam_h" else None,
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
