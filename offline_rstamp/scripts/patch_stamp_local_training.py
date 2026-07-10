#!/usr/bin/env python3
"""Patch STAMP for local single-node smoke training.

This patch is intentionally conservative:

1. Do not touch upstream launch scripts.
2. Make `train/main_uni.py` configurable through environment variables.
3. Avoid requiring flash-attn by switching `flash_attention_2` to `sdpa` by default.
4. Optionally inject `structured_prior_text` into the first user message so the
   R-STAMP smoke run can test whether explicit priors help.

Run this script on the server after pulling MLLM-SEG-data:

    python offline_rstamp/scripts/patch_stamp_local_training.py \
      --stamp-code-dir /.../MLLM-SEG/code/STAMP

The script creates a `.bak_rstamp_local` backup the first time it edits
`train/main_uni.py`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re


MARKER = "# ---- R-STAMP local training patch ----"


def replace_once(text: str, old: str, new: str) -> str:
    if old not in text:
        return text
    return text.replace(old, new, 1)


def patch_imports(text: str) -> str:
    if "import copy" not in text:
        text = text.replace("import json\n", "import json\nimport copy\n", 1)
    return text


def patch_cudnn_control(text: str) -> str:
    marker = "STAMP_DISABLE_CUDNN"
    if marker in text:
        return text
    needle = "IS_MAIN_PROCESS = os.environ.get(\"LOCAL_RANK\", \"-1\") in [\"-1\", \"0\"]\n"
    insert = (
        "IS_MAIN_PROCESS = os.environ.get(\"LOCAL_RANK\", \"-1\") in [\"-1\", \"0\"]\n"
        "if os.environ.get(\"STAMP_DISABLE_CUDNN\", \"0\") == \"1\":\n"
        "    torch.backends.cudnn.enabled = False\n"
        "    if IS_MAIN_PROCESS:\n"
        "        print(\"--- Disabled cuDNN for local smoke training ---\")\n"
    )
    return text.replace(needle, insert, 1)


def patch_flash_attention(text: str) -> str:
    # Be aggressive: different STAMP snapshots may use single quotes, double
    # quotes, or may already have been partly patched. The local smoke run must
    # not require flash-attn.
    text = text.replace('attn_implementation="flash_attention_2",', 'attn_implementation=os.environ.get("STAMP_ATTN_IMPL", "sdpa"),')
    text = text.replace("attn_implementation='flash_attention_2',", 'attn_implementation=os.environ.get("STAMP_ATTN_IMPL", "sdpa"),')
    text = text.replace('"flash_attention_2"', 'os.environ.get("STAMP_ATTN_IMPL", "sdpa")')
    text = text.replace("'flash_attention_2'", 'os.environ.get("STAMP_ATTN_IMPL", "sdpa")')
    return text


def patch_dataset_function(text: str) -> str:
    old = """        base_path = 'playground/data/json_files/'
        json_files = [
            'all_valid_llava_data_1000.json',
            'refclef_formatted_all_sentences_doubled_mp.json',
            'refcocog_formatted_all_sentences_doubled_mp.json',
            'refcoco_formatted_all_sentences_doubled_mp.json',
            'refcoco+_formatted_all_sentences_doubled_mp.json'
        ]
"""
    new = f"""        {MARKER}
        base_path = os.environ.get("STAMP_JSON_DIR", "playground/data/json_files_baseline")
        json_files_env = os.environ.get("STAMP_JSON_FILES", "")
        if json_files_env.strip():
            json_files = [x.strip() for x in json_files_env.split(",") if x.strip()]
        else:
            json_files = [
                "all_valid_llava_data_1000.json",
                "refcocog_formatted_all_sentences_doubled_mp.json",
                "refcoco_formatted_all_sentences_doubled_mp.json",
                "refcoco+_formatted_all_sentences_doubled_mp.json",
            ]
"""
    text = replace_once(text, old, new)

    old_loop = """        # Load all data
        all_data = []
        for file_path in file_paths:
            if IS_MAIN_PROCESS:
                print(f"--- Loading {file_path} ---")
            with open(file_path, 'r') as f:
                data = json.load(f)
                if 'llava' not in file_path:
                    data = data * 3
                all_data.extend(data)
"""
    new_loop = """        # Load all data
        all_data = []
        stamp_max_samples = int(os.environ.get("STAMP_MAX_SAMPLES", "0"))
        stamp_repeat_non_llava = int(os.environ.get("STAMP_REPEAT_NON_LLAVA", "1"))
        for file_path in file_paths:
            if not os.path.exists(file_path):
                if IS_MAIN_PROCESS:
                    print(f"--- Skipping missing file: {file_path} ---")
                continue
            if IS_MAIN_PROCESS:
                print(f"--- Loading {file_path} ---")
            with open(file_path, 'r') as f:
                data = json.load(f)
                if 'llava' not in file_path and stamp_repeat_non_llava > 1:
                    data = data * stamp_repeat_non_llava
                all_data.extend(data)
        if stamp_max_samples > 0:
            all_data = all_data[:stamp_max_samples]
"""
    text = replace_once(text, old_loop, new_loop)

    old_processing = """        processed_data = []
        IMAGE_RAW_ROOT_PATH = '/apdcephfs_qy4/share_302593112/realzliu/dataset_open/lisa_dataset'
        IMAGE_ROOT_PATH = '/efficient_sag4text/seg_data'
        for example in tqdm(all_data):
            # 1. Process image paths
            images = example['images']
            images = [os.path.join(IMAGE_RAW_ROOT_PATH, i) for i in images]
            images = [i.replace(IMAGE_RAW_ROOT_PATH, IMAGE_ROOT_PATH) for i in images]
            images = [i.replace('/coco_2014/', '/mscoco/images/') for i in images]
            example['images'] = images

            # 2. Normalize the 'masks' key
            # This is the key to solving the problem: ensure each example has a 'masks' key
            if 'masks' not in example:
                example['masks'] = None

            processed_data.append(example)


        # Create Dataset object
        ds = Dataset.from_list(all_data)
"""
    new_processing = """        processed_data = []
        use_structured_prior = os.environ.get("STAMP_USE_STRUCTURED_PRIOR", "0") == "1"
        for raw_example in tqdm(all_data):
            example = copy.deepcopy(raw_example)
            # Data prepared by offline_rstamp already uses absolute image/mask paths.
            if 'masks' not in example:
                example['masks'] = None

            if use_structured_prior and example.get("structured_prior_text"):
                prior_text = example["structured_prior_text"]
                try:
                    content = example["messages"][0]["content"]
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            part["text"] = "Structured reasoning prior: " + prior_text + "\\n" + part["text"]
                            break
                except Exception as exc:
                    if IS_MAIN_PROCESS:
                        print(f"WARNING: failed to inject structured prior: {exc}")

            processed_data.append(example)

        # Create Dataset object
        ds = Dataset.from_list(processed_data)
"""
    text = replace_once(text, old_processing, new_processing)
    return text


def patch_processor_pixels(text: str) -> str:
    text = text.replace(
        "        min_pixels = 1024 * 28 * 28\n        max_pixels = 1280 * 28 * 28\n",
        "        min_pixels = int(os.environ.get(\"STAMP_MIN_PIXELS\", str(256 * 28 * 28)))\n"
        "        max_pixels = int(os.environ.get(\"STAMP_MAX_PIXELS\", str(512 * 28 * 28)))\n",
    )
    return text


def patch_training_args(text: str) -> str:
    replacements = {
        "num_train_epochs=2,": 'num_train_epochs=float(os.environ.get("STAMP_NUM_EPOCHS", "1")),',
        "per_device_train_batch_size=8,": 'per_device_train_batch_size=int(os.environ.get("STAMP_BATCH_SIZE", "1")),',
        "gradient_accumulation_steps=4,": 'gradient_accumulation_steps=int(os.environ.get("STAMP_GRAD_ACCUM", "8")),',
        "learning_rate=3e-5,": 'learning_rate=float(os.environ.get("STAMP_LR", "3e-5")),',
        "logging_steps=1,": 'logging_steps=int(os.environ.get("STAMP_LOGGING_STEPS", "1")),',
        "save_steps=1000,": 'save_steps=int(os.environ.get("STAMP_SAVE_STEPS", "200")),',
        'report_to="wandb",': 'report_to=os.environ.get("STAMP_REPORT_TO", "none"),',
        "max_length=4096,": 'max_length=int(os.environ.get("STAMP_MAX_LENGTH", "2048")),',
        "gradient_checkpointing=True,": 'gradient_checkpointing=os.environ.get("STAMP_GRADIENT_CHECKPOINTING", "0") == "1",',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def patch_lora(text: str) -> str:
    replacements = {
        "r=64,": 'r=int(os.environ.get("STAMP_LORA_R", "16")),',
        "lora_alpha=128,": 'lora_alpha=int(os.environ.get("STAMP_LORA_ALPHA", "32")),',
        "lora_dropout=0.05,": 'lora_dropout=float(os.environ.get("STAMP_LORA_DROPOUT", "0.05")),',
    }
    for old, new in replacements.items():
        text = text.replace(old, new, 1)
    return text


def patch_entrypoint_model_and_output(text: str) -> str:
    """Make the script bottom use MODEL_NAME and OUT_DIR env vars.

    Upstream snapshots often ignore CLI args and instantiate QwenVLSFTTrainer
    with hard-coded paths at module bottom. This causes baseline and R-STAMP
    smoke runs to overwrite the same `output/qwen_vl_seg_sft/uni` directory.
    """
    if 'os.environ.get("MODEL_NAME"' in text and 'os.environ.get("OUT_DIR"' in text:
        return text

    replacement = (
        "trainer = QwenVLSFTTrainer(\n"
        "    model_name=os.environ.get(\"MODEL_NAME\", \"Qwen/Qwen2-VL-2B-Instruct\"),\n"
        "    output_dir=os.environ.get(\"OUT_DIR\", \"output/qwen_vl_seg_sft/uni\"),\n"
        ")"
    )

    # First try a broad regex for common formatting.
    pattern = re.compile(r"trainer\s*=\s*QwenVLSFTTrainer\((?:.|\n)*?\)\s*\n\s*trainer\.train\(\)", flags=re.S)
    new_text, n = pattern.subn(replacement + "\n    trainer.train()", text, count=1)
    if n:
        return new_text

    # Robust fallback: find the last textual occurrence and replace the balanced
    # call expression, regardless of the argument names/formatting.
    anchor = "trainer = QwenVLSFTTrainer("
    start = text.rfind(anchor)
    if start == -1:
        text += (
            "\n# R-STAMP patch warning: could not find QwenVLSFTTrainer entrypoint.\n"
            "# Please manually ensure MODEL_NAME and OUT_DIR environment variables are used.\n"
        )
        return text

    open_paren = text.find("(", start)
    depth = 0
    end = None
    for idx in range(open_paren, len(text)):
        ch = text[idx]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                end = idx + 1
                break
    if end is None:
        text += (
            "\n# R-STAMP patch warning: could not parse QwenVLSFTTrainer entrypoint parentheses.\n"
        )
        return text

    return text[:start] + replacement + text[end:]


def patch_file(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    backup = path.with_suffix(path.suffix + ".bak_rstamp_local")
    if not backup.exists():
        backup.write_text(text, encoding="utf-8")

    text = patch_imports(text)
    text = patch_cudnn_control(text)
    text = patch_flash_attention(text)
    text = patch_processor_pixels(text)
    text = patch_dataset_function(text)
    text = patch_training_args(text)
    text = patch_lora(text)
    text = patch_entrypoint_model_and_output(text)

    path.write_text(text, encoding="utf-8")
    print(f"Patched: {path}")
    print(f"Backup:  {backup}")


def patch_seg_trainer_speed(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    backup = path.with_suffix(path.suffix + ".bak_rstamp_local")
    if not backup.exists():
        backup.write_text(text, encoding="utf-8")

    guarded_empty_cache = (
        "        if os.environ.get(\"STAMP_EMPTY_CACHE_EACH_STEP\", \"0\") == \"1\":\n"
        "            torch.cuda.empty_cache()\n"
    )
    text = re.sub(
        r'        if os\.environ\.get\("STAMP_EMPTY_CACHE_EACH_STEP", "0"\) == "1":\n'
        r'(?:\s*if os\.environ\.get\("STAMP_EMPTY_CACHE_EACH_STEP", "0"\) == "1":\n)*'
        r'\s*torch\.cuda\.empty_cache\(\)\n',
        guarded_empty_cache,
        text,
        count=1,
    )
    if guarded_empty_cache not in text:
        text = text.replace("        torch.cuda.empty_cache()\n", guarded_empty_cache, 1)

    old_lm = """        # This call computes the standard cross-entropy loss on the 'text' field
        # It internally uses inputs['input_ids'] and inputs['labels']
        outputs = super().compute_loss(model, inputs, return_outputs=True)
        loss_lm = outputs[0]
"""
    new_lm = """        # This call computes the standard cross-entropy loss on the 'text' field.
        # Head-only uncertainty experiments can skip it because the language
        # backbone is frozen and only the segmentation head is trainable.
        if os.environ.get("STAMP_SKIP_LM_LOSS", "0") == "1":
            outputs = None
            loss_lm = torch.tensor(0.0, device=model.device)
        else:
            outputs = super().compute_loss(model, inputs, return_outputs=True)
            loss_lm = outputs[0]
"""
    text = text.replace(old_lm, new_lm, 1)

    text = text.replace(
        "            start_p = 0\n"
        "            num_gt = 0\n\n"
        "            for i in range(batch_size):\n",
        "            start_p = 0\n"
        "            num_gt = 0\n"
        "            loss_fn_weighted = WeightedDiceBCELoss(alpha=0.3, beta=0.7)\n\n"
        "            for i in range(batch_size):\n",
        1,
    )

    text = text.replace(
        "\n                    loss_fn_weighted = WeightedDiceBCELoss(alpha=0.3, beta=0.7)\n"
        "                    loss_seg_ = loss_fn_weighted(mask_pred.float(), binary_gt_labels.unsqueeze(1).float())\n",
        "\n                    loss_seg_ = loss_fn_weighted(mask_pred.float(), binary_gt_labels.unsqueeze(1).float())\n",
        1,
    )

    img_show_line = "                    img_show = T.ToTensor()(all_images[i][-1]).permute(1, 2, 0).cpu()\n"
    guarded_img_show = (
        "                    if IS_MAIN_PROCESS and os.environ.get(\"STAMP_SAVE_SEG_VIS\", \"0\") == \"1\":\n"
        "                        img_show = T.ToTensor()(all_images[i][-1]).permute(1, 2, 0).cpu()\n"
    )
    duplicate_guarded_img_show = (
        "                    if IS_MAIN_PROCESS and os.environ.get(\"STAMP_SAVE_SEG_VIS\", \"0\") == \"1\":\n"
        "                        if IS_MAIN_PROCESS and os.environ.get(\"STAMP_SAVE_SEG_VIS\", \"0\") == \"1\":\n"
        "                        img_show = T.ToTensor()(all_images[i][-1]).permute(1, 2, 0).cpu()\n"
    )
    text = text.replace(duplicate_guarded_img_show, guarded_img_show, 1)
    if guarded_img_show not in text:
        text = text.replace(img_show_line, guarded_img_show, 1)

    text = text.replace(
        "            if IS_MAIN_PROCESS:\n"
        "                import matplotlib.pyplot as plt\n",
        "            if IS_MAIN_PROCESS and os.environ.get(\"STAMP_SAVE_SEG_VIS\", \"0\") == \"1\":\n"
        "                import matplotlib.pyplot as plt\n",
        1,
    )

    path.write_text(text, encoding="utf-8")
    print(f"Patched speed knobs: {path}")
    print(f"Backup:              {backup}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stamp-code-dir", required=True, type=Path)
    args = parser.parse_args()
    target = args.stamp_code_dir / "train" / "main_uni.py"
    if not target.exists():
        raise FileNotFoundError(target)
    patch_file(target)
    seg_trainer = args.stamp_code_dir / "train" / "seg_trainer.py"
    if not seg_trainer.exists():
        raise FileNotFoundError(seg_trainer)
    patch_seg_trainer_speed(seg_trainer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
