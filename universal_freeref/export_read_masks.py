from __future__ import annotations

import argparse
import json
import sys
from functools import partial
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run READ's official model/dataset path with the paper's teacher-forced "
            "[SEG] validation protocol and export SAM logits for paired FreeRef."
        )
    )
    parser.add_argument("--read-code-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--vision-tower", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--dataset", choices=("refcoco", "refcoco+", "refcocog"), required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--method", default="READ-LLaVA-v1.5-7B-official")
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--model-max-length", type=int, default=2048)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    code_dir = args.read_code_dir.expanduser().resolve()
    sys.path.insert(0, str(code_dir))

    import torch
    import tqdm
    from torch.utils.data import DataLoader, Subset

    from dataloaders.test_dataset import TestReferDataset
    from dataloaders.trainval_dataset import collate_fn_val
    from model.READ import load_pretrained_model_READ
    from model.llava import conversation as conversation_lib
    from transformers import AutoConfig
    from utils import prepare_input, random_seed

    from .read_export import READOfficialExporter

    random_seed(seed=42)
    conversation_lib.default_conversation = conversation_lib.conv_templates["llava_v1"]
    vision_tower_path = args.vision_tower.expanduser().resolve()
    model_config = AutoConfig.from_pretrained(args.model_path.expanduser().resolve())
    # READ's public config records a Hugging Face model ID.  Override both names
    # used by the vendored LLaVA loader so offline evaluation consumes the exact
    # local CLIP snapshot prepared by our asset script.
    model_config.vision_tower = str(vision_tower_path)
    model_config.mm_vision_tower = str(vision_tower_path)
    tokenizer, model, vision_tower, _ = load_pretrained_model_READ(
        model_path=str(args.model_path.expanduser().resolve()),
        config=model_config,
        vision_tower=str(vision_tower_path),
        model_max_length=args.model_max_length,
    )
    tokenizer.padding_side = "right"
    model.eval()
    dataset = TestReferDataset(
        str(args.dataset_dir.expanduser().resolve()),
        vision_tower.image_processor,
        1024,
        datasetname=args.dataset,
        train_test_split=args.split,
        use_val_mode=True,
        use_test_mode=False,
    )
    start = max(args.offset, 0)
    stop = len(dataset) if args.limit <= 0 else min(len(dataset), start + args.limit)
    if start >= stop:
        raise ValueError(f"READ selection is empty: offset={start}, stop={stop}.")
    if start != 0 or stop != len(dataset):
        dataset = Subset(dataset, list(range(start, stop)))
    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=args.workers,
        shuffle=False,
        drop_last=False,
        pin_memory=False,
        collate_fn=partial(
            collate_fn_val,
            tokenizer=tokenizer,
            use_mm_start_end=getattr(model.config, "mm_use_im_start_end", False),
            padding="right",
        ),
    )
    exporter = READOfficialExporter(
        args.output_dir.expanduser().resolve(),
        args.method,
        f"{args.dataset}_{args.split}",
    )
    with torch.inference_mode():
        for input_dict in tqdm.tqdm(loader, desc=f"READ {args.dataset}/{args.split}"):
            torch.cuda.empty_cache()
            input_dict = prepare_input(input_dict, args.precision, is_cuda=True)
            output_dict = model(**input_dict)
            exporter.record(input_dict, output_dict)
    report = exporter.finalize()
    print(json.dumps(report, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
