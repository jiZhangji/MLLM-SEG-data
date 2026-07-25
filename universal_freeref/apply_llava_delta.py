from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply the official legacy LLaVA delta to a user-supplied licensed LLaMA-7B base."
    )
    parser.add_argument("--gsva-code-dir", type=Path, required=True)
    parser.add_argument("--base-model-path", type=Path, required=True)
    parser.add_argument("--delta-path", type=Path, required=True)
    parser.add_argument("--target-model-path", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import torch
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    gsva_dir = args.gsva_code_dir.expanduser().resolve()
    sys.path.insert(0, str(gsva_dir))
    from model.llava.model import LlavaLlamaForCausalLM

    base_path = args.base_model_path.expanduser().resolve()
    delta_path = args.delta_path.expanduser().resolve()
    target_path = args.target_model_path.expanduser().resolve()
    print(f"Loading licensed base model: {base_path}")
    base = AutoModelForCausalLM.from_pretrained(
        base_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    print(f"Loading official LLaVA delta: {delta_path}")
    delta = LlavaLlamaForCausalLM.from_pretrained(
        delta_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(delta_path, use_fast=False)
    base_state = base.state_dict()
    for name, parameter in tqdm(delta.state_dict().items(), desc="Applying LLaVA delta"):
        if name not in base_state:
            if name not in {"model.mm_projector.weight", "model.mm_projector.bias"}:
                raise KeyError(f"Unexpected delta-only parameter: {name}")
            continue
        base_parameter = base_state[name]
        if parameter.shape == base_parameter.shape:
            parameter.data.add_(base_parameter)
        else:
            if name not in {"model.embed_tokens.weight", "lm_head.weight"}:
                raise ValueError(
                    f"Unexpected shape mismatch for {name}: {parameter.shape} vs {base_parameter.shape}"
                )
            parameter.data[: base_parameter.shape[0], : base_parameter.shape[1]].add_(base_parameter)

    target_path.mkdir(parents=True, exist_ok=True)
    delta.save_pretrained(
        target_path,
        max_shard_size="5GB",
        safe_serialization=False,
    )
    tokenizer.save_pretrained(target_path)
    print(f"Merged legacy LLaVA model saved to: {target_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
