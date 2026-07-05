from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def inspect_tensor_file(path: Path) -> dict[str, object]:
    payload = torch.load(path, map_location="cpu")
    required = {"mask_logits", "mask_hidden", "grid_hw"}
    missing = required - set(payload)
    if missing:
        raise KeyError(f"Missing keys in {path}: {sorted(missing)}")

    mask_logits = payload["mask_logits"]
    mask_hidden = payload["mask_hidden"]
    grid_hw = tuple(payload["grid_hw"])

    if mask_logits.ndim != 3 or mask_logits.shape[-1] != 2:
        raise ValueError(f"mask_logits must be [B, N, 2], got {tuple(mask_logits.shape)}")
    if mask_hidden.ndim != 3:
        raise ValueError(f"mask_hidden must be [B, N, D], got {tuple(mask_hidden.shape)}")
    if mask_logits.shape[:2] != mask_hidden.shape[:2]:
        raise ValueError("mask_logits and mask_hidden token dimensions differ.")

    fg_prob = torch.softmax(mask_logits, dim=-1)[..., 1]
    grid_h, grid_w = grid_hw
    num_tokens = int(mask_logits.shape[1])

    return {
        "mask_logits_shape": list(mask_logits.shape),
        "mask_hidden_shape": list(mask_hidden.shape),
        "grid_hw": [grid_h, grid_w],
        "num_mask_tokens": num_tokens,
        "grid_matches_tokens": grid_h * grid_w == num_tokens,
        "coarse_foreground_ratio": float((fg_prob >= 0.5).float().mean().item()),
        "fg_probability_mean": float(fg_prob.mean().item()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path, help="Torch file with STAMP refinement tensors.")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    report = inspect_tensor_file(args.input)
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
