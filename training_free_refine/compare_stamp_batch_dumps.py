from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def dump_map(path: Path) -> dict[str, Path]:
    return {item.name: item for item in path.glob("*.pt")}


def main() -> int:
    parser = argparse.ArgumentParser(description="Check STAMP batch export against the batch-size-one reference.")
    parser.add_argument("--single-dir", type=Path, required=True)
    parser.add_argument("--batch-dir", type=Path, required=True)
    parser.add_argument("--expected", type=int, required=True)
    parser.add_argument("--min-token-agreement", type=float, default=0.9999)
    parser.add_argument("--max-logit-abs-diff", type=float, default=0.05)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    single = dump_map(args.single_dir)
    batched = dump_map(args.batch_dir)
    common = sorted(single.keys() & batched.keys())
    rows = []
    total_equal = 0
    total_tokens = 0
    max_abs_diff = 0.0
    text_matches = 0
    grid_matches = 0

    for name in common:
        left = torch.load(single[name], map_location="cpu", weights_only=False)
        right = torch.load(batched[name], map_location="cpu", weights_only=False)
        left_grid = tuple(int(value) for value in left["grid_hw"])
        right_grid = tuple(int(value) for value in right["grid_hw"])
        grid_match = left_grid == right_grid
        grid_matches += int(grid_match)
        left_logits = left["mask_logits"].float()
        right_logits = right["mask_logits"].float()
        if left_logits.shape == right_logits.shape:
            difference = (left_logits - right_logits).abs()
            sample_max = float(difference.max().item())
            left_pred = left_logits.argmax(dim=-1)
            right_pred = right_logits.argmax(dim=-1)
            equal = int((left_pred == right_pred).sum().item())
            tokens = int(left_pred.numel())
        else:
            sample_max = 1.0e30
            equal = 0
            tokens = max(int(left_logits.shape[-2]), int(right_logits.shape[-2]))
        total_equal += equal
        total_tokens += tokens
        max_abs_diff = max(max_abs_diff, sample_max)
        text_match = left.get("response_text") == right.get("response_text")
        text_matches += int(text_match)
        rows.append(
            {
                "name": name,
                "grid_match": grid_match,
                "text_match": text_match,
                "token_agreement": equal / max(tokens, 1),
                "max_logit_abs_diff": sample_max,
            }
        )

    token_agreement = total_equal / max(total_tokens, 1)
    passed = (
        len(common) == args.expected
        and grid_matches == args.expected
        and text_matches == args.expected
        and token_agreement >= args.min_token_agreement
        and max_abs_diff <= args.max_logit_abs_diff
    )
    report = {
        "passed": passed,
        "expected": args.expected,
        "compared": len(common),
        "grid_matches": grid_matches,
        "text_matches": text_matches,
        "token_agreement": token_agreement,
        "max_logit_abs_diff": max_abs_diff,
        "thresholds": {
            "min_token_agreement": args.min_token_agreement,
            "max_logit_abs_diff": args.max_logit_abs_diff,
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
