from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


SPLITS = (
    "refcoco_val",
    "refcoco_testA",
    "refcoco_testB",
    "refcoco+_val",
    "refcoco+_testA",
    "refcoco+_testB",
    "refcocog_val",
    "refcocog_test",
)
MODELS = ("STAMP-2B", "STAMP-7B")
EXPECTED = {
    "refcoco_val": 10834,
    "refcoco_testA": 5657,
    "refcoco_testB": 5095,
    "refcoco+_val": 10758,
    "refcoco+_testA": 5726,
    "refcoco+_testB": 4889,
    "refcocog_val": 4896,
    "refcocog_test": 9602,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize full paired frozen-SAM runs.")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sam-model-type", choices=("vit_b", "vit_l", "vit_h"), required=True)
    return parser.parse_args()


def slug(value: str) -> str:
    return value.lower().replace("-", "").replace("+", "plus")


def main() -> int:
    args = parse_args()
    expected_protocol = (
        "stamp_official_frozen_sam_h_v1"
        if args.sam_model_type == "vit_h"
        else f"stamp_released_prompting_frozen_sam_{args.sam_model_type}_v1"
    )
    rows: list[dict[str, object]] = []
    for model in MODELS:
        for split in SPLITS:
            path = args.input_root / slug(model) / slug(split) / "eval_summary.json"
            if not path.is_file():
                raise FileNotFoundError(path)
            value = json.loads(path.read_text(encoding="utf-8"))
            if value.get("protocol") != expected_protocol:
                raise ValueError(f"Unexpected protocol in {path}: {value.get('protocol')}")
            samples = int(value.get("samples", 0))
            complete = samples == EXPECTED[split]
            disclosed_legacy_gap = (
                model == "STAMP-2B"
                and split in {"refcoco_testA", "refcoco+_testB", "refcocog_test"}
                and samples == EXPECTED[split] - 1
            )
            if not (complete or disclosed_legacy_gap):
                raise ValueError(
                    f"Incomplete SAM split {path}: {samples} != {EXPECTED[split]}"
                )
            rows.append(
                {
                    "model": model,
                    "split": split,
                    "samples": int(value["samples"]),
                    "base_cIoU": 100.0 * float(value["coarse_cIoU"]),
                    "sam_cIoU": 100.0
                    * float(value.get("sam_cIoU", value.get("coarse_sam_cIoU"))),
                    "sam_mIoU": 100.0
                    * float(value.get("sam_mean_iou", value.get("coarse_sam_mean_iou"))),
                    "sam_bIoU": 100.0
                    * float(
                        value.get("sam_boundary_iou", value.get("coarse_sam_boundary_iou"))
                    ),
                    "summary": str(path),
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "sam_full.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "sam_full.json").write_text(
        json.dumps({"sam_model_type": args.sam_model_type, "rows": rows}, indent=2),
        encoding="utf-8",
    )

    lines = [
        f"# Frozen SAM {args.sam_model_type} Full Comparison",
        "",
        "| Model | Split | N | Base cIoU | SAM cIoU | SAM mIoU | SAM bIoU |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {model} | {split} | {samples} | {base_cIoU:.2f} | "
            "{sam_cIoU:.2f} | {sam_mIoU:.2f} | {sam_bIoU:.2f} |".format(
                **row
            )
        )
    markdown = "\n".join(lines) + "\n"
    (args.output_dir / "sam_full.md").write_text(markdown, encoding="utf-8")
    print(markdown, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
