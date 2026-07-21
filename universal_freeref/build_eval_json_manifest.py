from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from training_free_refine.data import ReferringSegDataset, extract_target_text

from .export_utils import atomic_save_mask, atomic_write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Align index-named external predictions with a flat STAMP evaluation JSON."
    )
    parser.add_argument("--method", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--eval-json", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--prediction-template", default="{index:08d}.png")
    parser.add_argument("--prediction-kind", choices=("mask", "probability", "logits"), default="mask")
    parser.add_argument("--array-key")
    parser.add_argument("--foreground-channel", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--allow-missing", action="store_true")
    return parser.parse_args()


def prediction_path(root: Path, template: str, index: int, name: str) -> Path:
    return root / template.format(index=index, name=name, stem=Path(name).stem)


def build_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    dataset = ReferringSegDataset(
        args.eval_json, data_root=args.data_root, limit=args.limit, offset=args.offset
    )
    rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for position in range(len(dataset)):
        sample = dataset[position]
        record = sample.record
        prediction = prediction_path(
            args.prediction_root.expanduser().resolve(),
            args.prediction_template,
            record.index,
            record.name,
        )
        if not prediction.is_file():
            missing.append({"index": record.index, "name": record.name, "prediction": str(prediction)})
            continue
        gt_path = args.output_dir / "gt_masks" / f"{record.index:08d}.png"
        atomic_save_mask(gt_path, sample.mask.numpy())
        row: dict[str, Any] = {
            "name": f"{record.name}_{record.index}",
            "method": args.method,
            "split": args.split,
            "instance_id": str(record.index),
            "image": str(record.image_path),
            "gt_mask": str(gt_path.resolve()),
            "prediction": str(prediction.resolve()),
            "prediction_kind": args.prediction_kind,
            "foreground_channel": args.foreground_channel,
            "threshold": args.threshold,
            "query": extract_target_text({}, record.user_text),
            "protocol": "paired_stamp_flat_json_external_prediction_import",
        }
        if args.array_key:
            row["array_key"] = args.array_key
        rows.append(row)
    return rows, missing


def main() -> int:
    args = parse_args()
    if min(args.limit, args.offset) < 0:
        raise ValueError("limit and offset must be non-negative.")
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("threshold must lie in (0, 1).")
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, missing = build_rows(args)
    if missing and not args.allow_missing:
        raise FileNotFoundError(
            f"{len(missing)} external predictions are missing; examples: "
            + json.dumps(missing[:5], ensure_ascii=True)
        )
    if not rows:
        raise ValueError("No complete image/GT/prediction rows were found.")
    manifest = args.output_dir / "manifest.jsonl"
    atomic_write_jsonl(manifest, rows)
    report = {
        "source": "external_predictions_aligned_to_stamp_flat_json",
        "method": args.method,
        "split": args.split,
        "samples": len(rows),
        "missing": len(missing),
        "prediction_kind": args.prediction_kind,
        "prediction_template": args.prediction_template,
        "manifest": str(manifest.resolve()),
    }
    (args.output_dir / "import_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
