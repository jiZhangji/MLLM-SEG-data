from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine val/test SEG diagnostic summaries.")
    parser.add_argument("--val-summary", required=True, type=Path)
    parser.add_argument("--test-summary", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    args = parse_args()
    summaries = {
        "val": read_json(args.val_summary),
        "test": read_json(args.test_summary),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    combined = {
        "checkpoint": summaries["val"]["checkpoint"],
        "splits": summaries,
    }
    json_path = args.output_dir / "seg_diagnostics_val_test.json"
    json_path.write_text(json.dumps(combined, indent=2, ensure_ascii=True), encoding="utf-8")

    lines = [
        "# OnePass 7B SEG diagnostics: val and test",
        "",
        "| Split | Variant | gIoU | cIoU | Delta gIoU | Delta cIoU |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for split in ("val", "test"):
        for name, values in summaries[split]["variants"].items():
            lines.append(
                f"| {split} | {name} | {values['gIoU']:.6f} | {values['cIoU']:.6f} | "
                f"{values['mean_iou_delta_vs_baseline']:+.6f} | "
                f"{values['cIoU_delta_vs_baseline']:+.6f} |"
            )
    markdown_path = args.output_dir / "seg_diagnostics_val_test.md"
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(markdown_path.read_text(encoding="utf-8"), flush=True)
    print(f"JSON: {json_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
