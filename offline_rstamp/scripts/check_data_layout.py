#!/usr/bin/env python3
"""Check the downloaded MLLM-SEG dataset layout without internet access."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_PATHS = {
    "coco_train2014": "shared/coco/train2014",
    "coco_annotations": "shared/coco/annotations",
    "refcoco_family": "annotations/refcoco_family",
}

RECOMMENDED_PATHS = {
    "grefcoco": "annotations/grefcoco",
    "refclef_referit": "datasets/refclef_referit",
    "reasonseg": "datasets/reasonseg",
    "reasonseg_test_segzero": "datasets/reasonseg_test_segzero",
    "llava_665k": "annotations/llava_665k",
    "open_vocab_cocostuff164k": "optional/open_vocabulary/cocostuff164k",
}


def count_files(path: Path, suffixes: tuple[str, ...] | None = None, limit: int = 1_000_000) -> int:
    if not path.exists():
        return 0
    n = 0
    for item in path.rglob("*"):
        if item.is_file() and (suffixes is None or item.suffix.lower() in suffixes):
            n += 1
            if n >= limit:
                break
    return n


def inspect_path(root: Path, rel: str) -> dict:
    path = root / rel
    info = {
        "relative_path": rel,
        "absolute_path": str(path),
        "exists": path.exists(),
        "is_dir": path.is_dir(),
    }
    if path.exists() and path.is_dir():
        info["num_json"] = count_files(path, (".json", ".jsonl"))
        info["num_images"] = count_files(path, (".jpg", ".jpeg", ".png", ".bmp", ".webp"))
        info["num_masks"] = count_files(path, (".png", ".npy", ".npz"))
    return info


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    root = args.data_root.expanduser().resolve()
    report = {
        "data_root": str(root),
        "required": {},
        "recommended": {},
        "ok": True,
        "missing_required": [],
        "missing_recommended": [],
    }

    for name, rel in REQUIRED_PATHS.items():
        info = inspect_path(root, rel)
        report["required"][name] = info
        if not info["exists"]:
            report["ok"] = False
            report["missing_required"].append(name)

    for name, rel in RECOMMENDED_PATHS.items():
        info = inspect_path(root, rel)
        report["recommended"][name] = info
        if not info["exists"]:
            report["missing_recommended"].append(name)

    print(json.dumps(report, indent=2, ensure_ascii=False))

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Report written to {args.report}")

    if report["missing_required"]:
        print("ERROR: missing required dataset parts: " + ", ".join(report["missing_required"]))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

