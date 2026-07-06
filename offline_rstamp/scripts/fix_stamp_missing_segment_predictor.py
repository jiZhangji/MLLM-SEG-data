#!/usr/bin/env python3
"""Fix STAMP import error: `No module named segment_predictor`.

Some STAMP checkouts contain `segment_predictor_cache.py` while
`train/seg_trainer.py` imports:

    from segment_predictor import find_image_patch_info

This script creates `segment_predictor.py` as a compatibility copy/shim.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stamp-code-dir", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = args.stamp_code_dir.expanduser().resolve()
    target = root / "segment_predictor.py"
    source = root / "segment_predictor_cache.py"

    if target.exists() and not args.overwrite:
        print(f"[OK] already exists: {target}")
        return 0

    if source.exists():
        shutil.copy2(source, target)
        print(f"[OK] copied {source.name} -> {target.name}")
        return 0

    target.write_text(
        '"""Compatibility shim for STAMP local training."""\n'
        "try:\n"
        "    from segment_predictor_cache import *  # noqa: F401,F403\n"
        "except Exception as exc:\n"
        "    raise ImportError('segment_predictor_cache.py is required by STAMP training') from exc\n",
        encoding="utf-8",
    )
    print(f"[WARN] source missing; wrote shim: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

