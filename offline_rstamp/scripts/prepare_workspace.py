#!/usr/bin/env python3
"""Prepare an offline experiment workspace for STAMP/R-STAMP."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


def safe_symlink_or_copy(src: Path, dst: Path, copy: bool = False) -> None:
    if dst.exists() or dst.is_symlink():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy:
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
        return
    try:
        os.symlink(src, dst, target_is_directory=src.is_dir())
    except OSError:
        print(f"WARNING: symlink failed, copying instead: {src} -> {dst}")
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--copy", action="store_true", help="Copy data folders instead of creating symlinks.")
    args = parser.parse_args()

    data_root = args.data_root.expanduser().resolve()
    workspace = args.workspace.expanduser().resolve()

    for name in ["code", "models", "outputs", "logs", "tmp"]:
        (workspace / name).mkdir(parents=True, exist_ok=True)

    data_link_root = workspace / "data"
    data_link_root.mkdir(parents=True, exist_ok=True)
    for name in ["shared", "annotations", "datasets", "optional", "download_status.json"]:
        src = data_root / name
        if src.exists():
            safe_symlink_or_copy(src, data_link_root / name, copy=args.copy)
        else:
            print(f"WARNING: not found, skip: {src}")

    for name in ["STAMP", "R-STAMP", "Seg-Zero", "Text4Seg"]:
        (workspace / "code" / name).mkdir(parents=True, exist_ok=True)

    print(f"workspace: {workspace}")
    print(f"data view: {data_link_root}")
    print(f"code root: {workspace / 'code'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

