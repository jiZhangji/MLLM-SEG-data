#!/usr/bin/env python3
"""Install the R-STAMP scaffold into a local code directory."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def copytree(src: Path, dst: Path, overwrite: bool) -> None:
    if dst.exists():
        if not overwrite:
            raise FileExistsError(f"{dst} already exists; pass --overwrite to replace scaffold files")
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-code-dir", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    src = root / "rstamp_src"
    target = args.target_code_dir.expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)

    for name in ["README.md", "configs", "rstamp"]:
        source = src / name
        dest = target / name
        if source.is_dir():
            copytree(source, dest, overwrite=args.overwrite)
        else:
            if dest.exists() and not args.overwrite:
                raise FileExistsError(f"{dest} already exists; pass --overwrite")
            shutil.copy2(source, dest)

    print(f"Installed R-STAMP scaffold to {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

