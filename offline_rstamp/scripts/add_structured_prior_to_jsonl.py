#!/usr/bin/env python3
"""Add simple structured-prior fields to a STAMP-like JSONL file.

This is a generic helper. If STAMP's generated annotation format differs,
adapt `rstamp/prior_targets.py` field names instead of editing training code
everywhere.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--rstamp-code-dir", required=True, type=Path)
    args = parser.parse_args()

    sys.path.insert(0, str(args.rstamp_code_dir.resolve()))
    from rstamp.prior_targets import enrich_jsonl

    enrich_jsonl(args.input_jsonl, args.output_jsonl)
    print(f"Wrote {args.output_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

