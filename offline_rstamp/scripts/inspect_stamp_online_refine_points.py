#!/usr/bin/env python3
"""Inspect a server STAMP checkout for online token-refinement integration.

The local MLLM-SEG-data repo does not include the official `code/STAMP` source.
Run this script on the server to collect the exact files/snippets needed to
patch online scheme B into the official batched training path.
"""

from __future__ import annotations

import argparse
import inspect
import json
import re
import sys
from pathlib import Path


PATTERNS = [
    "bi_logits",
    "do_classification",
    "seg_mask",
    "mask_token",
    "mask_token_id",
    "image_grid_thw",
    "Trainer",
    "compute_loss",
    "forward(",
]


def safe_read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="replace")


def line_matches(text: str, patterns: list[str]) -> list[dict[str, object]]:
    rows = []
    lines = text.splitlines()
    for idx, line in enumerate(lines, start=1):
        if any(pattern in line for pattern in patterns):
            start = max(1, idx - 8)
            end = min(len(lines), idx + 12)
            rows.append(
                {
                    "line": idx,
                    "text": line.strip(),
                    "context_start": start,
                    "context_end": end,
                    "context": "\n".join(f"{n}: {lines[n - 1]}" for n in range(start, end + 1)),
                }
            )
    return rows


def find_candidate_files(stamp_code_dir: Path) -> list[Path]:
    names = {
        "modeling_qwen2_vl.py",
        "main_uni.py",
        "segment_predictor.py",
        "segment_predictor_cache.py",
    }
    candidates = []
    for path in stamp_code_dir.rglob("*.py"):
        if path.name in names or any(part in {"train", "model"} for part in path.parts):
            try:
                text = safe_read(path)
            except Exception:
                continue
            if any(pattern in text for pattern in PATTERNS):
                candidates.append(path)
    return sorted(candidates)


def write_context_files(output_dir: Path, candidates: list[Path], stamp_code_dir: Path) -> list[dict[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for path in candidates:
        rel = path.relative_to(stamp_code_dir)
        text = safe_read(path)
        matches = line_matches(text, PATTERNS)
        safe_name = "__".join(rel.parts).replace(".py", "") + ".snippets.txt"
        snippet_path = output_dir / safe_name
        snippet_path.write_text(
            "\n\n".join(match["context"] for match in matches[:80]),
            encoding="utf-8",
        )
        summary.append(
            {
                "path": str(path),
                "relative_path": str(rel),
                "num_matches": len(matches),
                "snippet_file": str(snippet_path),
                "first_matches": matches[:20],
            }
        )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stamp-code-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    stamp_code_dir = args.stamp_code_dir.expanduser().resolve()
    if not stamp_code_dir.exists():
        raise FileNotFoundError(stamp_code_dir)

    candidates = find_candidate_files(stamp_code_dir)
    summary = write_context_files(args.output_dir, candidates, stamp_code_dir)
    report = {
        "stamp_code_dir": str(stamp_code_dir),
        "num_candidate_files": len(candidates),
        "patterns": PATTERNS,
        "files": summary,
    }
    report_path = args.output_dir / "online_refine_inspection.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote inspection report: {report_path}")
    for item in summary[:20]:
        print(f"- {item['relative_path']} -> {item['snippet_file']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
