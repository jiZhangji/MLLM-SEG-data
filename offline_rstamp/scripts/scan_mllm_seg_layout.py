#!/usr/bin/env python3
"""Scan a remote MLLM-SEG workspace and report useful code/data locations.

This script is intentionally dependency-free. Run it on the GPU/server machine,
then send the generated report back to Codex so we can write a precise organizer
script for the actual layout.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable


DEFAULT_IGNORE_DIRS = {
    ".git",
    ".cache",
    ".conda",
    ".mamba",
    ".ipynb_checkpoints",
    "__pycache__",
    "node_modules",
    "wandb",
    "runs",
    "logs",
    "outputs",
    "checkpoints",
    ".pytest_cache",
}

CODE_PATTERNS = [
    "run_seg_ref.py",
    "create_refcoco_new.py",
    "create_refcoco.py",
    "create_grefercoco.py",
    "create_cocostuff.py",
    "eval_ref.sh",
    "refer_seg.sh",
    "semantic_seg.sh",
    "launch_all_7B.sh",
    "launch_all_2B.sh",
    "launch_all_7B_2x80g.sh",
    "finetune_lora.sh",
    "fintune_lora.sh",
    "train*.py",
    "eval*.py",
]

DATA_HINTS = [
    "train2014",
    "annotations",
    "refcoco",
    "refcoco+",
    "refcocog",
    "grefcoco",
    "grefer",
    "refclef",
    "referit",
    "reasonseg",
    "ReasonSeg",
    "cocostuff",
    "llava",
]

IMPORTANT_SUFFIXES = {
    ".py",
    ".sh",
    ".yaml",
    ".yml",
    ".json",
    ".jsonl",
    ".txt",
    ".md",
}


def should_ignore(path: Path, ignore_dirs: set[str]) -> bool:
    return any(part in ignore_dirs for part in path.parts)


def safe_iterdir(path: Path) -> Iterable[Path]:
    try:
        yield from path.iterdir()
    except Exception:
        return


def rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def file_info(path: Path, root: Path) -> dict:
    info = {
        "path": str(path),
        "relative": rel(path, root),
        "name": path.name,
        "suffix": path.suffix,
    }
    try:
        stat = path.stat()
        info["size"] = stat.st_size
    except Exception:
        info["size"] = None
    return info


def tree_summary(root: Path, max_depth: int, max_entries_per_dir: int, ignore_dirs: set[str]) -> list[str]:
    lines: list[str] = []
    root = root.resolve()

    def walk(path: Path, depth: int) -> None:
        if depth > max_depth:
            return
        if should_ignore(path, ignore_dirs):
            return
        indent = "  " * depth
        label = path.name + ("/" if path.is_dir() else "")
        if depth == 0:
            label = str(path)
        lines.append(f"{indent}{label}")
        if not path.is_dir():
            return
        entries = sorted(safe_iterdir(path), key=lambda p: (not p.is_dir(), p.name.lower()))
        visible = [p for p in entries if not should_ignore(p, ignore_dirs)]
        for child in visible[:max_entries_per_dir]:
            walk(child, depth + 1)
        if len(visible) > max_entries_per_dir:
            lines.append(f"{indent}  ... ({len(visible) - max_entries_per_dir} more)")

    walk(root, 0)
    return lines


def find_by_names(root: Path, names: list[str], ignore_dirs: set[str], max_results: int) -> dict[str, list[dict]]:
    out = {name: [] for name in names}
    exact_names = [name for name in names if "*" not in name]
    wildcard_names = [name for name in names if "*" in name]

    for dirpath, dirnames, filenames in os.walk(root):
        current = Path(dirpath)
        dirnames[:] = [d for d in dirnames if d not in ignore_dirs]
        if should_ignore(current, ignore_dirs):
            continue
        for filename in filenames:
            p = current / filename
            if filename in exact_names and len(out[filename]) < max_results:
                out[filename].append(file_info(p, root))
            for pattern in wildcard_names:
                if p.match(pattern) and len(out[pattern]) < max_results:
                    out[pattern].append(file_info(p, root))
    return out


def find_data_dirs(root: Path, ignore_dirs: set[str], max_results: int) -> list[dict]:
    results: list[dict] = []
    seen: set[str] = set()
    for dirpath, dirnames, _filenames in os.walk(root):
        current = Path(dirpath)
        dirnames[:] = [d for d in dirnames if d not in ignore_dirs]
        if should_ignore(current, ignore_dirs):
            continue
        low = current.name.lower()
        full_low = str(current).lower()
        if any(h.lower() in low or h.lower() in full_low for h in DATA_HINTS):
            key = str(current)
            if key not in seen:
                seen.add(key)
                info = {
                    "path": str(current),
                    "relative": rel(current, root),
                    "name": current.name,
                    "children": [],
                }
                children = sorted(safe_iterdir(current), key=lambda p: (not p.is_dir(), p.name.lower()))
                for child in children[:20]:
                    try:
                        size = child.stat().st_size if child.is_file() else None
                    except Exception:
                        size = None
                    info["children"].append({
                        "name": child.name,
                        "type": "dir" if child.is_dir() else "file",
                        "size": size,
                    })
                results.append(info)
                if len(results) >= max_results:
                    break
    return results


def count_files_by_suffix(root: Path, ignore_dirs: set[str], max_walk_files: int) -> dict:
    counts: dict[str, int] = {}
    total = 0
    for dirpath, dirnames, filenames in os.walk(root):
        current = Path(dirpath)
        dirnames[:] = [d for d in dirnames if d not in ignore_dirs]
        if should_ignore(current, ignore_dirs):
            continue
        for filename in filenames:
            total += 1
            suffix = Path(filename).suffix.lower() or "<no_suffix>"
            counts[suffix] = counts.get(suffix, 0) + 1
            if total >= max_walk_files:
                return {"total_seen": total, "truncated": True, "suffix_counts": counts}
    return {"total_seen": total, "truncated": False, "suffix_counts": counts}


def find_important_files(root: Path, ignore_dirs: set[str], max_results: int) -> list[dict]:
    results: list[dict] = []
    for dirpath, dirnames, filenames in os.walk(root):
        current = Path(dirpath)
        dirnames[:] = [d for d in dirnames if d not in ignore_dirs]
        if should_ignore(current, ignore_dirs):
            continue
        for filename in filenames:
            p = current / filename
            if p.suffix.lower() in IMPORTANT_SUFFIXES:
                results.append(file_info(p, root))
                if len(results) >= max_results:
                    return results
    return results


def render_markdown(report: dict) -> str:
    lines: list[str] = []
    lines.append("# MLLM-SEG Layout Scan Report")
    lines.append("")
    lines.append(f"- root: `{report['root']}`")
    lines.append(f"- generated_by: `{report['script']}`")
    lines.append("")

    lines.append("## Top-level tree")
    lines.append("")
    lines.append("```text")
    lines.extend(report["tree"])
    lines.append("```")
    lines.append("")

    lines.append("## Key code files")
    lines.append("")
    for name, items in report["code_files"].items():
        if not items:
            continue
        lines.append(f"### {name}")
        lines.append("")
        for item in items:
            lines.append(f"- `{item['path']}`")
        lines.append("")

    lines.append("## Data-like directories")
    lines.append("")
    for item in report["data_dirs"]:
        lines.append(f"### `{item['path']}`")
        lines.append("")
        for child in item["children"]:
            if child["type"] == "dir":
                lines.append(f"- `{child['name']}/`")
            else:
                lines.append(f"- `{child['name']}` ({child['size']} bytes)")
        lines.append("")

    lines.append("## File suffix counts")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(report["suffix_counts"], indent=2, ensure_ascii=False))
    lines.append("```")
    lines.append("")

    lines.append("## Important files sample")
    lines.append("")
    for item in report["important_files"]:
        lines.append(f"- `{item['path']}`")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path, help="Workspace root, e.g. /.../MLLM-SEG")
    parser.add_argument("--output", type=Path, default=None, help="Markdown report path")
    parser.add_argument("--json-output", type=Path, default=None, help="JSON report path")
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--max-entries-per-dir", type=int, default=80)
    parser.add_argument("--max-results", type=int, default=200)
    parser.add_argument("--max-important-files", type=int, default=500)
    parser.add_argument("--max-walk-files", type=int, default=200000)
    parser.add_argument("--include-cache", action="store_true")
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    ignore_dirs = set() if args.include_cache else set(DEFAULT_IGNORE_DIRS)

    report = {
        "script": "offline_rstamp/scripts/scan_mllm_seg_layout.py",
        "root": str(root),
        "tree": tree_summary(root, args.max_depth, args.max_entries_per_dir, ignore_dirs),
        "code_files": find_by_names(root, CODE_PATTERNS, ignore_dirs, args.max_results),
        "data_dirs": find_data_dirs(root, ignore_dirs, args.max_results),
        "suffix_counts": count_files_by_suffix(root, ignore_dirs, args.max_walk_files),
        "important_files": find_important_files(root, ignore_dirs, args.max_important_files),
    }

    markdown = render_markdown(report)
    print(markdown)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown + "\n", encoding="utf-8")
        print(f"\nMarkdown report written to: {args.output}")

    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"JSON report written to: {args.json_output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

