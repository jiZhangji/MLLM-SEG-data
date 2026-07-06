#!/usr/bin/env python3
"""Check whether expected upstream repositories are present locally."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


REPOS = {
    "STAMP": {
        "remote_contains": ["HKUST-LongGroup/STAMP"],
        "required_files": [
            "requirements.txt",
            "run_seg_ref.py",
            "scripts/eval_ref.sh",
            "scripts/launch_all_7B.sh",
            "scripts/launch_all_2B.sh",
            "STAMP/data/create_refcoco_new.py",
        ],
    },
    "Seg-Zero": {
        "remote_contains": ["Seg-Zero", "JIA-Lab-research", "dvlab-research"],
        "required_files": [
            "training_scripts/run_visionreasoner_7b_4x80G.sh",
            "training_scripts/model_merger.py",
            "evaluation_scripts/eval_reasonseg_visionreasoner.sh",
        ],
        "optional": True,
    },
    "Text4Seg": {
        "remote_contains": ["mc-lan/Text4Seg"],
        "required_files": [
            "scripts/v1_5/fintune_lora.sh",
            "scripts/v1_5/eval/refer_seg.sh",
            "playground/data/create_json/create_refcoco.py",
        ],
        "optional": True,
    },
    "R-STAMP": {
        "remote_contains": [],
        "required_files": [
            "README.md",
            "configs/rstamp_2x80g.yaml",
            "rstamp/prior_schema.py",
            "rstamp/rewards.py",
            "rstamp/modules.py",
        ],
    },
}


def git_remote(path: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "remote", "-v"],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return ""
    return result.stdout.strip()


def inspect_repo(code_root: Path, name: str, spec: dict) -> dict:
    path = code_root / name
    remote = git_remote(path) if path.exists() else ""
    files = {}
    for rel in spec["required_files"]:
        files[rel] = (path / rel).exists()
    remote_ok = True
    if spec["remote_contains"]:
        remote_ok = any(token in remote for token in spec["remote_contains"])
    return {
        "path": str(path),
        "exists": path.exists(),
        "git_remote": remote,
        "remote_ok": remote_ok,
        "required_files": files,
        "missing_files": [rel for rel, ok in files.items() if not ok],
        "optional": bool(spec.get("optional", False)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-root", required=True, type=Path)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    code_root = args.code_root.expanduser().resolve()
    report = {"code_root": str(code_root), "repos": {}, "ok": True}
    for name, spec in REPOS.items():
        info = inspect_repo(code_root, name, spec)
        report["repos"][name] = info
        if not info["optional"] and (not info["exists"] or info["missing_files"]):
            report["ok"] = False

    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Report written to {args.report}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

