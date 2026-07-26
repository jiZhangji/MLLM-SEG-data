from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi


UPLOAD_PATTERNS = [
    "freeref_framework_real.pdf",
    "freeref_framework_real.svg",
    "freeref_framework_real.png",
    "framework_candidate_contact_sheet.png",
    "framework_candidates.csv",
    "selected_real_sample.json",
    "freeref_framework_real_components/*.png",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Upload viewable real-framework outputs to Hugging Face."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-id", default="shimiandeshu/MLLM-SEG")
    parser.add_argument("--repo-type", choices=["model", "dataset", "space"], default="model")
    parser.add_argument("--path-in-repo", default="paper_assets/framework_runs/stamp")
    parser.add_argument(
        "--commit-message",
        default="Upload real FreeRef framework figure",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    if not output_dir.is_dir():
        raise FileNotFoundError(f"Framework output directory not found: {output_dir}")
    missing = [
        pattern
        for pattern in UPLOAD_PATTERNS[:6]
        if not (output_dir / pattern).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Framework generation is incomplete; missing: " + ", ".join(missing)
        )

    api = HfApi()
    commit = api.upload_folder(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        folder_path=str(output_dir),
        path_in_repo=args.path_in_repo.strip("/"),
        allow_patterns=UPLOAD_PATTERNS,
        commit_message=args.commit_message,
    )
    base_url = (
        f"https://huggingface.co/{args.repo_id}/resolve/main/"
        f"{args.path_in_repo.strip('/')}"
    )
    result = {
        "repo_id": args.repo_id,
        "repo_type": args.repo_type,
        "path_in_repo": args.path_in_repo.strip("/"),
        "commit_url": str(commit.commit_url),
        "preview_url": f"{base_url}/freeref_framework_real.png",
        "candidate_sheet_url": f"{base_url}/framework_candidate_contact_sheet.png",
        "pdf_url": f"{base_url}/freeref_framework_real.pdf",
        "uploaded_patterns": UPLOAD_PATTERNS,
        "excluded": ["selected_real_sample.npz"],
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
