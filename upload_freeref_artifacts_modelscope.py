#!/usr/bin/env python3
"""Stage and upload the reproducible FreeRef artifacts to ModelScope.

The raw output tree contains more than 100,000 files. ModelScope counts those
files before applying CLI include filters, so this tool first creates a small,
clean staging tree and uploads only that tree.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path


DEFAULT_ROOT = Path(
    "/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG"
)
DEFAULT_REPO_ID = "shimian123/FreeRef"
MAX_MODELSCOPE_FILES = 100_000
MAX_MODELSCOPE_FILE_BYTES = 100 * 1024**3
MAX_COMPACT_FILES = 90_000
MAX_FILES_PER_OUTPUT = 500
MAX_FILES_FOR_ASSET_OUTPUT = 5_000

OUTPUT_PATTERNS = (
    "freeref_final_h100_overnight_v2",
    "freeref_intervention_concentration_*",
    "freeref_efficiency_*",
    "freeref_paper_studies_n500",
    "freeref_postprocess_baselines_n500",
    "training_free_refine_stamp7b_*",
    "training_free_refine_stamp2b_*",
    "text4seg_training_free_*",
    "pixellm_public_freeref",
    "pixellm_public_freeref_full",
    "polyformer_freeref_full",
    "gsva_freeref_full",
    "universal_freeref_lisa*",
    "framework_figure_real",
    "freeref_intro_final_staff_shirt",
    "freeref_all_three_exact_binary_masks",
)

ALLOWED_SUFFIXES = {
    ".json",
    ".jsonl",
    ".csv",
    ".tsv",
    ".md",
    ".txt",
    ".yaml",
    ".yml",
    ".png",
    ".jpg",
    ".jpeg",
    ".pdf",
    ".svg",
}

SKIPPED_PARTS = {
    ".cache",
    ".git",
    ".pytest_cache",
    ".tmp",
    "__pycache__",
    "checkpoints",
    "dumps",
    "logs",
    "masks",
    "panels",
    "predictions",
    "probabilities",
    "sample_panels",
    "worker_logs",
    "workers",
}

MASK_ARCHIVE_DIR = "freeref_all_three_exact_binary_masks"
ASSET_OUTPUTS = {
    MASK_ARCHIVE_DIR,
    "framework_figure_real",
    "freeref_intro_final_staff_shirt",
}
SKIPPED_DIR_MARKERS = (
    "cache",
    "checkpoint",
    "dump",
    "mask",
    "panel",
    "prediction",
    "probabilit",
    "worker",
)
SUMMARY_TOKENS = (
    "complete",
    "config",
    "concentration",
    "efficiency",
    "figure",
    "index",
    "manifest",
    "metric",
    "plot",
    "readme",
    "report",
    "result",
    "status",
    "summary",
    "table",
)
DIRECT_SUMMARY_SUFFIXES = {".csv", ".md", ".pdf", ".svg", ".tsv", ".yaml", ".yml"}
CONDITIONAL_SUFFIXES = {".json", ".jsonl", ".jpeg", ".jpg", ".png", ".txt"}


@dataclass(frozen=True)
class StagedFile:
    source: Path
    destination: Path
    size: int
    sha256: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage, upload, and verify the essential FreeRef artifacts."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(os.environ.get("MLLM_SEG_ROOT", DEFAULT_ROOT)),
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--repo-type", choices=("model", "dataset"), default="model")
    parser.add_argument("--path-in-repo", default="artifacts")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument(
        "--max-file-mb",
        type=int,
        default=200,
        help="Skip individual artifacts larger than this value (default: 200 MB).",
    )
    parser.add_argument(
        "--stage-dir",
        type=Path,
        help="Defaults to ROOT/modelscope_stage/freeref_artifacts_compact_v2.",
    )
    parser.add_argument(
        "--reuse-stage",
        action="store_true",
        help="Reuse an existing completed staging tree after an interrupted upload.",
    )
    parser.add_argument(
        "--stage-only",
        action="store_true",
        help="Build and validate the staging tree without uploading.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_output_dirs(outputs: Path) -> list[Path]:
    selected = []
    if not outputs.is_dir():
        raise FileNotFoundError(f"Outputs directory does not exist: {outputs}")
    for child in outputs.iterdir():
        if not child.is_dir():
            continue
        if any(fnmatch.fnmatchcase(child.name, pattern) for pattern in OUTPUT_PATTERNS):
            selected.append(child)
    return sorted(selected, key=lambda path: path.name)


def should_stage(source_root: Path, path: Path, max_bytes: int) -> tuple[bool, str]:
    relative = path.relative_to(source_root)
    parts = set(relative.parts[:-1])
    skipped_parts = SKIPPED_PARTS
    if source_root.name == MASK_ARCHIVE_DIR:
        skipped_parts = SKIPPED_PARTS - {"masks"}
    if parts & skipped_parts:
        return False, "skipped-directory"
    if path.suffix.lower() not in ALLOWED_SUFFIXES:
        return False, "unsupported-extension"
    suffix = path.suffix.lower()
    if source_root.name not in ASSET_OUTPUTS:
        name = path.name.lower()
        if suffix not in DIRECT_SUMMARY_SUFFIXES and not (
            suffix in CONDITIONAL_SUFFIXES
            and (
                any(token in name for token in SUMMARY_TOKENS)
                or (suffix in {".jpeg", ".jpg", ".png"} and len(relative.parts) <= 2)
            )
        ):
            return False, "non-summary-artifact"
    size = path.stat().st_size
    if size > max_bytes:
        return False, "over-size-limit"
    return True, "selected"


def safe_reset_stage(root: Path, stage: Path) -> None:
    root = root.resolve()
    stage = stage.resolve()
    allowed_parent = (root / "modelscope_stage").resolve()
    if stage == allowed_parent or allowed_parent not in stage.parents:
        raise ValueError(
            f"Refusing to reset stage outside {allowed_parent}: {stage}"
        )
    if stage.exists():
        print(f"Removing previous staging tree: {stage}", flush=True)
        shutil.rmtree(stage)
    stage.mkdir(parents=True, exist_ok=True)


def write_summary(
    stage: Path,
    selected_dirs: list[Path],
    staged: list[StagedFile],
    skipped_counts: dict[str, int],
) -> None:
    total_bytes = sum(item.size for item in staged)
    summary = [
        "# FreeRef ModelScope Artifact Upload",
        "",
        "This staging tree contains compact experiment summaries and paper assets.",
        "Raw checkpoints, logs, dumps, predictions, datasets, and third-party model",
        "weights are intentionally excluded.",
        "",
        f"- Selected output directories: {len(selected_dirs)}",
        f"- Staged files: {len(staged)}",
        f"- Staged bytes: {total_bytes}",
        "",
        "## Included output directories",
        "",
    ]
    summary.extend(f"- `{path.name}`" for path in selected_dirs)
    summary.extend(["", "## Skipped files", ""])
    summary.extend(f"- {key}: {value}" for key, value in sorted(skipped_counts.items()))
    (stage / "UPLOAD_SUMMARY.md").write_text("\n".join(summary) + "\n", encoding="utf-8")

    manifest_path = stage / "MANIFEST.tsv"
    with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("path\tsize_bytes\tsha256\tsource\n")
        for item in sorted(staged, key=lambda value: value.destination.as_posix()):
            relative = item.destination.relative_to(stage).as_posix()
            handle.write(
                f"{relative}\t{item.size}\t{item.sha256}\t{item.source.as_posix()}\n"
            )


def build_stage(root: Path, stage: Path, max_bytes: int) -> list[Path]:
    outputs = root / "outputs"
    selected_dirs = selected_output_dirs(outputs)
    if not selected_dirs:
        raise RuntimeError("No configured FreeRef output directories were found")

    safe_reset_stage(root, stage)
    staged: list[StagedFile] = []
    skipped_counts: dict[str, int] = {}
    for source_root in selected_dirs:
        before = len(staged)
        print(f"Scanning: {source_root.name}", flush=True)
        candidates: list[Path] = []
        inspected = 0
        for current, dirnames, filenames in os.walk(source_root):
            if source_root.name not in ASSET_OUTPUTS:
                dirnames[:] = [
                    name
                    for name in dirnames
                    if name not in SKIPPED_PARTS
                    and not any(marker in name.lower() for marker in SKIPPED_DIR_MARKERS)
                    and name.lower() != "samples"
                    and not name.lower().startswith("sample_")
                ]
            current_path = Path(current)
            for filename in filenames:
                inspected += 1
                if inspected % 50_000 == 0:
                    print(
                        f"  inspected {inspected:,} files; "
                        f"selected {len(candidates):,}",
                        flush=True,
                    )
                source = current_path / filename
                keep, reason = should_stage(source_root, source, max_bytes)
                if not keep:
                    skipped_counts[reason] = skipped_counts.get(reason, 0) + 1
                    continue
                candidates.append(source)

        candidates.sort(
            key=lambda path: (
                0
                if any(token in path.name.lower() for token in SUMMARY_TOKENS)
                else 1,
                len(path.relative_to(source_root).parts),
                path.relative_to(source_root).as_posix(),
            )
        )
        source_limit = (
            MAX_FILES_FOR_ASSET_OUTPUT
            if source_root.name in ASSET_OUTPUTS
            else MAX_FILES_PER_OUTPUT
        )
        source_limit = min(source_limit, max(0, MAX_COMPACT_FILES - len(staged)))
        if len(candidates) > source_limit:
            skipped_counts["per-output-file-cap"] = (
                skipped_counts.get("per-output-file-cap", 0)
                + len(candidates)
                - source_limit
            )
            candidates = candidates[:source_limit]

        for source in candidates:
                relative = source.relative_to(source_root)
                destination = stage / "results" / source_root.name / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                staged.append(
                    StagedFile(
                        source=source,
                        destination=destination,
                        size=destination.stat().st_size,
                        sha256=sha256_file(destination),
                    )
                )
        print(f"  staged {len(staged) - before:,} files", flush=True)

    if not staged:
        raise RuntimeError("Selection produced an empty staging tree")
    write_summary(stage, selected_dirs, staged, skipped_counts)
    (stage / ".STAGING_COMPLETE").write_text("complete\n", encoding="ascii")
    return [item.destination for item in staged]


def validate_stage(stage: Path) -> tuple[list[Path], int]:
    marker = stage / ".STAGING_COMPLETE"
    if not marker.is_file():
        raise RuntimeError(f"Staging marker is missing: {marker}")
    files = sorted(path for path in stage.rglob("*") if path.is_file())
    total_bytes = sum(path.stat().st_size for path in files)
    if len(files) >= MAX_MODELSCOPE_FILES:
        raise RuntimeError(
            f"Staged file count {len(files)} exceeds ModelScope limit "
            f"{MAX_MODELSCOPE_FILES}"
        )
    oversized = [path for path in files if path.stat().st_size > MAX_MODELSCOPE_FILE_BYTES]
    if oversized:
        raise RuntimeError(f"Files exceed ModelScope 100 GB limit: {oversized[:3]}")
    return files, total_bytes


def remote_path(prefix: str, relative: Path) -> str:
    prefix = prefix.strip("/")
    value = relative.as_posix()
    return f"{prefix}/{value}" if prefix else value


def list_remote_file_paths(api, args: argparse.Namespace) -> set[str]:
    remote_items = api.list_repo_files(args.repo_id, args.repo_type, recursive=True)
    return {
        item if isinstance(item, str) else item.path
        for item in remote_items
        if isinstance(item, str) or getattr(item, "type", "blob") != "tree"
    }


def retry_missing_files(
    api,
    args: argparse.Namespace,
    stage: Path,
    expected_to_local: dict[str, Path],
    missing: list[str],
    attempt: int,
) -> None:
    retry_stage = stage.parent / f".{stage.name}_missing_retry"
    if retry_stage.exists():
        shutil.rmtree(retry_stage)
    retry_stage.mkdir(parents=True)
    try:
        for remote_name in missing:
            source = expected_to_local[remote_name]
            relative = source.relative_to(stage)
            destination = retry_stage / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        print(
            f"Missing-file retry {attempt}: uploading {len(missing)} files "
            "with cache disabled.",
            flush=True,
        )
        api.upload_folder(
            args.repo_id,
            args.repo_type,
            retry_stage,
            path_in_repo=args.path_in_repo.strip("/"),
            commit_message=f"Retry missing FreeRef artifacts (attempt {attempt})",
            max_workers=min(args.max_workers, 4),
            use_cache=False,
        )
    finally:
        if retry_stage.exists():
            shutil.rmtree(retry_stage)


def upload_and_verify(args: argparse.Namespace, stage: Path, files: list[Path]) -> None:
    try:
        from modelscope_hub import HubApi
    except ImportError as error:
        raise RuntimeError(
            "modelscope-hub is unavailable; run: python -m pip install -U modelscope-hub"
        ) from error

    api = HubApi()
    identity = api.whoami()
    username = getattr(identity, "username", str(identity))
    print(f"Authenticated ModelScope user: {username}", flush=True)
    if not api.repo_exists(args.repo_id, args.repo_type):
        raise RuntimeError(f"ModelScope repository does not exist: {args.repo_id}")

    print(f"Uploading {len(files)} staged files to {args.repo_id}:{args.path_in_repo}/")
    api.upload_folder(
        args.repo_id,
        args.repo_type,
        stage,
        path_in_repo=args.path_in_repo.strip("/"),
        commit_message="Upload compact FreeRef experiment artifacts",
        max_workers=args.max_workers,
        use_cache=True,
    )

    expected_to_local = {
        remote_path(args.path_in_repo, path.relative_to(stage)): path for path in files
    }
    expected = set(expected_to_local)
    missing: list[str] = []
    for attempt in range(1, 5):
        if attempt == 1:
            print("Waiting for the remote file index to refresh...", flush=True)
            time.sleep(15)
        remote_files = list_remote_file_paths(api, args)
        missing = sorted(expected - remote_files)
        if not missing:
            print(
                f"Remote verification passed: {len(expected)} files are present.",
                flush=True,
            )
            return
        print(
            f"Remote verification attempt {attempt}: {len(missing)} files missing; "
            f"first entries: {missing[:5]}",
            flush=True,
        )
        if attempt < 4:
            retry_missing_files(
                api, args, stage, expected_to_local, missing, attempt
            )
            time.sleep(10)

    raise RuntimeError(
        f"Remote verification still reports {len(missing)} missing files after "
        f"three retries; first entries: {missing[:5]}"
    )


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    if args.max_workers <= 0 or args.max_file_mb <= 0:
        raise ValueError("max-workers and max-file-mb must be positive")
    stage = (
        args.stage_dir.expanduser().resolve()
        if args.stage_dir
        else (root / "modelscope_stage" / "freeref_artifacts_compact_v2").resolve()
    )

    if args.reuse_stage:
        print(f"Reusing staging tree: {stage}", flush=True)
    else:
        print(f"Building staging tree: {stage}", flush=True)
        build_stage(root, stage, args.max_file_mb * 1024**2)

    files, total_bytes = validate_stage(stage)
    print(f"Staged files: {len(files):,}", flush=True)
    print(f"Staged size: {total_bytes / 1024**2:.2f} MiB", flush=True)
    print(f"Manifest: {stage / 'MANIFEST.tsv'}", flush=True)
    if args.stage_only:
        print("Stage-only mode complete; upload was not started.", flush=True)
        return 0

    upload_and_verify(args, stage, files)
    print("FreeRef ModelScope artifact upload completed successfully.", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        raise
