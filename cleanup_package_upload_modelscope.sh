#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${MLLM_SEG_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG}"
REPO_ID="${MODELSCOPE_REPO_ID:-shimian123/FreeRef}"
REPO_TYPE="${MODELSCOPE_REPO_TYPE:-model}"
BACKUP_ID="${BACKUP_ID:-$(date -u +%Y%m%d_%H%M%S)}"
REMOTE_PATH="${MODELSCOPE_BACKUP_PATH:-backups/freeref_server_core_${BACKUP_ID}}"
BACKUP_DIR="$ROOT/modelscope_backup/freeref_server_core_${BACKUP_ID}"
COMPACT_STAGE="$ROOT/modelscope_stage/freeref_artifacts_compact_v2"
CONFIRM_DELETE="${CONFIRM_DELETE:-NO}"

die() {
    echo "ERROR: $*" >&2
    exit 1
}

[[ "$(realpath -m -- "$ROOT")" == "/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG" ]] ||
    die "Unexpected project root: $ROOT"
[[ -d "$ROOT/MLLM-SEG-data" ]] || die "Main code directory is missing"
[[ -f "$COMPACT_STAGE/.STAGING_COMPLETE" ]] ||
    die "Compact result staging is incomplete: $COMPACT_STAGE"
[[ "$CONFIRM_DELETE" == "YES" ]] ||
    die "Set CONFIRM_DELETE=YES after reviewing the deletion list"

PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
"$PYTHON_BIN" - <<'PY'
from modelscope_hub import HubApi
identity = HubApi().whoami()
print("Authenticated ModelScope account:", identity)
PY

mkdir -p "$BACKUP_DIR"
BEFORE_REPORT="$BACKUP_DIR/BEFORE_CLEANUP.txt"
AFTER_REPORT="$BACKUP_DIR/AFTER_CLEANUP.txt"
DELETION_REPORT="$BACKUP_DIR/DELETED_PATHS.txt"

{
    date -u
    hostname
    df -h "$ROOT"
    du -sh "$ROOT"/* 2>/dev/null | sort -h
} > "$BEFORE_REPORT"

DELETE_TARGETS=(
    "$ROOT/data"
    "$ROOT/outputs"
    "$ROOT/modelscope_stage/freeref_artifacts"
)

echo "The following disposable paths will be deleted:"
for target in "${DELETE_TARGETS[@]}"; do
    if [[ -e "$target" ]]; then
        du -sh "$target" 2>/dev/null || true
    else
        echo "missing: $target"
    fi
done

safe_remove() {
    local target="$1"
    local resolved
    resolved="$(realpath -m -- "$target")"
    [[ "$resolved" == "$ROOT/"* ]] || die "Refusing to delete outside project root: $resolved"
    case "$resolved" in
        "$ROOT/LH"|"$ROOT/models"|"$ROOT/MLLM-SEG-data"|"$COMPACT_STAGE")
            die "Refusing to delete protected path: $resolved"
            ;;
    esac
    if [[ -e "$target" ]]; then
        echo "$resolved" >> "$DELETION_REPORT"
        rm -rf --one-file-system -- "$target"
        echo "Deleted: $resolved"
    fi
}

: > "$DELETION_REPORT"
for target in "${DELETE_TARGETS[@]}"; do
    safe_remove "$target"
done

{
    date -u
    df -h "$ROOT"
    du -sh "$ROOT"/* 2>/dev/null | sort -h
} > "$AFTER_REPORT"

PACKAGE_ROOT_ITEMS=()
while IFS= read -r -d '' item; do
    name="$(basename "$item")"
    case "$name" in
        LH|models|data|outputs|modelscope_backup|.cache|.git)
            continue
            ;;
    esac
    PACKAGE_ROOT_ITEMS+=("$name")
done < <(find "$ROOT" -mindepth 1 -maxdepth 1 -print0)

[[ ${#PACKAGE_ROOT_ITEMS[@]} -gt 0 ]] || die "No files selected for packaging"

MANIFEST="$BACKUP_DIR/PACKAGE_MANIFEST.txt"
{
    echo "FreeRef server core backup"
    echo "Created UTC: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "Source root: $ROOT"
    echo "Excluded: LH models data outputs modelscope_backup .cache .git"
    echo "Included root entries:"
    printf '%s\n' "${PACKAGE_ROOT_ITEMS[@]}"
    echo
    git -C "$ROOT/MLLM-SEG-data" rev-parse HEAD 2>/dev/null || true
} > "$MANIFEST"

if command -v pigz >/dev/null 2>&1; then
    COMPRESSOR=(pigz -1)
else
    COMPRESSOR=(gzip -1)
fi

PART_PREFIX="$BACKUP_DIR/freeref_server_core_${BACKUP_ID}.tar.gz.part-"
echo "Packaging ${#PACKAGE_ROOT_ITEMS[@]} root entries into <=8 GiB parts..."
(
    cd "$ROOT"
    tar \
        --exclude='*/.git' \
        --exclude='*/.cache' \
        --exclude='*/__pycache__' \
        --exclude='*.pyc' \
        --exclude='*.pyo' \
        -cf - "${PACKAGE_ROOT_ITEMS[@]}"
) | "${COMPRESSOR[@]}" | split -b 8G -d -a 3 - "$PART_PREFIX"

(
    cd "$BACKUP_DIR"
    find . -maxdepth 1 -type f -name '*.part-*' -print0 |
        sort -z |
        xargs -0 sha256sum
) > "$BACKUP_DIR/SHA256SUMS"

echo "Package files:"
du -sh "$BACKUP_DIR"/* | sort -h

export REPO_ID REPO_TYPE REMOTE_PATH BACKUP_DIR
"$PYTHON_BIN" - <<'PY'
import os
import time
from pathlib import Path
from modelscope_hub import HubApi

repo_id = os.environ["REPO_ID"]
repo_type = os.environ["REPO_TYPE"]
remote_prefix = os.environ["REMOTE_PATH"].strip("/")
folder = Path(os.environ["BACKUP_DIR"]).resolve()
api = HubApi()

files = sorted(path for path in folder.rglob("*") if path.is_file())
if not files:
    raise SystemExit("Backup directory is empty")

print(f"Uploading {len(files)} backup files to {repo_id}:{remote_prefix}/")
api.upload_folder(
    repo_id,
    repo_type,
    folder,
    path_in_repo=remote_prefix,
    commit_message="Upload chunked FreeRef server core backup",
    max_workers=2,
    use_cache=True,
)

expected = {
    f"{remote_prefix}/{path.relative_to(folder).as_posix()}" for path in files
}
missing = sorted(expected)
for attempt in range(1, 11):
    time.sleep(20 if attempt == 1 else 30)
    remote_items = api.list_repo_files(repo_id, repo_type, recursive=True)
    remote_files = {
        item if isinstance(item, str) else item.path
        for item in remote_items
        if isinstance(item, str) or getattr(item, "type", "blob") != "tree"
    }
    missing = sorted(expected - remote_files)
    if not missing:
        print(f"Remote verification passed: {len(expected)} backup files are present.")
        break
    print(
        f"Remote verification attempt {attempt}/10: "
        f"{len(missing)} files are not indexed yet."
    )
else:
    raise SystemExit(
        f"Remote verification failed after retries: "
        f"{len(missing)} files missing: {missing[:5]}"
    )
PY

touch "$BACKUP_DIR/UPLOAD_COMPLETE"
echo "Cleanup, packaging, upload, and remote verification completed."
echo "Local backup: $BACKUP_DIR"
echo "Remote path: $REPO_ID:$REMOTE_PATH/"
