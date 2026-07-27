#!/usr/bin/env bash
set -euo pipefail

ROOT="${MLLM_SEG_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG}"
REPO="${QUALITATIVE_REPO:-${ROOT}/MLLM-SEG-data}"
PYTHON_BIN="${QUALITATIVE_PYTHON:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/my_global_cache/conda/envs/STAMP/bin/python}"
FINAL="${MASK_EXPORT_OUTPUT_DIR:-${ROOT}/outputs/freeref_all_three_exact_binary_masks}"

BALANCED_SOURCE="${MASK_BALANCED_SOURCE:-${ROOT}/outputs/freeref_qualitative_candidates_n24}"
HARD_SOURCE="${MASK_HARD_SOURCE:-${ROOT}/outputs/freeref_hard_recovery_candidates_n36}"
ZOOM_SOURCE="${MASK_ZOOM_SOURCE:-${ROOT}/outputs/freeref_hard_recovery_binary_zoom_n36}"

HF_REPO_ID="${MASK_HF_REPO_ID:-shimiandeshu/MLLM-SEG}"
HF_PATH="${MASK_HF_PATH:-paper_assets/qualitative/all_three_exact_binary_masks}"

read_ids() {
  "${PYTHON_BIN}" -c \
    'import csv,sys; print(" ".join(row["sample_id"] for row in csv.DictReader(open(sys.argv[1], newline="", encoding="utf-8"))))' \
    "$1"
}

check_source() {
  local source="$1"
  for name in main_table_qualitative_rows.csv postprocess_qualitative_rows.csv; do
    if [[ ! -f "${source}/${name}" ]]; then
      echo "ERROR: missing selection CSV: ${source}/${name}" >&2
      exit 1
    fi
  done
}

run_export() {
  local label="$1"
  local source="$2"
  local selection_mode="$3"
  local output="${FINAL}/${label}"
  local main_ids post_ids count
  main_ids="$(read_ids "${source}/main_table_qualitative_rows.csv")"
  post_ids="$(read_ids "${source}/postprocess_qualitative_rows.csv")"
  count="$(wc -w <<<"${main_ids}")"
  echo "===== Export ${label}: ${count} main samples ====="
  env \
    MLLM_SEG_ROOT="${ROOT}" \
    QUALITATIVE_OUTPUT_DIR="${output}" \
    QUALITATIVE_RENDER_STYLE="masks_only" \
    QUALITATIVE_MAIN_SELECTION_MODE="${selection_mode}" \
    QUALITATIVE_POST_SELECTION_MODE="${selection_mode}" \
    QUALITATIVE_MAIN_SAMPLE_IDS="${main_ids}" \
    QUALITATIVE_POST_SAMPLE_IDS="${post_ids}" \
    QUALITATIVE_SAMPLE_COUNT="${count}" \
    QUALITATIVE_CANDIDATE_POOL="1" \
    QUALITATIVE_SKIP_ARCHIVE="1" \
    bash "${REPO}/paper_assets/qualitative_comparison/run_qualitative_figures.sh"
}

check_source "${BALANCED_SOURCE}"
check_source "${HARD_SOURCE}"
check_source "${ZOOM_SOURCE}"

hard_main="$(read_ids "${HARD_SOURCE}/main_table_qualitative_rows.csv")"
zoom_main="$(read_ids "${ZOOM_SOURCE}/main_table_qualitative_rows.csv")"
hard_post="$(read_ids "${HARD_SOURCE}/postprocess_qualitative_rows.csv")"
zoom_post="$(read_ids "${ZOOM_SOURCE}/postprocess_qualitative_rows.csv")"
if [[ "${hard_main}" != "${zoom_main}" || "${hard_post}" != "${zoom_post}" ]]; then
  echo "ERROR: hard-recovery and binary-zoom selections do not match." >&2
  exit 1
fi

mkdir -p "${FINAL}"
cd "${REPO}"
run_export "balanced_n24" "${BALANCED_SOURCE}" "balanced"
run_export "hard_recovery_n36" "${HARD_SOURCE}" "hard_recovery"
run_export "binary_zoom_n36" "${ZOOM_SOURCE}" "hard_recovery"

export MASK_EXPORT_FINAL="${FINAL}"
archive="$(${PYTHON_BIN} - <<'PY'
import os
import zipfile
from pathlib import Path

root = Path(os.environ["MASK_EXPORT_FINAL"])
archive = root / "all_three_exact_binary_masks.zip"
with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as handle:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path != archive:
            handle.write(path, path.relative_to(root).as_posix())
print(archive)
PY
)"

"${PYTHON_BIN}" - "${archive}" "${FINAL}" "${HF_REPO_ID}" "${HF_PATH}" <<'PY'
import sys
from pathlib import Path
from huggingface_hub import HfApi

archive = Path(sys.argv[1])
root = Path(sys.argv[2])
repo_id = sys.argv[3]
target = sys.argv[4]
api = HfApi()

print("Uploading complete mask archive...")
archive_commit = api.upload_file(
    repo_id=repo_id,
    repo_type="model",
    path_or_fileobj=archive,
    path_in_repo=f"{target}/{archive.name}",
    commit_message="Upload exact binary masks for three qualitative selections",
)
print("Archive commit:", archive_commit.commit_url)

print("Uploading individual masks and metric indexes...")
folder_commit = api.upload_folder(
    repo_id=repo_id,
    repo_type="model",
    folder_path=root,
    path_in_repo=target,
    ignore_patterns=[archive.name, "*_panels/**"],
    commit_message="Upload browsable exact qualitative binary masks",
)
print("Folder commit:", folder_commit.commit_url)
print(f"HF directory: https://huggingface.co/{repo_id}/tree/main/{target}")
PY

echo "Archive: ${archive}"
echo "Complete: yes"
