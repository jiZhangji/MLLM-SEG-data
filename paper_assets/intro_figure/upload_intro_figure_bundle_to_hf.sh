#!/usr/bin/env bash
set -euo pipefail

ROOT="${MLLM_SEG_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG}"
OUTPUT_DIR="${INTRO_OUTPUT_DIR:-${ROOT}/outputs/freeref_intro_motivation_bundle}"
STAMP_ENV="${STAMP_ENV_PATH:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/my_global_cache/conda/envs/STAMP}"
PYTHON_BIN="${INTRO_PYTHON:-${STAMP_ENV}/bin/python}"
HF_REPO_ID="${INTRO_HF_REPO_ID:-shimiandeshu/MLLM-SEG}"
HF_PATH="${INTRO_HF_PATH:-paper_assets/intro_figure/bundle_latest}"

for required in \
  "${OUTPUT_DIR}/intro_complete_figures.zip" \
  "${OUTPUT_DIR}/intro_individual_panels.zip" \
  "${OUTPUT_DIR}/freeref_intro_figure_bundle.zip"; do
  if [[ ! -f "${required}" ]]; then
    echo "ERROR: missing bundle output: ${required}" >&2
    exit 1
  fi
done

export HF_HOME="${HF_HOME:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/my_global_cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export INTRO_UPLOAD_DIR="${OUTPUT_DIR}"
export INTRO_HF_REPO_ID="${HF_REPO_ID}"
export INTRO_HF_PATH="${HF_PATH}"

"${PYTHON_BIN}" - <<'PY'
import os
from huggingface_hub import HfApi

api = HfApi()
print("HF account:", api.whoami()["name"])
result = api.upload_folder(
    repo_id=os.environ["INTRO_HF_REPO_ID"],
    repo_type="model",
    folder_path=os.environ["INTRO_UPLOAD_DIR"],
    path_in_repo=os.environ["INTRO_HF_PATH"],
    allow_patterns=["*.png", "*.pdf", "*.svg", "*.csv", "*.json", "*.zip"],
    commit_message="Upload alternate FreeRef introduction figures and panel bundles",
)
print("Upload completed:", result)
PY

echo "HF path: https://huggingface.co/${HF_REPO_ID}/tree/main/${HF_PATH}"
