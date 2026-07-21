#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
METHODS="${METHODS:-pixellm segagent}"

echo "[1/2] Cloning official source repositories"
ROOT="${ROOT}" METHODS="${METHODS}" bash "${SCRIPT_DIR}/prepare_universal_freeref_repos.sh"

echo "[2/2] Downloading public checkpoints and SegAgent evaluation data"
ROOT="${ROOT}" METHODS="${METHODS}" DOWNLOAD_DATASETS=1 \
  bash "${SCRIPT_DIR}/download_missing_method_weights.sh"

echo "Paper assets are ready below ${ROOT}/code/third_party and ${ROOT}/models/freeref_missing_methods."
