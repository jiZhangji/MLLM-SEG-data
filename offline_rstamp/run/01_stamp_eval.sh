#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [[ -f "${ROOT_DIR}/offline_rstamp/paths.local.sh" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/offline_rstamp/paths.local.sh"
else
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/offline_rstamp/paths.example.sh"
fi

if [[ ! -d "${STAMP_CODE_DIR}" ]]; then
  echo "ERROR: STAMP_CODE_DIR not found: ${STAMP_CODE_DIR}" >&2
  exit 1
fi

cd "${STAMP_CODE_DIR}"

echo "Running STAMP eval wrapper."
echo "STAMP_CODE_DIR=${STAMP_CODE_DIR}"
echo "MLLM_SEG_DATA_ROOT=${MLLM_SEG_DATA_ROOT}"
echo "STAMP_MODEL_PATH=${STAMP_MODEL_PATH}"
echo "SAM_CKPT=${SAM_CKPT}"

if [[ -f "scripts/eval_ref.sh" ]]; then
  bash scripts/eval_ref.sh
else
  echo "ERROR: scripts/eval_ref.sh not found in STAMP. Please adapt this wrapper to your STAMP version." >&2
  exit 1
fi

