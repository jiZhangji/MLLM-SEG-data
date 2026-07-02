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

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

echo "Running STAMP 2×80GB training wrapper."
echo "If your official STAMP script assumes 4/8 GPUs, edit scripts/launch_all_7B.sh or create a 2-GPU copy."

if [[ -f "scripts/launch_all_7B.sh" ]]; then
  bash scripts/launch_all_7B.sh
elif [[ -f "scripts/launch_all_2B.sh" ]]; then
  bash scripts/launch_all_2B.sh
else
  echo "ERROR: no STAMP launch script found. Expected scripts/launch_all_7B.sh or scripts/launch_all_2B.sh." >&2
  exit 1
fi

