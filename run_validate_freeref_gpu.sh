#!/usr/bin/env bash
set -euo pipefail

ROOT="${MLLM_SEG_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAMP_ENV_PATH="${STAMP_ENV_PATH:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/my_global_cache/conda/envs/STAMP}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
LIMIT="${FREEREF_GPU_VALIDATION_LIMIT:-64}"
INPUT_DIR="${FREEREF_GPU_VALIDATION_INPUT:-${ROOT}/outputs/refine_stamp_dumps/refcoco_testA_full_stamp7b}"
OUTPUT_DIR="${FREEREF_GPU_VALIDATION_OUTPUT:-${ROOT}/outputs/freeref_gpu_validation_stamp7b_refcoco_testA_n${LIMIT}}"

if [[ ! -x "${STAMP_ENV_PATH}/bin/python" ]]; then
  echo "ERROR: STAMP environment is missing: ${STAMP_ENV_PATH}" >&2
  exit 1
fi
COUNT="$(find "${INPUT_DIR}" -maxdepth 1 -type f -name '*.pt' 2>/dev/null | wc -l || true)"
if (( COUNT < LIMIT )); then
  echo "ERROR: found ${COUNT} dumps in ${INPUT_DIR}; need at least ${LIMIT}." >&2
  exit 1
fi
mkdir -p "${OUTPUT_DIR}"
cd "${SCRIPT_DIR}"
CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}" \
  "${STAMP_ENV_PATH}/bin/python" -m efficiency_benchmark.validate_gpu_refiner \
    --input-dir "${INPUT_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --limit "${LIMIT}" \
    --seed 0

echo "CPU/GPU FreeRef validation complete: ${OUTPUT_DIR}/comparison.md"
