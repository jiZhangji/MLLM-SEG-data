#!/usr/bin/env bash
set -euo pipefail

ROOT="${MLLM_SEG_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG}"
DATA_ROOT="${DATA_ROOT:-${ROOT}/MLLM-SEG-data}"
TRAIN_DUMPS="${TRAIN_DUMPS:-${ROOT}/outputs/refine_stamp_dumps/refcocog_train_0}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/outputs}"
CACHE_FILE="${CACHE_FILE:-${OUT_ROOT}/token_refine_cache/refcocog_train_0_token_cache.pt}"
GPU="${GPU:-0}"

cd "${DATA_ROOT}"

CUDA_VISIBLE_DEVICES="${GPU}" python -m token_refine.train_adapter_from_dumps \
  --input-dir "${TRAIN_DUMPS}" \
  --output-dir "${OUT_ROOT}/token_refine_cache_build_tmp" \
  --cache-file "${CACHE_FILE}" \
  --build-token-cache-only \
  --batch-size 1 \
  --num-workers 0 \
  --device cuda

echo "Token cache ready: ${CACHE_FILE}"
