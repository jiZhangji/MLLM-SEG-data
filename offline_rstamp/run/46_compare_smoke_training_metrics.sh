#!/usr/bin/env bash
set -euo pipefail

MLLM_SEG_ROOT="${MLLM_SEG_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG}"
TOOL_REPO="${MLLM_SEG_ROOT}/MLLM-SEG-data"

python "${TOOL_REPO}/offline_rstamp/scripts/compare_smoke_training_metrics.py" \
  --root "${MLLM_SEG_ROOT}" \
  --output-dir "${MLLM_SEG_ROOT}/outputs"

