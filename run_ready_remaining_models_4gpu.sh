#!/usr/bin/env bash
set -euo pipefail

ROOT="${MLLM_SEG_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_ROOT="${READY_REMAINING_OUTPUT_ROOT:-${ROOT}/outputs/remaining_six_4gpu}"

echo "This legacy entry point now delegates to the complete ready-method phase."
echo "It runs READ, PolyFormer-L, the LISA diagnostic, and GSVA when its licensed base is ready."
exec env \
  MLLM_SEG_ROOT="${ROOT}" \
  REMAINING_SIX_PHASE=ready \
  REMAINING_SIX_RUN_ROOT="${OUTPUT_ROOT}" \
  bash "${SCRIPT_DIR}/run_remaining_six_experiments_4gpu.sh"
