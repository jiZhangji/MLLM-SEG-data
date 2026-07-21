#!/usr/bin/env bash
set -euo pipefail

ROOT="${MLLM_SEG_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REFINE_ENV="${FREEREF_CONDA_ENV:-STAMP}"
PAPER_OUTPUT="${FREEREF_PAPER_OUTPUT:-${ROOT}/outputs/freeref_paper_suite}"

echo "===== Active paper jobs ====="
pgrep -af '[r]un_freeref_paper_suite|[r]un_pixellm_freeref|[r]un_segagent_freeref|[e]xport_pixellm_masks|run_segagent_official' || echo none
echo "===== GPUs ====="
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv,noheader
echo "===== PixelLM ====="
bash "${SCRIPT_DIR}/check_pixellm_freeref_status.sh"
echo "===== SegAgent ====="
bash "${SCRIPT_DIR}/check_segagent_freeref_status.sh"
echo "===== Paper table ====="
PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}" conda run --no-capture-output -n "${REFINE_ENV}" \
  python -m universal_freeref.summarize_paper_suite \
    --root "${ROOT}" \
    --output-dir "${PAPER_OUTPUT}" >/dev/null
cat "${PAPER_OUTPUT}/paper_results.md"
