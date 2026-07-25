#!/usr/bin/env bash
set -euo pipefail

ROOT="${MLLM_SEG_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "===== GPUs ====="
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv,noheader
echo "===== Active remaining-six processes ====="
pgrep -af '[r]un_remaining_six|[r]un_read_freeref|[r]un_gsva_freeref|[r]un_rela|[r]un_polyformer|[r]un_lisa_freeref' || echo none
echo "===== READ ====="
MLLM_SEG_ROOT="${ROOT}" bash "${SCRIPT_DIR}/check_read_freeref_status.sh"
echo "===== GSVA ====="
MLLM_SEG_ROOT="${ROOT}" bash "${SCRIPT_DIR}/check_gsva_freeref_status.sh"
echo "===== ReLA ====="
MLLM_SEG_ROOT="${ROOT}" bash "${SCRIPT_DIR}/check_rela_freeref_status.sh"
echo "===== PolyFormer/LISA ====="
MLLM_SEG_ROOT="${ROOT}" bash "${SCRIPT_DIR}/check_ready_remaining_models_4gpu.sh"
echo "===== Table 1(b) row artifacts ====="
find "${ROOT}/outputs" -path '*/combined/table1b_row.tsv' -type f -print 2>/dev/null | sort
echo "===== Orchestrator log tails ====="
for log in "${ROOT}/outputs/remaining_six_4gpu"/*.log; do
  [[ -f "${log}" ]] || continue
  echo "--- ${log}"
  tail -n 8 "${log}" || true
done
