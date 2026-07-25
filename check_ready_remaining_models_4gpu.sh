#!/usr/bin/env bash
set -euo pipefail

ROOT="${MLLM_SEG_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "===== GPUs ====="
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv,noheader
echo "===== Active jobs ====="
pgrep -af '[r]un_ready_remaining_models_4gpu|[r]un_polyformer_freeref|[e]xport_polyformer_masks|[r]un_lisa_freeref_eval|[e]xport_lisa_masks' || echo none
echo "===== PolyFormer ====="
for spec in refcoco_val refcoco_testA refcoco_testB refcocoplus_val refcocoplus_testA refcocoplus_testB refcocog_val refcocog_test; do
  base="${ROOT}/outputs/polyformer_freeref_full/${spec}"
  printf '%-22s comparison=%-3s summary=%-3s\n' "${spec}" \
    "$([[ -f "${base}/comparison.md" ]] && echo yes || echo no)" \
    "$([[ -f "${base}/freeref/eval_summary.json" ]] && echo yes || echo no)"
done
echo "===== LISA ====="
LISA_RESULTS_ROOT="${ROOT}/outputs/lisa_official_all8" \
LISA_FREEREF_ROOT="${ROOT}/outputs/universal_freeref_lisa_all8" \
  bash "${SCRIPT_DIR}/check_lisa_freeref_status.sh"
echo "===== Log tails ====="
tail -n 20 "${ROOT}/outputs/freeref_ready_remaining_4gpu/polyformer.log" 2>/dev/null || true
tail -n 20 "${ROOT}/outputs/freeref_ready_remaining_4gpu/lisa.log" 2>/dev/null || true
