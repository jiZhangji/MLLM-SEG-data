#!/usr/bin/env bash
set -u

ROOT="${MLLM_SEG_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "===== GPUs ====="
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu \
  --format=csv,noheader
echo
echo "===== GPU processes ====="
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader || true

run_status() {
  local title="$1"
  local script="$2"
  echo
  echo "===== ${title} ====="
  if [[ -f "${SCRIPT_DIR}/${script}" ]]; then
    bash "${SCRIPT_DIR}/${script}" || true
  else
    echo "status script missing: ${script}"
  fi
}

run_status "STAMP coarse + FreeRef" "check_training_free_remaining_2gpu_status.sh"
run_status "STAMP official SAM-H four branches" "check_frozen_samh_status.sh"
run_status "Text4Seg-p24 four branches" "check_text4seg_public_p24_status.sh"
run_status "LISA paper protocol + FreeRef" "check_lisa_paper_freeref_status.sh"

echo
echo "===== Related launchers/workers ====="
pgrep -af '[r]un_freeref_remaining_4gpu|[r]un_text4seg_public_p24_full_eval|[r]un_frozen_samh_full_eval|[r]un_lisa_paper_freeref_eval|[e]val_text4seg_sam_h|[e]val_stamp_sam_h|[e]val_lisa_paper_protocol|[u]niversal_freeref.evaluate' || echo "none"

echo
echo "===== Recent scheduler logs ====="
for log in \
  "${ROOT}/outputs/freeref_4gpu_text4seg.log" \
  "${ROOT}/outputs/freeref_4gpu_stamp_samh.log" \
  "${ROOT}/outputs/freeref_4gpu_lisa_h100_0.log" \
  "${ROOT}/outputs/freeref_4gpu_lisa_h100_1.log"; do
  echo "--- ${log} ---"
  if [[ -f "${log}" ]]; then
    tail -n 5 "${log}"
  else
    echo "not created"
  fi
done
