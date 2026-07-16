#!/usr/bin/env bash
set -euo pipefail

ROOT="${MLLM_SEG_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG}"
GPU="${CUDA_DEVICE:-0}"
OUTPUT="${SPECIAL_COMBINED_OUTPUT:-${ROOT}/outputs/training_free_special_datasets_comparison}"

echo "GPU ${GPU}:"
nvidia-smi -i "${GPU}" --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader
echo
if [[ -f "${OUTPUT}/serial_status.tsv" ]]; then
  column -t -s $'\t' "${OUTPUT}/serial_status.tsv" 2>/dev/null || cat "${OUTPUT}/serial_status.tsv"
else
  echo "Serial status file: not created"
fi
echo
echo "Combined summary: $([[ -f "${OUTPUT}/combined_summary.json" ]] && echo yes || echo no)"
echo "Active processes:"
if ! pgrep -af 'run_training_free_special_datasets_serial|prepare_special_eval_data|export_stamp_refinement_dumps|training_free_refine.eval_stamp_dumps'; then
  echo "none"
fi
