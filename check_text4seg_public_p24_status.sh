#!/usr/bin/env bash
set -euo pipefail

ROOT="${MLLM_SEG_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG}"
COMBINED_OUTPUT="${TEXT4SEG_P24_COMBINED_OUTPUT:-${ROOT}/outputs/text4seg_public_p24_freeref_full_comparison}"

printf '%-20s %12s %10s %10s\n' split masks export summary
for ITEM in \
  refcoco_val:10834 refcoco_testA:5657 refcoco_testB:5095 \
  refcoco+_val:10758 refcoco+_testA:5726 refcoco+_testB:4889 \
  refcocog_val:4896 refcocog_test:9602; do
  SPLIT="${ITEM%%:*}"
  EXPECTED="${ITEM#*:}"
  BASE="${ROOT}/outputs/text4seg_official_${SPLIT}"
  REFINE="${ROOT}/outputs/text4seg_training_free_${SPLIT}"
  MASKS=0
  if [[ -d "${BASE}/gt_masks" ]]; then
    MASKS="$(find "${BASE}/gt_masks" -maxdepth 1 -type f -name '*.png' | wc -l)"
  fi
  printf '%-20s %6s/%-5s %10s %10s\n' \
    "${SPLIT}" "${MASKS}" "${EXPECTED}" \
    "$([[ -f "${BASE}/export_summary.json" ]] && echo yes || echo no)" \
    "$([[ -f "${REFINE}/eval_summary.json" ]] && echo yes || echo no)"
done

echo
echo "Combined summary: $([[ -f "${COMBINED_OUTPUT}/combined_summary.md" ]] && echo yes || echo no)"
echo "Active processes:"
pgrep -af 'run_text4seg_public_p24_full_eval|export_text4seg_masks.*p24|eval_text4seg_outputs' || echo none
