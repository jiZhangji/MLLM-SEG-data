#!/usr/bin/env bash
set -euo pipefail

ROOT="${MLLM_SEG_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG}"
BASE_OUTPUT="${TEXT4SEG_P16_BASE_OUTPUT:-${ROOT}/outputs/text4seg_llava15_7b_p16_paired}"
REFINE_OUTPUT="${TEXT4SEG_P16_REFINE_OUTPUT:-${ROOT}/outputs/text4seg_llava15_7b_p16_freeref}"
SPLITS="${TEXT4SEG_P16_SPLITS:-refcoco_val refcoco_testA refcoco_testB refcoco+_val refcoco+_testA refcoco+_testB refcocog_val refcocog_test}"

printf '%-18s %12s %10s %8s\n' "split" "masks" "export" "summary"
for SPLIT in ${SPLITS}; do
  JSON="${ROOT}/code/STAMP/playground/data/json_eval_baseline/${SPLIT}.json"
  EXPECTED="?"
  if [[ -f "${JSON}" ]]; then
    EXPECTED="$(python -c 'import json,sys; print(len(json.load(open(sys.argv[1], encoding="utf-8"))))' "${JSON}")"
  fi
  COUNT="$(find "${BASE_OUTPUT}/${SPLIT}/gt_masks" -maxdepth 1 -type f -name '*.png' 2>/dev/null | wc -l)"
  EXPORT="$([[ -f "${BASE_OUTPUT}/${SPLIT}/export_summary.json" ]] && echo yes || echo no)"
  SUMMARY="$([[ -f "${REFINE_OUTPUT}/${SPLIT}/eval_summary.json" ]] && echo yes || echo no)"
  printf '%-18s %6s/%-5s %10s %8s\n' "${SPLIT}" "${COUNT}" "${EXPECTED}" "${EXPORT}" "${SUMMARY}"
done

echo
echo "Combined summary: $([[ -f "${ROOT}/outputs/text4seg_llava15_7b_p16_freeref_comparison/combined_summary.md" ]] && echo yes || echo no)"
echo "Active processes:"
pgrep -af 'run_text4seg_llava7b_p16_full_eval|export_text4seg_masks.*p16' || echo none
