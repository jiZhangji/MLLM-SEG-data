#!/usr/bin/env bash
set -euo pipefail

ROOT="${MLLM_SEG_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG}"
MODEL_PATH="${TEXT4SEG_P16_MODEL_PATH:-${ROOT}/models/Text4Seg/llava-v1.5-7b-p16}"
BASE_OUTPUT="${TEXT4SEG_P16_BASE_OUTPUT:-${ROOT}/outputs/text4seg_llava15_7b_p16_paired}"
REFINE_OUTPUT="${TEXT4SEG_P16_REFINE_OUTPUT:-${ROOT}/outputs/text4seg_llava15_7b_p16_freeref}"
COMBINED_OUTPUT="${TEXT4SEG_P16_COMBINED_OUTPUT:-${ROOT}/outputs/text4seg_llava15_7b_p16_freeref_comparison}"
SPLITS="${TEXT4SEG_P16_SPLITS:-refcoco_val refcoco_testA refcoco_testB refcoco+_val refcoco+_testA refcoco+_testB refcocog_val refcocog_test}"

echo "p16 checkpoint: $([[ -f "${MODEL_PATH}/config.json" ]] && echo ready || echo missing) (${MODEL_PATH})"
printf '%-18s %12s %10s %8s\n' "split" "masks" "export" "summary"
for SPLIT in ${SPLITS}; do
  JSON="${ROOT}/code/STAMP/playground/data/json_eval_baseline/${SPLIT}.json"
  EXPECTED="?"
  if [[ -f "${JSON}" ]]; then
    EXPECTED="$(python -c 'import json,sys; print(len(json.load(open(sys.argv[1], encoding="utf-8"))))' "${JSON}")"
  fi
  COUNT=0
  if [[ -d "${BASE_OUTPUT}/${SPLIT}/gt_masks" ]]; then
    COUNT="$(find "${BASE_OUTPUT}/${SPLIT}/gt_masks" -maxdepth 1 -type f -name '*.png' | wc -l)"
  fi
  EXPORT="$([[ -f "${BASE_OUTPUT}/${SPLIT}/export_summary.json" ]] && echo yes || echo no)"
  SUMMARY="$([[ -f "${REFINE_OUTPUT}/${SPLIT}/eval_summary.json" ]] && echo yes || echo no)"
  printf '%-18s %6s/%-5s %10s %8s\n' "${SPLIT}" "${COUNT}" "${EXPECTED}" "${EXPORT}" "${SUMMARY}"
done

echo
echo "Combined summary: $([[ -f "${COMBINED_OUTPUT}/combined_summary.md" ]] && echo yes || echo no)"
echo "Active processes:"
if command -v pgrep >/dev/null 2>&1; then
  pgrep -af 'run_text4seg_llava7b_p16_full_eval|export_text4seg_masks.*p16' || echo none
else
  echo none
fi
