#!/usr/bin/env bash
set -euo pipefail

ROOT="${MLLM_SEG_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG}"

printf '%-10s %-18s %12s %10s\n' "model" "split" "SAM masks" "summary"
for spec in \
  "stamp-2b refcoco_val refcoco_val_full_stamp2b" \
  "stamp-2b refcoco_testA refcoco_testA_full_stamp2b" \
  "stamp-2b refcoco_testB refcoco_testB_full_stamp2b" \
  "stamp-2b refcoco+_val refcocoplus_val_full_stamp2b" \
  "stamp-2b refcoco+_testA refcocoplus_testA_full_stamp2b" \
  "stamp-2b refcoco+_testB refcocoplus_testB_full_stamp2b" \
  "stamp-2b refcocog_val refcocog_val_full" \
  "stamp-2b refcocog_test refcocog_test_full" \
  "stamp-7b refcoco_val refcoco_val_full_stamp7b" \
  "stamp-7b refcoco_testA refcoco_testA_full_stamp7b" \
  "stamp-7b refcoco_testB refcoco_testB_full_stamp7b" \
  "stamp-7b refcoco+_val refcocoplus_val_full_stamp7b" \
  "stamp-7b refcoco+_testA refcocoplus_testA_full_stamp7b" \
  "stamp-7b refcoco+_testB refcocoplus_testB_full_stamp7b" \
  "stamp-7b refcocog_val refcocog_val_full_stamp7b" \
  "stamp-7b refcocog_test refcocog_test_full_stamp7b"; do
  read -r MODEL SPLIT DUMPS <<<"${spec}"
  SAFE_SPLIT="${SPLIT//+/plus}"
  OUT="${ROOT}/outputs/frozen_samh_${MODEL}_${SAFE_SPLIT}"
  EXPECTED="$(find "${ROOT}/outputs/refine_stamp_dumps/${DUMPS}" -maxdepth 1 -type f -name '*.pt' 2>/dev/null | wc -l || true)"
  COMPLETE="$(find "${OUT}/coarse_sam_masks" -maxdepth 1 -type f -name '*.png' 2>/dev/null | wc -l || true)"
  SUMMARY="$([[ -f "${OUT}/eval_summary.json" ]] && echo yes || echo no)"
  printf '%-10s %-18s %5s/%-6s %10s\n' "${MODEL}" "${SPLIT}" "${COMPLETE}" "${EXPECTED}" "${SUMMARY}"
done

echo
echo "Text4Seg existing frozen-SAM-H summaries:"
for split in val test; do
  path="${ROOT}/outputs/text4seg_training_free_refcocog_${split}/eval_summary.json"
  echo "  refcocog_${split}: $([[ -f "${path}" ]] && echo yes || echo no)"
done
echo "Combined summary: $([[ -f "${ROOT}/outputs/frozen_samh_full_comparison/combined_summary.md" ]] && echo yes || echo no)"
echo
echo "Active processes:"
pgrep -af 'run_frozen_samh_full_eval|training_free_refine.eval_stamp_sam_h' || echo none
