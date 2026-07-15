#!/usr/bin/env bash
set -euo pipefail

ROOT="${MLLM_SEG_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG}"

count_files() {
  local directory="$1"
  local pattern="$2"
  if [[ ! -d "${directory}" ]]; then
    echo 0
    return
  fi
  find "${directory}" -maxdepth 1 -type f -name "${pattern}" | wc -l
}

TEXT4SEG_RESULTS="${ROOT}/outputs/text4seg_official_refcocog_val"
TEXT4SEG_SUMMARY="${ROOT}/outputs/text4seg_training_free_refcocog_val/eval_summary.json"
STAMP7B_VAL="${ROOT}/outputs/refine_stamp_dumps/refcocog_val_full_stamp7b"
STAMP7B_TEST="${ROOT}/outputs/refine_stamp_dumps/refcocog_test_full_stamp7b"
STAMP7B_SUMMARY="${ROOT}/outputs/training_free_refine_stamp7b_refcocog_full_comparison/combined_summary.json"

echo "Text4Seg complete samples: $(count_files "${TEXT4SEG_RESULTS}/gt_masks" '*.png') / 4896"
echo "Text4Seg final summary: $([[ -f "${TEXT4SEG_SUMMARY}" ]] && echo yes || echo no)"
echo "STAMP-7B val dumps: $(count_files "${STAMP7B_VAL}" '*.pt') / 4896"
echo "STAMP-7B test dumps: $(count_files "${STAMP7B_TEST}" '*.pt') / 9602"
echo "STAMP-7B final summary: $([[ -f "${STAMP7B_SUMMARY}" ]] && echo yes || echo no)"
echo
echo "Active related processes:"
if ! pgrep -af 'run_text4seg_training_free_eval|run_training_free_stamp7b_full_eval|training_free_refine.export_text4seg_masks|training_free_refine.eval_text4seg_outputs|export_stamp_refinement_dumps|training_free_refine.eval_stamp_dumps'; then
  echo "none"
fi
