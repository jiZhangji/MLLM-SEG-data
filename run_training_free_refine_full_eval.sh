#!/usr/bin/env bash
set -euo pipefail

ROOT="${MLLM_SEG_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG}"
REPO="${ROOT}/MLLM-SEG-data"
VAL_DUMPS="${VAL_DUMPS:-${ROOT}/outputs/refine_stamp_dumps/refcocog_val_full}"
TEST_DUMPS="${TEST_DUMPS:-${ROOT}/outputs/refine_stamp_dumps/refcocog_test_full}"
VAL_OUTPUT="${VAL_OUTPUT:-${ROOT}/outputs/training_free_refine_refcocog_val_full}"
TEST_OUTPUT="${TEST_OUTPUT:-${ROOT}/outputs/training_free_refine_refcocog_test_full}"
COMBINED_OUTPUT="${COMBINED_OUTPUT:-${ROOT}/outputs/training_free_refine_refcocog_full_comparison}"

cd "${REPO}"

echo "[1/4] Running training-free refinement tests"
python -m unittest tests.test_training_free_refine

echo "[2/4] Evaluating complete RefCOCOg val dumps"
python -m training_free_refine.eval_stamp_dumps \
  --input-dir "${VAL_DUMPS}" \
  --output-dir "${VAL_OUTPUT}" \
  --n-segments 1024 \
  --graph-lambda 1.0 \
  --save-visualizations 8

echo "[3/4] Evaluating complete RefCOCOg test dumps"
python -m training_free_refine.eval_stamp_dumps \
  --input-dir "${TEST_DUMPS}" \
  --output-dir "${TEST_OUTPUT}" \
  --n-segments 1024 \
  --graph-lambda 1.0 \
  --save-visualizations 8

echo "[4/4] Combining val/test results"
python -m training_free_refine.summarize_results \
  --val-summary "${VAL_OUTPUT}/eval_summary.json" \
  --test-summary "${TEST_OUTPUT}/eval_summary.json" \
  --output-dir "${COMBINED_OUTPUT}"

echo "Training-free full evaluation completed."
