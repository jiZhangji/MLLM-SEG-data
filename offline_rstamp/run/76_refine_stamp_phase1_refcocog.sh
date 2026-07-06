#!/usr/bin/env bash
set -euo pipefail

# End-to-end Phase-1 diagnosis:
#   1) export STAMP Phase-2 tensors on a small RefCOCOg split;
#   2) evaluate whether uncertainty/boundary selectors beat random.

MLLM_SEG_ROOT="${MLLM_SEG_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG}"
TOOL_REPO="${MLLM_SEG_ROOT}/MLLM-SEG-data"
SPLIT="${SPLIT:-refcocog_val}"
EVAL_LIMIT="${EVAL_LIMIT:-20}"
DUMP_DIR="${DUMP_DIR:-${MLLM_SEG_ROOT}/outputs/refine_stamp_dumps/${SPLIT}_${EVAL_LIMIT}}"
QUALITY_DIR="${QUALITY_DIR:-${MLLM_SEG_ROOT}/outputs/refine_stamp_phase1_selector_quality/${SPLIT}_${EVAL_LIMIT}}"
TOP_K="${TOP_K:-64}"
VISUALIZE_LIMIT="${VISUALIZE_LIMIT:-8}"

cd "${TOOL_REPO}"

OUTPUT_DIR="${DUMP_DIR}" bash offline_rstamp/run/75_export_refcocog_refine_stamp_dumps.sh

OUTPUT_DIR="${QUALITY_DIR}" \
TOP_K="${TOP_K}" \
VISUALIZE_LIMIT="${VISUALIZE_LIMIT}" \
bash offline_rstamp/run/73_eval_refine_stamp_selector_quality.sh "${DUMP_DIR}"

echo "Phase-1 selector quality report:"
echo "${QUALITY_DIR}/selector_quality_summary.md"
