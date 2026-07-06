#!/usr/bin/env bash
set -euo pipefail

# 2x2 diagnosis for the failed prompt-prior smoke result.
#
# It answers:
# 1) Does prior text hurt the baseline model?
# 2) Did R-STAMP smoke fine-tuning damage the model even without prior text?
# 3) Is the failure specific to R-STAMP + prior at inference?

MLLM_SEG_ROOT="${MLLM_SEG_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG}"
STAMP_CODE_DIR="${STAMP_CODE_DIR:-${MLLM_SEG_ROOT}/code/STAMP}"
TOOL_REPO_DIR="${TOOL_REPO_DIR:-${MLLM_SEG_ROOT}/MLLM-SEG-data}"
EVAL_LIMIT="${EVAL_LIMIT:-50}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true
export STAMP_ATTN_IMPL="${STAMP_ATTN_IMPL:-sdpa}"
export STAMP_DISABLE_CUDNN="${STAMP_DISABLE_CUDNN:-1}"

BASELINE_MODEL="${MLLM_SEG_ROOT}/outputs/smoke_baseline_1x48g/final_model"
RSTAMP_MODEL="${MLLM_SEG_ROOT}/outputs/smoke_rstamp_1x48g/final_model"
BASELINE_JSON="${STAMP_CODE_DIR}/playground/data/json_files_baseline/refcocog_formatted_all_sentences_doubled_mp.json"
RSTAMP_JSON="${STAMP_CODE_DIR}/playground/data/json_files_rstamp/refcocog_formatted_all_sentences_doubled_mp.json"

cd "${STAMP_CODE_DIR}"

echo "=== A: baseline model, original query vs prior prompt ==="
python "${TOOL_REPO_DIR}/offline_rstamp/scripts/eval_smoke_iou.py" \
  --root "${MLLM_SEG_ROOT}" \
  --stamp-code-dir "${STAMP_CODE_DIR}" \
  --baseline-model "${BASELINE_MODEL}" \
  --rstamp-model "${BASELINE_MODEL}" \
  --baseline-json "${BASELINE_JSON}" \
  --rstamp-json "${RSTAMP_JSON}" \
  --output-dir "${MLLM_SEG_ROOT}/outputs/smoke_eval_iou_ablation/baseline_model_prior_enabled" \
  --limit "${EVAL_LIMIT}" \
  --rstamp-use-prior

echo "=== B: R-STAMP model, original query vs prior prompt ==="
python "${TOOL_REPO_DIR}/offline_rstamp/scripts/eval_smoke_iou.py" \
  --root "${MLLM_SEG_ROOT}" \
  --stamp-code-dir "${STAMP_CODE_DIR}" \
  --baseline-model "${RSTAMP_MODEL}" \
  --rstamp-model "${RSTAMP_MODEL}" \
  --baseline-json "${BASELINE_JSON}" \
  --rstamp-json "${RSTAMP_JSON}" \
  --output-dir "${MLLM_SEG_ROOT}/outputs/smoke_eval_iou_ablation/rstamp_model_no_prior_vs_prior" \
  --limit "${EVAL_LIMIT}" \
  --rstamp-use-prior

echo "Reports:"
find "${MLLM_SEG_ROOT}/outputs/smoke_eval_iou_ablation" -name "smoke_iou_comparison.md" -print
