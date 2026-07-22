#!/usr/bin/env bash
set -euo pipefail

ROOT="${MLLM_SEG_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU="${STAMP2B_REPAIR_GPU:-0}"
SAMH_JOBS="${STAMP2B_REPAIR_SAMH_PARALLEL_JOBS:-2}"
SAMH_FREE_MB="${STAMP2B_REPAIR_SAMH_MIN_FREE_MB:-24000}"
STAMP_ENV_PATH="${STAMP_ENV_PATH:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/my_global_cache/conda/envs/STAMP}"
SPLITS="refcoco_testA refcoco+_testB"

cd "${SCRIPT_DIR}"
echo "Repairing the two missing STAMP-2B samples on physical GPU ${GPU}."
CUDA_DEVICE="${GPU}" \
STAMP2B_OTHER_SPLITS="${SPLITS}" \
EMPTY_ON_FAILURE=1 \
  bash run_training_free_stamp2b_refcoco_family_eval.sh

for specification in \
  "refcoco_testA:refcoco_testA_full_stamp2b" \
  "refcoco+_testB:refcoplus_testB_full_stamp2b"; do
  split="${specification%%:*}"
  dump_name="${specification#*:}"
  expected="$("${STAMP_ENV_PATH}/bin/python" -c \
    'import json,sys; print(len(json.load(open(sys.argv[1], encoding="utf-8"))))' \
    "${ROOT}/code/STAMP/playground/data/json_eval_baseline/${split}.json")"
  actual="$(find "${ROOT}/outputs/refine_stamp_dumps/${dump_name}" -maxdepth 1 -type f -name '*.pt' | wc -l)"
  if [[ "${actual}" != "${expected}" ]]; then
    echo "ERROR: ${split} still has ${actual}/${expected} dumps after repair." >&2
    exit 1
  fi
  echo "VERIFIED ${split}: ${actual}/${expected} dumps"
done

"${STAMP_ENV_PATH}/bin/python" -m training_free_refine.summarize_splits \
  --summary "refcoco_val=${ROOT}/outputs/training_free_refine_stamp2b_refcoco_val_full/eval_summary.json" \
  --summary "refcoco_testA=${ROOT}/outputs/training_free_refine_stamp2b_refcoco_testA_full/eval_summary.json" \
  --summary "refcoco_testB=${ROOT}/outputs/training_free_refine_stamp2b_refcoco_testB_full/eval_summary.json" \
  --summary "refcoco+_val=${ROOT}/outputs/training_free_refine_stamp2b_refcoplus_val_full/eval_summary.json" \
  --summary "refcoco+_testA=${ROOT}/outputs/training_free_refine_stamp2b_refcoplus_testA_full/eval_summary.json" \
  --summary "refcoco+_testB=${ROOT}/outputs/training_free_refine_stamp2b_refcoplus_testB_full/eval_summary.json" \
  --output-dir "${ROOT}/outputs/training_free_refine_stamp2b_refcoco_family_full_comparison" \
  --title "STAMP-2B Training-Free RefCOCO and RefCOCO+ Evaluation"

timestamp="$(date '+%Y%m%d_%H%M%S')"
for directory in \
  "${ROOT}/outputs/stamp_official_samh_stamp-2b_refcoco_testA" \
  "${ROOT}/outputs/stamp_official_samh_stamp-2b_refcoplus_testB"; do
  if [[ -f "${directory}/eval_summary.json" ]]; then
    mv "${directory}/eval_summary.json" "${directory}/eval_summary.before_sample_repair_${timestamp}.json"
  fi
done

CUDA_DEVICE="${GPU}" \
SAMH_ONLY_JOBS="STAMP-2B:refcoco_testA STAMP-2B:refcoco+_testB" \
SAMH_PARALLEL_JOBS="${SAMH_JOBS}" \
SAMH_MIN_FREE_MB="${SAMH_FREE_MB}" \
  bash run_frozen_samh_full_eval.sh

echo "STAMP-2B missing-sample repair completed."
echo "Status: bash check_training_free_remaining_2gpu_status.sh && bash check_frozen_samh_status.sh"
