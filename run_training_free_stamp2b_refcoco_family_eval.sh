#!/usr/bin/env bash
set -euo pipefail

ROOT="${MLLM_SEG_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${TRAINING_FREE_REPO:-${SCRIPT_DIR}}"
STAMP_ENV="${STAMP_ENV_PATH:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/my_global_cache/conda/envs/STAMP}"
MODEL_NAME="${STAMP2B_MODEL_NAME:-${ROOT}/models/STAMP-2B-uni}"
GPU="${CUDA_DEVICE:-1}"
EVAL_LIMIT="${STAMP2B_OTHER_EVAL_LIMIT:-0}"
BATCH_SIZE="${STAMP2B_EVAL_BATCH_SIZE:-1}"
SPLITS_TEXT="${STAMP2B_OTHER_SPLITS:-refcoco_val refcoco_testA refcoco_testB refcoco+_val refcoco+_testA refcoco+_testB}"
STAMP_DATA="${ROOT}/code/STAMP/playground/data"
JSON_DIR="${STAMP_DATA}/json_eval_baseline"
MASK_ROOT="${STAMP_DATA}/masks_eval_baseline"
COMBINED_OUTPUT="${STAMP2B_OTHER_COMBINED_OUTPUT:-${ROOT}/outputs/training_free_refine_stamp2b_refcoco_family_full_comparison}"

if [[ ! -x "${STAMP_ENV}/bin/python" ]]; then
  echo "ERROR: STAMP Python environment not found at ${STAMP_ENV}." >&2
  exit 1
fi
if [[ ! -d "${MODEL_NAME}" ]]; then
  echo "ERROR: STAMP-2B model not found at ${MODEL_NAME}." >&2
  exit 1
fi
if [[ ! -d "${ROOT}/data/annotations/refcoco_family" ]]; then
  echo "ERROR: RefCOCO-family annotations not found below ${ROOT}/data/annotations/refcoco_family." >&2
  exit 1
fi

export PATH="${STAMP_ENV}/bin:${PATH}"
export CUDA_VISIBLE_DEVICES="${GPU}"
export TOKENIZERS_PARALLELISM=false
export STAMP_DISABLE_CUDNN="${STAMP_DISABLE_CUDNN:-1}"
mkdir -p "${ROOT}/outputs" "${JSON_DIR}" "${MASK_ROOT}" "${COMBINED_OUTPUT}"

if command -v flock >/dev/null 2>&1; then
  exec 9>"${ROOT}/outputs/.training_free_stamp2b_refcoco_family.lock"
  if ! flock -n 9; then
    echo "Another STAMP-2B RefCOCO-family Training-Free job already holds the lock." >&2
    exit 0
  fi
fi

read -r -a SPLITS <<< "${SPLITS_TEXT}"
if (( ${#SPLITS[@]} == 0 )); then
  echo "ERROR: no evaluation splits were selected." >&2
  exit 1
fi

cd "${REPO}"
if (( BATCH_SIZE > 1 )); then
  echo "Refreshing STAMP refinement export patch for batch size ${BATCH_SIZE}."
  (
    if command -v flock >/dev/null 2>&1; then flock 8; fi
    python offline_rstamp/scripts/patch_stamp_refinement_export.py \
      --stamp-code-dir "${ROOT}/code/STAMP" \
      --target both
  ) 8>"${ROOT}/outputs/.stamp_refinement_export_patch.lock"
fi
needs_prepare=0
for split in "${SPLITS[@]}"; do
  if [[ ! -f "${JSON_DIR}/${split}.json" ]]; then
    needs_prepare=1
    break
  fi
done
if (( needs_prepare )); then
  echo "[1/3] Preparing missing baseline RefCOCO-family evaluation JSONs"
  python offline_rstamp/scripts/prepare_refcoco_eval_data.py \
    --root "${ROOT}" \
    --splits "${SPLITS[@]}" \
    --prior-mode none \
    --output-json-dir "${JSON_DIR}" \
    --mask-root "${MASK_ROOT}"
else
  echo "[1/3] Reusing existing RefCOCO-family JSONs and masks"
fi

echo "[2/3] Exporting/resuming STAMP-2B dumps and applying the frozen refiner"
SUMMARY_ARGS=()
for split in "${SPLITS[@]}"; do
  safe_split="${split//+/plus}"
  if [[ "${EVAL_LIMIT}" == "0" ]]; then
    run_suffix="full"
  else
    run_suffix="limit${EVAL_LIMIT}"
  fi
  json_path="${JSON_DIR}/${split}.json"
  dump_dir="${ROOT}/outputs/refine_stamp_dumps/${safe_split}_${run_suffix}_stamp2b"
  result_dir="${ROOT}/outputs/training_free_refine_stamp2b_${safe_split}_${run_suffix}"
  expected="$(python -c 'import json,sys; n=len(json.load(open(sys.argv[1], encoding="utf-8"))); limit=int(sys.argv[2]); print(min(n, limit) if limit else n)' "${json_path}" "${EVAL_LIMIT}")"
  mkdir -p "${dump_dir}" "${result_dir}"
  existing="$(find "${dump_dir}" -maxdepth 1 -type f -name '*.pt' 2>/dev/null | wc -l)"
  echo "--- ${split}: existing dumps ${existing} / ${expected} ---"
  MODEL_NAME="${MODEL_NAME}" \
  SPLIT="${split}" \
  JSON_PATH="${json_path}" \
  EVAL_LIMIT="${EVAL_LIMIT}" \
  OUTPUT_DIR="${dump_dir}" \
  BATCH_SIZE="${BATCH_SIZE}" \
    bash offline_rstamp/run/75_export_refcocog_refine_stamp_dumps.sh
  python -m training_free_refine.eval_stamp_dumps \
    --input-dir "${dump_dir}" \
    --output-dir "${result_dir}" \
    --n-segments 1024 \
    --graph-lambda 1.0 \
    --save-visualizations 8
  SUMMARY_ARGS+=(--summary "${split}=${result_dir}/eval_summary.json")
done

echo "[3/3] Combining all STAMP-2B split summaries"
python -m training_free_refine.summarize_splits \
  "${SUMMARY_ARGS[@]}" \
  --output-dir "${COMBINED_OUTPUT}" \
  --title "STAMP-2B Training-Free RefCOCO and RefCOCO+ Evaluation"

echo "STAMP-2B RefCOCO-family Training-Free evaluation completed."
echo "Summary: ${COMBINED_OUTPUT}/combined_summary.md"
