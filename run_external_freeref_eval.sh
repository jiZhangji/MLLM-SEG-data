#!/usr/bin/env bash
set -euo pipefail

ROOT="${MLLM_SEG_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
METHOD="${METHOD:?Set METHOD, for example METHOD=HIPIE}"
SPLIT="${SPLIT:?Set SPLIT, for example SPLIT=refcoco_val}"
PREDICTION_ROOT="${PREDICTION_ROOT:?Set PREDICTION_ROOT to the saved prediction directory}"
PREDICTION_TEMPLATE="${PREDICTION_TEMPLATE:-}"
[[ -n "${PREDICTION_TEMPLATE}" ]] || PREDICTION_TEMPLATE='{index:08d}.png'
PREDICTION_KIND="${PREDICTION_KIND:-mask}"
ARRAY_KEY="${ARRAY_KEY:-}"
EVAL_JSON="${EVAL_JSON:-${ROOT}/code/STAMP/playground/data/json_eval_baseline/${SPLIT}.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/outputs/external_freeref/${METHOD// /_}/${SPLIT//+/plus}}"
CONDA_ENV="${FREEREF_CONDA_ENV:-STAMP}"
LIMIT="${EVAL_LIMIT:-0}"
OFFSET="${EVAL_OFFSET:-0}"

mkdir -p "${OUTPUT_DIR}/import" "${OUTPUT_DIR}/freeref"
EXTRA_IMPORT_ARGS=()
[[ -n "${ARRAY_KEY}" ]] && EXTRA_IMPORT_ARGS+=(--array-key "${ARRAY_KEY}")
PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}" conda run --no-capture-output -n "${CONDA_ENV}" \
  python -m universal_freeref.build_eval_json_manifest \
    --method "${METHOD}" \
    --split "${SPLIT}" \
    --eval-json "${EVAL_JSON}" \
    --data-root "${ROOT}" \
    --prediction-root "${PREDICTION_ROOT}" \
    --prediction-template "${PREDICTION_TEMPLATE}" \
    --prediction-kind "${PREDICTION_KIND}" \
    --output-dir "${OUTPUT_DIR}/import" \
    --limit "${LIMIT}" \
    --offset "${OFFSET}" \
    "${EXTRA_IMPORT_ARGS[@]}"

PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}" conda run --no-capture-output -n "${CONDA_ENV}" \
  python -m universal_freeref.evaluate \
    --manifest "${OUTPUT_DIR}/import/manifest.jsonl" \
    --output-dir "${OUTPUT_DIR}/freeref" \
    --n-segments 1024 \
    --graph-lambda 1.0 \
    --boundary-sigma 8.0 \
    --save-visualizations 12

echo "External paired FreeRef result: ${OUTPUT_DIR}/freeref/eval_summary.json"
