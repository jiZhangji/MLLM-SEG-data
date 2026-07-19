#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${REPO_DIR}/.." && pwd)"

SPLIT="val"
GPU=0
COUNT=12
BATCH_SIZE=4
OUTPUT_DIR=""
QWEN_CHECKPOINT="${PROJECT_ROOT}/outputs/fair_7b_onepass6_stamp4_e2_v2/onepass7b/onepass_qwen7b.pt"
WARM_CHECKPOINT="${PROJECT_ROOT}/outputs/onepass7b_stamp_lora_warmstart_e2/onepass_qwen7b.pt"
GROUND_CHECKPOINT="${PROJECT_ROOT}/outputs/onepass7b_stamp_lora_seg_grounding_e2/onepass_qwen7b.pt"

usage() {
  cat <<'EOF'
Usage: bash run_onepass7b_three_method_visualization.sh [options]

Options:
  --split val|test
  --gpu ID
  --count N
  --batch-size N
  --output-dir PATH
  --qwen-checkpoint PATH
  --warm-checkpoint PATH
  --ground-checkpoint PATH
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --split) SPLIT="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --count) COUNT="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --qwen-checkpoint) QWEN_CHECKPOINT="$2"; shift 2 ;;
    --warm-checkpoint) WARM_CHECKPOINT="$2"; shift 2 ;;
    --ground-checkpoint) GROUND_CHECKPOINT="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ "${SPLIT}" != "val" && "${SPLIT}" != "test" ]]; then
  echo "--split must be val or test." >&2
  exit 2
fi

EVAL_JSON="${PROJECT_ROOT}/code/STAMP/playground/data/json_eval_baseline/refcocog_${SPLIT}.json"
QWEN_ROWS="${PROJECT_ROOT}/outputs/fair_7b_onepass6_stamp4_e2_v2/onepass7b_eval_${SPLIT}/onepass7b_eval_rows.csv"
WARM_ROWS="${PROJECT_ROOT}/outputs/onepass7b_stamp_lora_warmstart_e2_eval_${SPLIT}/onepass7b_eval_rows.csv"
GROUND_ROWS="${PROJECT_ROOT}/outputs/onepass7b_seg_grounding_eval_${SPLIT}/onepass7b_eval_rows.csv"
if [[ -z "${OUTPUT_DIR}" ]]; then
  OUTPUT_DIR="${PROJECT_ROOT}/outputs/onepass7b_three_method_visualization_${SPLIT}"
fi

for path in \
  "${QWEN_CHECKPOINT}" "${WARM_CHECKPOINT}" "${GROUND_CHECKPOINT}" \
  "${QWEN_ROWS}" "${WARM_ROWS}" "${GROUND_ROWS}" "${EVAL_JSON}"
do
  if [[ ! -f "${path}" ]]; then
    echo "Required file not found: ${path}" >&2
    exit 1
  fi
done

mkdir -p "${OUTPUT_DIR}"
cd "${REPO_DIR}"

echo "=== OnePass-7B three-method visualization ==="
echo "split=${SPLIT} gpu=${GPU} count=${COUNT} output=${OUTPUT_DIR}"

CUDA_VISIBLE_DEVICES="${GPU}" \
TOKENIZERS_PARALLELISM=false \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m onepass_qwen7b.visualize_methods \
  --run "Qwen-init" "${QWEN_CHECKPOINT}" "${QWEN_ROWS}" \
  --run "STAMP-LoRA-init" "${WARM_CHECKPOINT}" "${WARM_ROWS}" \
  --run "SEG-grounding" "${GROUND_CHECKPOINT}" "${GROUND_ROWS}" \
  --stamp-code-dir "${PROJECT_ROOT}/code/STAMP" \
  --base-model "${PROJECT_ROOT}/models/Qwen2-VL-7B-Instruct" \
  --eval-json "${EVAL_JSON}" \
  --data-root "${PROJECT_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --count "${COUNT}" \
  --batch-size "${BATCH_SIZE}" \
  --device cuda

echo "Done. Main files:"
echo "  ${OUTPUT_DIR}/method_iou_diagnostics.png"
echo "  ${OUTPUT_DIR}/visualization_summary.md"
echo "  ${OUTPUT_DIR}/cases/"
echo "  ${OUTPUT_DIR}/seg_branch/"
