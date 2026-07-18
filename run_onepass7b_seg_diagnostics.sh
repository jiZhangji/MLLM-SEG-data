#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${REPO_DIR}/.." && pwd)"

CHECKPOINT=""
OUTPUT_ROOT="${PROJECT_ROOT}/outputs/onepass7b_seg_diagnostics"
BASE_MODEL="${PROJECT_ROOT}/models/Qwen2-VL-7B-Instruct"
STAMP_CODE_DIR="${PROJECT_ROOT}/code/STAMP"
DATA_ROOT="${PROJECT_ROOT}"
VAL_JSON="${STAMP_CODE_DIR}/playground/data/json_eval_baseline/refcocog_val.json"
TEST_JSON="${STAMP_CODE_DIR}/playground/data/json_eval_baseline/refcocog_test.json"
GPU_VAL=0
GPU_TEST=1
BATCH_SIZE=8
NUM_WORKERS=8
LIMIT=0

usage() {
  cat <<'EOF'
Usage: bash run_onepass7b_seg_diagnostics.sh --checkpoint PATH [options]

Options:
  --output-root PATH   Output root (default: ../outputs/onepass7b_seg_diagnostics)
  --base-model PATH    Local Qwen2-VL-7B base model
  --gpu-val ID         GPU for RefCOCOg val (default: 0)
  --gpu-test ID        GPU for RefCOCOg test (default: 1)
  --batch-size N       Per-GPU evaluation batch size, must be >=2 (default: 8)
  --num-workers N      DataLoader workers per process (default: 8)
  --limit N            Samples per split; 0 means full split
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --base-model) BASE_MODEL="$2"; shift 2 ;;
    --gpu-val) GPU_VAL="$2"; shift 2 ;;
    --gpu-test) GPU_TEST="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --num-workers) NUM_WORKERS="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "${CHECKPOINT}" ]]; then
  echo "--checkpoint is required." >&2
  usage >&2
  exit 2
fi
if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT}" >&2
  exit 1
fi
if [[ ! -d "${BASE_MODEL}" ]]; then
  echo "Base model not found: ${BASE_MODEL}" >&2
  exit 1
fi
if [[ "${BATCH_SIZE}" -lt 2 ]]; then
  echo "--batch-size must be at least 2 for SEG shuffle." >&2
  exit 2
fi

mkdir -p "${OUTPUT_ROOT}/val" "${OUTPUT_ROOT}/test"
cd "${REPO_DIR}"

run_split() {
  local gpu="$1"
  local split="$2"
  local json="$3"
  local output_dir="${OUTPUT_ROOT}/${split}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
  TOKENIZERS_PARALLELISM=false \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python -m onepass_qwen7b.diagnose_seg \
    --stamp-code-dir "${STAMP_CODE_DIR}" \
    --base-model "${BASE_MODEL}" \
    --eval-json "${json}" \
    --data-root "${DATA_ROOT}" \
    --checkpoint "${CHECKPOINT}" \
    --output-dir "${output_dir}" \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --limit "${LIMIT}" \
    --device cuda \
    > "${output_dir}/diagnostics.log" 2>&1
}

echo "=== OnePass 7B SEG diagnostics ==="
echo "checkpoint=${CHECKPOINT}"
echo "val:  GPU ${GPU_VAL}, ${VAL_JSON}"
echo "test: GPU ${GPU_TEST}, ${TEST_JSON}"
echo "output=${OUTPUT_ROOT}"

run_split "${GPU_VAL}" val "${VAL_JSON}" &
VAL_PID=$!
run_split "${GPU_TEST}" test "${TEST_JSON}" &
TEST_PID=$!

cleanup() {
  kill "${VAL_PID}" "${TEST_PID}" 2>/dev/null || true
}
trap cleanup INT TERM

FAILED=0
if ! wait "${VAL_PID}"; then
  echo "val diagnostics failed; inspect ${OUTPUT_ROOT}/val/diagnostics.log" >&2
  FAILED=1
fi
if ! wait "${TEST_PID}"; then
  echo "test diagnostics failed; inspect ${OUTPUT_ROOT}/test/diagnostics.log" >&2
  FAILED=1
fi
trap - INT TERM
if [[ "${FAILED}" -ne 0 ]]; then
  exit 1
fi

python -m onepass_qwen7b.summarize_seg_diagnostics \
  --val-summary "${OUTPUT_ROOT}/val/seg_diagnostic_summary.json" \
  --test-summary "${OUTPUT_ROOT}/test/seg_diagnostic_summary.json" \
  --output-dir "${OUTPUT_ROOT}" \
  | tee "${OUTPUT_ROOT}/seg_diagnostics_val_test.txt"

echo "Done. Send these files for analysis:"
echo "  ${OUTPUT_ROOT}/seg_diagnostics_val_test.md"
echo "  ${OUTPUT_ROOT}/seg_diagnostics_val_test.json"
