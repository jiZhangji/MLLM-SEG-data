#!/usr/bin/env bash
set -euo pipefail

MLLM_SEG_ROOT="${MLLM_SEG_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG}"
TOOL_REPO="${MLLM_SEG_ROOT}/MLLM-SEG-data"
STAMP_DATA="${MLLM_SEG_ROOT}/code/STAMP/playground/data"
JSON_PATH="${JSON_PATH:-${STAMP_DATA}/json_files_baseline/refcocog_formatted_all_sentences_doubled_mp.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${MLLM_SEG_ROOT}/outputs/refine_stamp_dumps/refcocog_train_0}"
GPU_LIST="${GPU_LIST:-0 1 2 3}"
TOTAL_ITEMS="${TOTAL_ITEMS:-0}"

cd "${TOOL_REPO}"

if [[ "${TOTAL_ITEMS}" == "0" ]]; then
  TOTAL_ITEMS="$(python - <<PY
import json
from pathlib import Path
print(len(json.loads(Path("${JSON_PATH}").read_text(encoding="utf-8"))))
PY
)"
fi

read -r -a GPUS <<< "${GPU_LIST}"
NUM_GPUS="${#GPUS[@]}"
if [[ "${NUM_GPUS}" -lt 1 ]]; then
  echo "GPU_LIST is empty." >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"
CHUNK_SIZE="$(( (TOTAL_ITEMS + NUM_GPUS - 1) / NUM_GPUS ))"

echo "Parallel export:"
echo "  json: ${JSON_PATH}"
echo "  output: ${OUTPUT_DIR}"
echo "  total_items: ${TOTAL_ITEMS}"
echo "  gpu_list: ${GPU_LIST}"
echo "  chunk_size: ${CHUNK_SIZE}"

pids=()
for idx in "${!GPUS[@]}"; do
  gpu="${GPUS[$idx]}"
  offset="$(( idx * CHUNK_SIZE ))"
  remaining="$(( TOTAL_ITEMS - offset ))"
  if [[ "${remaining}" -le 0 ]]; then
    continue
  fi
  limit="${CHUNK_SIZE}"
  if [[ "${remaining}" -lt "${CHUNK_SIZE}" ]]; then
    limit="${remaining}"
  fi
  log_path="${OUTPUT_DIR}/export_gpu${gpu}_offset${offset}_limit${limit}.log"
  echo "Starting GPU ${gpu}: offset=${offset}, limit=${limit}, log=${log_path}"
  (
    CUDA_VISIBLE_DEVICES="${gpu}" \
    SPLIT="refcocog_train" \
    EVAL_LIMIT="${limit}" \
    OFFSET="${offset}" \
    JSON_PATH="${JSON_PATH}" \
    OUTPUT_DIR="${OUTPUT_DIR}" \
    bash offline_rstamp/run/75_export_refcocog_refine_stamp_dumps.sh
  ) > "${log_path}" 2>&1 &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "${pid}"
done

echo "Parallel export complete."
echo "Dump count:"
find "${OUTPUT_DIR}" -maxdepth 1 -name "*.pt" | wc -l
