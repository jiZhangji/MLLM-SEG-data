#!/usr/bin/env bash
set -euo pipefail

ROOT="${MLLM_SEG_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GSVA_DIR="${GSVA_DIR:-${ROOT}/code/third_party/gsva}"
CONDA_ENV="${GSVA_CONDA_ENV:-gsva-freeref}"
WEIGHTS_ROOT="${GSVA_WEIGHTS_ROOT:-${ROOT}/models/freeref_missing_methods}"
BASE_MODEL="${GSVA_LLAMA7B_BASE:-}"
DELTA="${GSVA_LLAVA_DELTA:-${WEIGHTS_ROOT}/gsva/LLaVA-Lightning-7B-delta-v1-1}"
TARGET="${GSVA_MLLM_MODEL_PATH:-${WEIGHTS_ROOT}/gsva/LLaVA-Lightning-7B-v1-1-merged}"
MERGE_MARKER="${TARGET}/.freeref_merge_complete"

if [[ -f "${MERGE_MARKER}" && -f "${TARGET}/config.json" ]] &&
   find "${TARGET}" -maxdepth 1 -type f \( -name '*.bin' -o -name '*.safetensors' \) \
     -size +100M -print -quit | grep -q .; then
  echo "SKIP existing merged GSVA LLaVA base: ${TARGET}"
  exit 0
fi
if [[ -z "${BASE_MODEL}" ]]; then
  echo "ERROR: GSVA requires the licensed original LLaMA-7B Hugging Face base." >&2
  echo "Set GSVA_LLAMA7B_BASE=/path/to/your/authorized/llama-7b-hf and rerun." >&2
  exit 3
fi
for required in "${BASE_MODEL}/config.json" "${DELTA}/config.json"; do
  [[ -f "${required}" ]] || { echo "ERROR: missing GSVA merge prerequisite: ${required}" >&2; exit 2; }
done
echo "Applying the official public LLaVA delta to the user-supplied licensed base."
PYTHONPATH="${GSVA_DIR}:${SCRIPT_DIR}:${PYTHONPATH:-}" \
  conda run --no-capture-output -n "${CONDA_ENV}" \
  python -m universal_freeref.apply_llava_delta \
    --gsva-code-dir "${GSVA_DIR}" \
    --base-model-path "${BASE_MODEL}" \
    --delta-path "${DELTA}" \
    --target-model-path "${TARGET}"
touch "${MERGE_MARKER}"
echo "GSVA merged LLaVA base is ready: ${TARGET}"
