#!/usr/bin/env bash
set -euo pipefail

ROOT="${MLLM_SEG_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV="${LISA_CONDA_ENV:-lisa-freeref}"
SPLITS="${LISA_PAPER_SPLITS:-refcoco|unc|testA}"
PAPER_OUTPUT_ROOT="${LISA_PAPER_OUTPUT_ROOT:-${ROOT}/outputs/lisa_paper_reproduction}"
FREEREF_OUTPUT_ROOT="${LISA_PAPER_FREEREF_OUTPUT_ROOT:-${ROOT}/outputs/lisa_paper_freeref}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
MIN_FREE_MB="${MIN_FREE_MB:-30000}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

mkdir -p "${PAPER_OUTPUT_ROOT}" "${FREEREF_OUTPUT_ROOT}"
while true; do
  FREE_MB="$(nvidia-smi -i "${CUDA_DEVICE}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -dc '0-9')"
  if [[ -n "${FREE_MB}" ]] && (( FREE_MB >= MIN_FREE_MB )); then
    break
  fi
  echo "GPU ${CUDA_DEVICE} free: ${FREE_MB:-unknown} MiB; waiting 10 seconds for ${MIN_FREE_MB} MiB..."
  sleep 10
done

LISA_REQUIRE_PAPER_MATCH=1 \
LISA_PAPER_SPLITS="${SPLITS}" \
LISA_PAPER_OUTPUT_ROOT="${PAPER_OUTPUT_ROOT}" \
CUDA_DEVICE="${CUDA_DEVICE}" \
  bash "${SCRIPT_DIR}/run_lisa_paper_reproduction.sh"

for split_spec in ${SPLITS}; do
  slug="${split_spec//|/_}"
  paper_dir="${PAPER_OUTPUT_ROOT}/${slug}"
  summary="${paper_dir}/paper_reproduction_summary.json"
  manifest="${paper_dir}/manifest.jsonl"
  output_dir="${FREEREF_OUTPUT_ROOT}/${slug}"
  paper_match="$(conda run -n "${CONDA_ENV}" python -c \
    'import json,sys; print(int(json.load(open(sys.argv[1], encoding="utf-8")).get("paper_match") is True))' \
    "${summary}")"
  if [[ "${paper_match}" != "1" ]]; then
    echo "ERROR: refusing FreeRef evaluation because the LISA paper baseline did not match: ${summary}" >&2
    exit 3
  fi
  echo "===== Apply FreeRef to verified LISA baseline: ${split_spec} ====="
  CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}" \
    conda run --no-capture-output -n "${CONDA_ENV}" \
    python -m universal_freeref.evaluate \
      --manifest "${manifest}" \
      --output-dir "${output_dir}" \
      --save-visualizations 12
done

echo "LISA paper-protocol + FreeRef evaluation completed."
echo "Baseline: ${PAPER_OUTPUT_ROOT}"
echo "FreeRef: ${FREEREF_OUTPUT_ROOT}"
