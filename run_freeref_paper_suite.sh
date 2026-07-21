#!/usr/bin/env bash
set -euo pipefail

ROOT="${MLLM_SEG_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_PIXELLM="${PAPER_RUN_PIXELLM:-1}"
RUN_SEGAGENT="${PAPER_RUN_SEGAGENT:-auto}"
REFINE_ENV="${FREEREF_CONDA_ENV:-STAMP}"
PAPER_OUTPUT="${FREEREF_PAPER_OUTPUT:-${ROOT}/outputs/freeref_paper_suite}"

mkdir -p "${PAPER_OUTPUT}"
cd "${SCRIPT_DIR}"

if [[ "${RUN_PIXELLM}" == "1" ]]; then
  echo "===== [1/3] PixelLM public checkpoint -> FreeRef ====="
  bash run_pixellm_freeref_full_eval.sh
else
  echo "===== [1/3] PixelLM skipped by PAPER_RUN_PIXELLM=${RUN_PIXELLM} ====="
fi

run_segagent=0
if [[ "${RUN_SEGAGENT}" == "1" ]]; then
  run_segagent=1
elif [[ "${RUN_SEGAGENT}" == "auto" ]]; then
  model="${SEGAGENT_MODEL_PATH:-${ROOT}/models/freeref_missing_methods/segagent/SegAgent-Model}"
  dataset="${SEGAGENT_DATASET_ROOT:-${ROOT}/models/freeref_missing_methods/segagent/SegAgent-Dataset}"
  click_root="${SEGAGENT_SIMPLECLICK_ROOT:-${ROOT}/models/freeref_missing_methods/segagent/simpleclick_models}"
  segagent_env="${SEGAGENT_CONDA_ENV:-segagent-freeref}"
  if [[ ! -f "${model}/config.json" ]]; then
    nested_model="$(find "${model}" -maxdepth 3 -type f -name config.json -printf '%h\n' -quit 2>/dev/null || true)"
    [[ -n "${nested_model}" ]] && model="${nested_model}"
  fi
  if conda env list | awk 'NF && $1 !~ /^#/ {print $1}' | grep -Fxq "${segagent_env}" && \
     [[ -f "${model}/config.json" && -d "${dataset}" ]] && \
     find "${click_root}" -type f -name '*large*.pth' -print -quit 2>/dev/null | grep -q .; then
    run_segagent=1
  fi
fi
if (( run_segagent == 1 )); then
  echo "===== [2/3] SegAgent-SimpleClick -> FreeRef ====="
  bash run_segagent_freeref_full_eval.sh
else
  echo "===== [2/3] SegAgent skipped: public model/data/SimpleClick checkpoint is incomplete ====="
fi

echo "===== [3/3] Building provenance-aware paper table ====="
PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}" conda run --no-capture-output -n "${REFINE_ENV}" \
  python -m universal_freeref.summarize_paper_suite \
    --root "${ROOT}" \
    --output-dir "${PAPER_OUTPUT}"

echo "FreeRef paper suite complete: ${PAPER_OUTPUT}/paper_results.md"
