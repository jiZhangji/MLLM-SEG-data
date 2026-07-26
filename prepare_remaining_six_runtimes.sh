#!/usr/bin/env bash
set -euo pipefail

ROOT="${MLLM_SEG_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROLE="${REMAINING_SIX_INSTANCE_ROLE:-all}"
echo "Preparing isolated runtimes serially to avoid conda/pip package-lock conflicts."
case "${ROLE}" in
  all)
    MLLM_SEG_ROOT="${ROOT}" bash "${SCRIPT_DIR}/prepare_read_freeref_assets.sh"
    MLLM_SEG_ROOT="${ROOT}" bash "${SCRIPT_DIR}/prepare_polyformer_freeref_assets.sh"
    MLLM_SEG_ROOT="${ROOT}" bash "${SCRIPT_DIR}/prepare_lisa_freeref_env.sh"
    MLLM_SEG_ROOT="${ROOT}" bash "${SCRIPT_DIR}/prepare_rela_freeref_assets.sh"
    MLLM_SEG_ROOT="${ROOT}" bash "${SCRIPT_DIR}/prepare_gsva_freeref_env.sh"
    ;;
  h100)
    MLLM_SEG_ROOT="${ROOT}" bash "${SCRIPT_DIR}/prepare_polyformer_freeref_assets.sh"
    MLLM_SEG_ROOT="${ROOT}" bash "${SCRIPT_DIR}/prepare_lisa_freeref_env.sh"
    MLLM_SEG_ROOT="${ROOT}" bash "${SCRIPT_DIR}/prepare_rela_freeref_assets.sh"
    ;;
  h200)
    MLLM_SEG_ROOT="${ROOT}" bash "${SCRIPT_DIR}/prepare_read_freeref_assets.sh"
    MLLM_SEG_ROOT="${ROOT}" bash "${SCRIPT_DIR}/prepare_rela_freeref_assets.sh"
    MLLM_SEG_ROOT="${ROOT}" bash "${SCRIPT_DIR}/prepare_gsva_freeref_env.sh"
    ;;
  *)
    echo "ERROR: REMAINING_SIX_INSTANCE_ROLE must be all, h100, or h200." >&2
    exit 2
    ;;
esac

GSVA_MERGED="${GSVA_MLLM_MODEL_PATH:-${ROOT}/models/freeref_missing_methods/gsva/LLaVA-Lightning-7B-v1-1-merged}"
if [[ "${ROLE}" != h100 &&
      ( -n "${GSVA_LLAMA7B_BASE:-}" || -f "${GSVA_MERGED}/.freeref_merge_complete" ) ]]; then
  MLLM_SEG_ROOT="${ROOT}" bash "${SCRIPT_DIR}/prepare_gsva_llava_legacy.sh"
else
  echo "GSVA merge deferred: set GSVA_LLAMA7B_BASE to your authorized LLaMA-7B-HF path."
fi
echo "UNINEXT runtime deferred until its official ConvNeXt-L Stage-2 checkpoint is obtainable."
echo "Remaining-six runnable environments are prepared for role=${ROLE}."
