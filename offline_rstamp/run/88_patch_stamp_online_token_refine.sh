#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [[ -f "${ROOT_DIR}/offline_rstamp/paths.local.sh" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/offline_rstamp/paths.local.sh"
else
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/offline_rstamp/paths.example.sh"
fi

MLLM_SEG_ROOT="${MLLM_SEG_ROOT:-$(cd "${ROOT_DIR}/.." && pwd)}"
STAMP_CODE_DIR="${STAMP_CODE_DIR:-${MLLM_SEG_ROOT}/code/STAMP}"

python "${ROOT_DIR}/offline_rstamp/scripts/patch_stamp_online_token_refine.py" \
  --tool-repo "${ROOT_DIR}" \
  --stamp-code-dir "${STAMP_CODE_DIR}"

echo "Patched STAMP online token refinement."
echo "Enable during STAMP training with:"
echo "  STAMP_ONLINE_TOKEN_REFINE=1"
echo "  STAMP_FREEZE_FOR_ONLINE_REFINE=1"
echo "  STAMP_REFINE_CALIBRATE=1"
