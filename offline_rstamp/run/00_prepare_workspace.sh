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
  echo "WARNING: using paths.example.sh. Copy it to paths.local.sh and edit paths first." >&2
fi

python "${ROOT_DIR}/offline_rstamp/scripts/check_data_layout.py" \
  --data-root "${MLLM_SEG_DATA_ROOT}" \
  --report "${MLLM_SEG_WORKSPACE}/data_layout_report.json"

python "${ROOT_DIR}/offline_rstamp/scripts/prepare_workspace.py" \
  --data-root "${MLLM_SEG_DATA_ROOT}" \
  --workspace "${MLLM_SEG_WORKSPACE}"

echo "Workspace prepared: ${MLLM_SEG_WORKSPACE}"
echo "Next: copy STAMP code into ${STAMP_CODE_DIR}"

