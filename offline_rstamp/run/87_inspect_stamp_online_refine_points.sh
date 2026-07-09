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

OUTPUT_DIR="${OUTPUT_DIR:-${MLLM_SEG_ROOT}/outputs/online_token_refine_inspection}"

python "${ROOT_DIR}/offline_rstamp/scripts/inspect_stamp_online_refine_points.py" \
  --stamp-code-dir "${STAMP_CODE_DIR}" \
  --output-dir "${OUTPUT_DIR}"

echo "Inspection files:"
echo "  ${OUTPUT_DIR}/online_refine_inspection.json"
find "${OUTPUT_DIR}" -maxdepth 1 -name "*.snippets.txt" -print | sort
