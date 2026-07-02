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

python "${ROOT_DIR}/offline_rstamp/scripts/install_rstamp_scaffold.py" \
  --target-code-dir "${RSTAMP_CODE_DIR}" \
  --overwrite

echo "R-STAMP scaffold installed to: ${RSTAMP_CODE_DIR}"
echo "Next: integrate rstamp modules into STAMP model/trainer."

