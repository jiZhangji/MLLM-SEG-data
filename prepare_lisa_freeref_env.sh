#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LISA_SETUP_ENV=1 LISA_SETUP_ONLY=1 \
  bash "${SCRIPT_DIR}/run_lisa_freeref_eval.sh"
