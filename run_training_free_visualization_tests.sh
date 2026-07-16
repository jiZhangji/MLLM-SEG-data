#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${TRAINING_FREE_REPO:-${SCRIPT_DIR}}"
PYTHON_BIN="${TRAINING_FREE_PYTHON:-$(command -v python)}"
ROOT="${MLLM_SEG_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG}"

cd "${REPO}"
echo "[1/3] Compiling Training-Free visualization code"
"${PYTHON_BIN}" -m compileall -q training_free_refine

echo "[2/3] Running core and visualization tests"
"${PYTHON_BIN}" -m unittest \
  tests.test_training_free_refine \
  tests.test_training_free_visualization

echo "[3/3] Checking shell entry points"
bash -n \
  run_training_free_visualizations.sh \
  run_training_free_visualization_tests.sh

if [[ "${VISUALIZATION_DATA_SMOKE:-0}" == "1" ]]; then
  echo "Running a two-sample smoke test on existing STAMP/Text4Seg result files"
  TRAINING_FREE_VIS_LIMIT=2 \
  TRAINING_FREE_VIS_PANELS_PER_GROUP=1 \
  TRAINING_FREE_VIS_OUTPUT="${TRAINING_FREE_VIS_OUTPUT:-${ROOT}/outputs/training_free_visualizations_smoke}" \
    bash run_training_free_visualizations.sh
fi

echo "Training-Free visualization tests passed."
