#!/usr/bin/env bash
set -euo pipefail

STAMP_ENV_PATH="${STAMP_ENV_PATH:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/my_global_cache/conda/envs/STAMP}"
TEXT4SEG_CONDA_ENV="${TEXT4SEG_CONDA_ENV:-text4seg-tf}"

install_densecrf() {
  local label="$1"
  shift
  if "$@" -c 'import pydensecrf.densecrf, pydensecrf.utils' >/dev/null 2>&1; then
    echo "${label}: pydensecrf already available"
    return
  fi
  echo "${label}: installing pydensecrf build dependencies"
  "$@" -m pip install 'Cython<3' setuptools wheel
  "$@" -m pip install --no-build-isolation \
    'git+https://github.com/lucasb-eyer/pydensecrf.git'
  "$@" -c 'import pydensecrf.densecrf, pydensecrf.utils'
  echo "${label}: pydensecrf OK"
}

install_densecrf STAMP "${STAMP_ENV_PATH}/bin/python"
install_densecrf Text4Seg conda run --no-capture-output -n "${TEXT4SEG_CONDA_ENV}" python

if ! "${STAMP_ENV_PATH}/bin/python" -c 'import cupy, cucim' >/dev/null 2>&1; then
  echo "ERROR: GPU FreeRef dependencies are missing from the STAMP environment." >&2
  echo "Run prepare_freeref_efficiency_env.sh for that environment, then retry." >&2
  exit 1
fi
if ! conda run --no-capture-output -n "${TEXT4SEG_CONDA_ENV}" \
  python -c 'import cupy, cucim' >/dev/null 2>&1; then
  echo "ERROR: GPU FreeRef dependencies are missing from the Text4Seg environment." >&2
  echo "Run prepare_freeref_efficiency_env.sh for that environment, then retry." >&2
  exit 1
fi

echo "FreeRef post-processing baseline environments are ready."
