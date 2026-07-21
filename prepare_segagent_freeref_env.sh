#!/usr/bin/env bash
set -euo pipefail

ROOT="${MLLM_SEG_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG}"
SEGAGENT_DIR="${SEGAGENT_DIR:-${ROOT}/code/third_party/segagent}"
SOURCE_ENV="${SEGAGENT_SOURCE_CONDA_ENV:-lisa-freeref}"
TARGET_ENV="${SEGAGENT_CONDA_ENV:-segagent-freeref}"

has_env() {
  conda env list --json | python -c \
    'import json, pathlib, sys; name=sys.argv[1]; data=json.load(sys.stdin); raise SystemExit(0 if any(pathlib.Path(path).name == name for path in data.get("envs", [])) else 1)' \
    "$1"
}

env_prefix() {
  conda info --json | python -c \
    'import json, pathlib, sys; name=sys.argv[1]; data=json.load(sys.stdin); print(pathlib.Path(data["envs_dirs"][0]) / name)' \
    "$1"
}

if ! has_env "${SOURCE_ENV}"; then
  echo "ERROR: source conda environment is missing: ${SOURCE_ENV}" >&2
  exit 1
fi

if ! has_env "${TARGET_ENV}"; then
  TARGET_PREFIX="$(env_prefix "${TARGET_ENV}")"
  if [[ -d "${TARGET_PREFIX}" ]]; then
    backup="${TARGET_PREFIX}.incomplete.$(date +%Y%m%d-%H%M%S)"
    echo "Moving incomplete conda directory to ${backup}"
    mv "${TARGET_PREFIX}" "${backup}"
  fi
  echo "Cloning ${SOURCE_ENV} -> ${TARGET_ENV}"
  conda create -y -n "${TARGET_ENV}" --clone "${SOURCE_ENV}"
fi

IMPORT_CHECK='import albumentations, cv2, easydict, mmcv, segment_anything, tensorboard, timm, tiktoken, torch; assert torch.cuda.is_available()'
if ! PYTHONNOUSERSITE=1 conda run -n "${TARGET_ENV}" python -c "${IMPORT_CHECK}" >/dev/null 2>&1; then
  echo "Installing the SegAgent/SimpleClick runtime into ${TARGET_ENV}"
  PYTHONNOUSERSITE=1 conda run --no-capture-output -n "${TARGET_ENV}" \
    python -m pip install setuptools==80.9.0 wheel
  PYTHONNOUSERSITE=1 conda run --no-capture-output -n "${TARGET_ENV}" \
    python -m pip install \
      numpy==1.26.4 \
      opencv-python-headless==4.10.0.84 \
      albumentations==0.5.2 \
      easydict==1.9 \
      addict==2.4.0 \
      imgaug==0.4.0 \
      timm==0.6.13 \
      segment-anything==1.0 \
      tensorboard==2.17.0 \
      protobuf==4.25.4 \
      tiktoken==0.7.0 \
      transformers-stream-generator==0.0.5 \
      yapf==0.40.2
  # MMCV 1.6.2 imports pkg_resources from its setup.py. New isolated build
  # environments omit it, so build against the pinned setuptools above.
  PYTHONNOUSERSITE=1 conda run --no-capture-output -n "${TARGET_ENV}" \
    python -m pip install --no-build-isolation mmcv==1.6.2
fi

PYTHONNOUSERSITE=1 conda run --no-capture-output -n "${TARGET_ENV}" python -c \
  'import albumentations, cv2, easydict, mmcv, segment_anything, tensorboard, timm, tiktoken, torch; print("SegAgent environment ready:", torch.__version__, torch.version.cuda, cv2.__version__)'

if [[ ! -f "${SEGAGENT_DIR}/evaltools/model_loader.py" ]]; then
  echo "ERROR: SegAgent model loader is missing: ${SEGAGENT_DIR}/evaltools/model_loader.py" >&2
  exit 1
fi
echo "Checking the official SegAgent model-loader import"
(
  cd "${SEGAGENT_DIR}/evaltools"
  PYTHONPATH="${SEGAGENT_DIR}:${SEGAGENT_DIR}/evaltools:${PYTHONPATH:-}" PYTHONNOUSERSITE=1 \
    conda run --no-capture-output -n "${TARGET_ENV}" python -c \
      'from model_loader import load_model; print("SegAgent official model_loader import: OK")'
)
