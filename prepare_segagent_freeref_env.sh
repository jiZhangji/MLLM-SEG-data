#!/usr/bin/env bash
set -euo pipefail

SOURCE_ENV="${SEGAGENT_SOURCE_CONDA_ENV:-lisa-freeref}"
TARGET_ENV="${SEGAGENT_CONDA_ENV:-segagent-freeref}"

has_env() {
  TARGET_NAME="$1" conda env list --json | python -c \
    'import json, os, pathlib, sys; data=json.load(sys.stdin); name=os.environ["TARGET_NAME"]; raise SystemExit(0 if any(pathlib.Path(path).name == name for path in data.get("envs", [])) else 1)'
}

env_prefix() {
  TARGET_NAME="$1" conda info --json | python -c \
    'import json, os, pathlib, sys; data=json.load(sys.stdin); print(pathlib.Path(data["envs_dirs"][0]) / os.environ["TARGET_NAME"])'
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

IMPORT_CHECK='import albumentations, cv2, easydict, mmcv, segment_anything, timm, tiktoken, torch; assert torch.cuda.is_available()'
if ! PYTHONNOUSERSITE=1 conda run -n "${TARGET_ENV}" python -c "${IMPORT_CHECK}" >/dev/null 2>&1; then
  echo "Installing the SegAgent/SimpleClick runtime into ${TARGET_ENV}"
  PYTHONNOUSERSITE=1 conda run --no-capture-output -n "${TARGET_ENV}" \
    python -m pip install \
      numpy==1.26.4 \
      opencv-python-headless==4.10.0.84 \
      albumentations==0.5.2 \
      easydict==1.9 \
      addict==2.4.0 \
      imgaug==0.4.0 \
      mmcv==1.6.2 \
      timm==0.6.13 \
      segment-anything==1.0 \
      tiktoken==0.7.0 \
      transformers-stream-generator==0.0.5 \
      yapf==0.40.2
fi

PYTHONNOUSERSITE=1 conda run --no-capture-output -n "${TARGET_ENV}" python -c \
  'import albumentations, cv2, easydict, mmcv, segment_anything, timm, tiktoken, torch; print("SegAgent environment ready:", torch.__version__, torch.version.cuda, cv2.__version__)'
