#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
STAMP_ENV_PATH="${STAMP_ENV_PATH:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/my_global_cache/conda/envs/STAMP}"
TEXT4SEG_CONDA_ENV="${TEXT4SEG_CONDA_ENV:-text4seg-tf}"
LISA_CONDA_ENV="${LISA_CONDA_ENV:-lisa-freeref}"

if [[ ! -x "${STAMP_ENV_PATH}/bin/python" ]]; then
  echo "ERROR: STAMP environment is missing: ${STAMP_ENV_PATH}" >&2
  exit 1
fi

install_gpu_freeref() {
  local label="$1"
  shift
  echo "Installing GPU FreeRef runtime in ${label}"
  "$@" -m pip install \
    'numpy<2' \
    'cupy-cuda12x>=13.3,<14' \
    'cucim-cu12>=24.12,<26' \
    'nvidia-ml-py>=12'
  "$@" -c \
    'import cupy as cp; from cucim.skimage.color import rgb2lab; from cucim.core.operations.morphology import distance_transform_edt; from cupyx.scipy.sparse.linalg import cg; import pynvml; print("GPU FreeRef runtime OK", cp.__version__)'
  CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}" "$@" -c \
    'import torch; from training_free_refine import GpuTrainingFreeUncertaintyRefiner, TrainingFreeRefineConfig; image=torch.randint(0,256,(32,32,3),device="cuda",dtype=torch.uint8); mask=torch.zeros((32,32),device="cuda"); mask[8:24,8:24]=1; out=GpuTrainingFreeUncertaintyRefiner(TrainingFreeRefineConfig(n_segments=16)).refine_hard_mask(image,mask); torch.cuda.synchronize(); assert out["refined_mask"].is_cuda and out["refined_mask"].shape==(32,32); print("GPU FreeRef end-to-end preflight OK")'
}

install_gpu_freeref "STAMP" "${STAMP_ENV_PATH}/bin/python"
install_gpu_freeref "Text4Seg" conda run --no-capture-output -n "${TEXT4SEG_CONDA_ENV}" python

echo "Installing NVML measurement support in ${LISA_CONDA_ENV}"
conda run --no-capture-output -n "${LISA_CONDA_ENV}" \
  python -m pip install 'nvidia-ml-py>=12'
conda run --no-capture-output -n "${LISA_CONDA_ENV}" \
  python -c 'import pynvml; print("LISA NVML runtime OK")'

echo "Efficiency benchmark environments are ready."
