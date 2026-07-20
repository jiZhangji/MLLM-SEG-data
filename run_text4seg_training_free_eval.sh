#!/usr/bin/env bash
set -euo pipefail

ROOT="${MLLM_SEG_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${TRAINING_FREE_REPO:-${SCRIPT_DIR}}"
TEXT4SEG_DIR="${TEXT4SEG_DIR:-${ROOT}/code/Text4Seg}"
CONDA_ENV="${TEXT4SEG_CONDA_ENV:-text4seg-tf}"
MODEL_PATH="${TEXT4SEG_MODEL_PATH:-lmc22/text4seg-llava-7b-p24}"
VISION_TOWER="${TEXT4SEG_VISION_TOWER:-openai/clip-vit-large-patch14-336}"
DESCRIPTOR_GRID_SIZE="${TEXT4SEG_DESCRIPTOR_GRID_SIZE:-${TEXT4SEG_VISUAL_TOKENS:-}}"
if [[ -z "${DESCRIPTOR_GRID_SIZE}" ]]; then
  case "${MODEL_PATH,,}" in
    *p16*) DESCRIPTOR_GRID_SIZE=16 ;;
    *p24*) DESCRIPTOR_GRID_SIZE=24 ;;
    *p32*) DESCRIPTOR_GRID_SIZE=32 ;;
    *)
      echo "ERROR: cannot infer the Text4Seg descriptor grid from MODEL_PATH=${MODEL_PATH}." >&2
      echo "Set TEXT4SEG_DESCRIPTOR_GRID_SIZE to 16, 24, or 32 explicitly." >&2
      exit 1
      ;;
  esac
fi
EVAL_JSON="${TEXT4SEG_EVAL_JSON:-${ROOT}/code/STAMP/playground/data/json_eval_baseline/refcocog_val.json}"
EVAL_LIMIT="${TEXT4SEG_EVAL_LIMIT:-0}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
RESULTS_ROOT="${TEXT4SEG_RESULTS_ROOT:-${ROOT}/outputs/text4seg_official_refcocog_val}"
REFINE_OUTPUT="${TEXT4SEG_REFINE_OUTPUT:-${ROOT}/outputs/text4seg_training_free_refcocog_val}"
DEFAULT_SAM_PATH="${ROOT}/models/sam_vit_h_4b8939.pth"
if [[ -f "${ROOT}/models/SAM/sam_vit_h_4b8939.pth" ]]; then
  DEFAULT_SAM_PATH="${ROOT}/models/SAM/sam_vit_h_4b8939.pth"
fi
SAM_PATH="${TEXT4SEG_SAM_PATH:-${DEFAULT_SAM_PATH}}"

export HF_HOME="${HF_HOME:-${ROOT}/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "${ROOT}/code" "${ROOT}/models" "${ROOT}/outputs" "${HF_HOME}" "$(dirname "${SAM_PATH}")"

if command -v flock >/dev/null 2>&1; then
  exec 9>"${ROOT}/outputs/.text4seg_training_free.lock"
  if ! flock -n 9; then
    echo "Another Text4Seg training-free job already holds the lock." >&2
    exit 0
  fi
fi

echo "Text4Seg checkpoint: ${MODEL_PATH}"
echo "Semantic-descriptor grid: p${DESCRIPTOR_GRID_SIZE} (${DESCRIPTOR_GRID_SIZE}x${DESCRIPTOR_GRID_SIZE})"
echo "Evaluation protocol: paired flat JSON (not the official REFER loader)"

echo "[1/7] Fetching official Text4Seg code"
if [[ ! -d "${TEXT4SEG_DIR}/.git" ]]; then
  GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 --filter=blob:none \
    https://github.com/mc-lan/Text4Seg.git "${TEXT4SEG_DIR}"
else
  echo "Using existing Text4Seg checkout: ${TEXT4SEG_DIR}"
fi

echo "[2/7] Creating an H200-compatible Text4Seg environment"
if ! conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV}"; then
  conda create -y -n "${CONDA_ENV}" python=3.10 pip
fi

if ! conda run -n "${CONDA_ENV}" python -c \
  'import torch, transformers, pycocotools, skimage; assert torch.__version__.startswith("2.6."); assert transformers.__version__.startswith("4.37.")' \
  >/dev/null 2>&1; then
  conda run --no-capture-output -n "${CONDA_ENV}" python -m pip install --upgrade pip wheel setuptools
  conda run --no-capture-output -n "${CONDA_ENV}" python -m pip install \
    torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
  conda run --no-capture-output -n "${CONDA_ENV}" python -m pip install \
    'numpy<2' transformers==4.37.2 tokenizers==0.15.1 sentencepiece==0.1.99 \
    accelerate==0.27.2 peft==0.8.2 bitsandbytes shortuuid safetensors \
    scipy scikit-image scikit-learn pillow opencv-python-headless pycocotools \
    matplotlib tqdm einops==0.6.1 einops-exts==0.0.4 timm==0.6.13 protobuf
fi
conda run --no-capture-output -n "${CONDA_ENV}" python -m pip install \
  --no-deps -e "${TEXT4SEG_DIR}"

echo "[3/7] Verifying the paired STAMP/Text4Seg evaluation JSON"
if [[ ! -f "${EVAL_JSON}" ]]; then
  echo "ERROR: evaluation JSON was not found at ${EVAL_JSON}." >&2
  exit 1
fi

echo "[4/7] Resolving the official SAM-H comparison checkpoint: ${SAM_PATH}"
SAM_BYTES="$(stat -c '%s' "${SAM_PATH}" 2>/dev/null || echo 0)"
if (( SAM_BYTES < 2000000000 )); then
  rm -f "${SAM_PATH}"
  if command -v wget >/dev/null 2>&1; then
    wget -c -O "${SAM_PATH}.part" https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
  else
    curl -L --retry 5 -C - -o "${SAM_PATH}.part" https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
  fi
  mv "${SAM_PATH}.part" "${SAM_PATH}"
fi

echo "[5/7] Verifying CUDA and integration tests"
conda run --no-capture-output -n "${CONDA_ENV}" python -c \
  'import torch; assert torch.cuda.is_available(); print("torch", torch.__version__, "cuda", torch.version.cuda, torch.cuda.get_device_name(0))'
PYTHONPATH="${REPO}:${PYTHONPATH:-}" conda run --no-capture-output -n "${CONDA_ENV}" \
  python -m unittest tests.test_training_free_refine

echo "[6/7] Running/resuming Text4Seg on the paired evaluation JSON"
TEXT4SEG_COMPLETED=0
if [[ -d "${RESULTS_ROOT}/gt_masks" ]]; then
  TEXT4SEG_COMPLETED="$(find "${RESULTS_ROOT}/gt_masks" -maxdepth 1 -type f -name '*.png' | wc -l)"
fi
TEXT4SEG_EXPECTED="$(conda run -n "${CONDA_ENV}" python -c 'import json,sys; n=len(json.load(open(sys.argv[1], encoding="utf-8"))); limit=int(sys.argv[2]); print(min(n, limit) if limit else n)' "${EVAL_JSON}" "${EVAL_LIMIT}")"
echo "Existing complete Text4Seg samples: ${TEXT4SEG_COMPLETED} / ${TEXT4SEG_EXPECTED}"
cd "${REPO}"
CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" PYTHONPATH="${REPO}:${PYTHONPATH:-}" \
  conda run --no-capture-output -n "${CONDA_ENV}" \
  python -m training_free_refine.export_text4seg_masks \
    --text4seg-code-dir "${TEXT4SEG_DIR}" \
    --model-path "${MODEL_PATH}" \
    --vision-tower "${VISION_TOWER}" \
    --eval-json "${EVAL_JSON}" \
    --data-root "${ROOT}" \
    --sam-path "${SAM_PATH}" \
    --output-dir "${RESULTS_ROOT}" \
    --descriptor-grid-size "${DESCRIPTOR_GRID_SIZE}" \
    --limit "${EVAL_LIMIT}" \
    --seed 0

echo "[7/7] Evaluating coarse, training-free refined, and SAM-H masks"
cd "${REPO}"
PYTHONPATH="${REPO}:${PYTHONPATH:-}" conda run --no-capture-output -n "${CONDA_ENV}" \
  python -m training_free_refine.eval_text4seg_outputs \
    --manifest "${RESULTS_ROOT}/manifest.jsonl" \
    --output-dir "${REFINE_OUTPUT}" \
    --n-segments 1024 \
    --graph-lambda 1.0 \
    --boundary-sigma 8.0 \
    --save-visualizations 8

echo "Text4Seg training-free evaluation completed."
echo "Summary: ${REFINE_OUTPUT}/eval_summary.json"
echo "Rows: ${REFINE_OUTPUT}/eval_rows.csv"
