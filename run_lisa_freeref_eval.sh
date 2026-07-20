#!/usr/bin/env bash
set -euo pipefail

ROOT="${MLLM_SEG_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${FREEREF_REPO:-${SCRIPT_DIR}}"
LISA_DIR="${LISA_DIR:-${ROOT}/code/third_party/lisa}"
CONDA_ENV="${LISA_CONDA_ENV:-lisa-freeref}"
WEIGHTS_ROOT="${FREEREF_WEIGHTS_ROOT:-${ROOT}/models/freeref_missing_methods}"
MODEL_PATH="${LISA_MODEL_PATH:-${WEIGHTS_ROOT}/lisa/LISA-7B-v1}"
VISION_TOWER="${LISA_VISION_TOWER:-${WEIGHTS_ROOT}/shared/clip-vit-large-patch14}"
EVAL_JSON_DIR="${LISA_EVAL_JSON_DIR:-${ROOT}/code/STAMP/playground/data/json_eval_baseline}"
SPLITS="${LISA_SPLITS:-refcoco_testA}"
EVAL_LIMIT="${LISA_EVAL_LIMIT:-0}"
EVAL_OFFSET="${LISA_EVAL_OFFSET:-0}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
PRECISION="${LISA_PRECISION:-bf16}"
MAX_EXPRESSIONS_PER_CALL="${LISA_MAX_EXPRESSIONS_PER_CALL:-0}"
RESULTS_ROOT="${LISA_RESULTS_ROOT:-${ROOT}/outputs/lisa_official}"
REFINE_ROOT="${LISA_FREEREF_ROOT:-${ROOT}/outputs/universal_freeref_lisa}"
SETUP_ENV="${LISA_SETUP_ENV:-1}"

export HF_HOME="${HF_HOME:-${ROOT}/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "${ROOT}/code/third_party" "${ROOT}/outputs" "${RESULTS_ROOT}" "${REFINE_ROOT}"

echo "WARNING: this runner uses STAMP flat JSON for paired analysis; it does not reproduce the LISA paper protocol."
echo "Use run_lisa_paper_reproduction.sh and pass its reproduction gate before paper comparison."

if command -v flock >/dev/null 2>&1; then
  exec 9>"${ROOT}/outputs/.lisa_freeref.lock"
  if ! flock -n 9; then
    echo "Another LISA -> FreeRef job already holds ${ROOT}/outputs/.lisa_freeref.lock." >&2
    exit 0
  fi
fi

echo "[1/6] Checking the official LISA source"
if [[ ! -f "${LISA_DIR}/model/LISA.py" ]]; then
  git clone --depth 1 https://github.com/JIA-Lab-research/LISA.git "${LISA_DIR}"
else
  echo "Using existing LISA checkout: ${LISA_DIR}"
fi

echo "[2/6] Preparing the H200/RTX 4090 compatible LISA environment"
if [[ "${SETUP_ENV}" == "1" ]]; then
  if ! conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV}"; then
    conda create -y -n "${CONDA_ENV}" python=3.10 pip
  fi
  if ! conda run -n "${CONDA_ENV}" python -c \
    'import torch, transformers, peft, cv2, pycocotools; assert torch.__version__.startswith("2.6."); assert transformers.__version__ == "4.31.0"' \
    >/dev/null 2>&1; then
    conda run --no-capture-output -n "${CONDA_ENV}" python -m pip install --upgrade pip wheel setuptools
    conda run --no-capture-output -n "${CONDA_ENV}" python -m pip install \
      torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
    conda run --no-capture-output -n "${CONDA_ENV}" python -m pip install \
      numpy==1.26.4 transformers==4.31.0 tokenizers==0.13.3 \
      huggingface-hub==0.17.3 accelerate==0.21.0 peft==0.4.0 \
      sentencepiece==0.1.99 einops==0.4.1 opencv-python-headless==4.8.0.76 \
      Pillow==10.4.0 pycocotools==2.0.7 scipy==1.11.4 tqdm==4.67.1 \
      safetensors protobuf scikit-image scikit-learn matplotlib
  fi
fi

echo "[3/6] Checking local checkpoints and CUDA"
for required in \
  "${MODEL_PATH}/config.json" \
  "${VISION_TOWER}/config.json"; do
  if [[ ! -f "${required}" ]]; then
    echo "ERROR: required local model file is missing: ${required}" >&2
    exit 1
  fi
done
if ! find "${MODEL_PATH}" -maxdepth 1 -type f \( -name '*.bin' -o -name '*.safetensors' \) -size +100M -print -quit | grep -q .; then
  echo "ERROR: no complete LISA weight shard was found below ${MODEL_PATH}." >&2
  exit 1
fi
CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" conda run --no-capture-output -n "${CONDA_ENV}" python -c \
  'import torch; assert torch.cuda.is_available(); print("torch", torch.__version__, "cuda", torch.version.cuda, torch.cuda.get_device_name(0))'

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

summaries=()
echo "[4/6] Running/resuming official LISA mask export"
for split in ${SPLITS}; do
  eval_json="${EVAL_JSON_DIR}/${split}.json"
  safe_split="${split//+/plus}"
  output_dir="${RESULTS_ROOT}/${safe_split}"
  refine_dir="${REFINE_ROOT}/${safe_split}"
  if [[ ! -f "${eval_json}" ]]; then
    echo "ERROR: evaluation JSON is missing: ${eval_json}" >&2
    exit 1
  fi
  echo "RUN LISA ${split}: ${eval_json}"
  cd "${REPO}"
  CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" PYTHONPATH="${REPO}:${PYTHONPATH:-}" \
    conda run --no-capture-output -n "${CONDA_ENV}" \
    python -m universal_freeref.export_lisa_masks \
      --lisa-code-dir "${LISA_DIR}" \
      --model-path "${MODEL_PATH}" \
      --vision-tower "${VISION_TOWER}" \
      --eval-json "${eval_json}" \
      --data-root "${ROOT}" \
      --output-dir "${output_dir}" \
      --split "${split}" \
      --precision "${PRECISION}" \
      --max-expressions-per-call "${MAX_EXPRESSIONS_PER_CALL}" \
      --limit "${EVAL_LIMIT}" \
      --offset "${EVAL_OFFSET}" \
      --seed 0

  echo "[5/6] Applying FreeRef and computing paired metrics for ${split}"
  PYTHONPATH="${REPO}:${PYTHONPATH:-}" conda run --no-capture-output -n "${CONDA_ENV}" \
    python -m universal_freeref.evaluate \
      --manifest "${output_dir}/manifest.jsonl" \
      --output-dir "${refine_dir}" \
      --n-segments 1024 \
      --graph-lambda 1.0 \
      --boundary-sigma 8.0 \
      --save-visualizations 12
  summaries+=(--summary "LISA-7B-v1_${split}=${refine_dir}/eval_summary.json")
done

echo "[6/6] Combining LISA before/after results"
cd "${REPO}"
PYTHONPATH="${REPO}:${PYTHONPATH:-}" conda run --no-capture-output -n "${CONDA_ENV}" \
  python -m universal_freeref.summarize \
    "${summaries[@]}" \
    --output-dir "${REFINE_ROOT}/combined" \
    --title "LISA-7B-v1 Original SAM Mask vs. FreeRef"

echo "LISA -> original prediction -> FreeRef evaluation completed."
echo "Summary: ${REFINE_ROOT}/combined/comparison.md"
