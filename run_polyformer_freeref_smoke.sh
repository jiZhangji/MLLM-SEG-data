#!/usr/bin/env bash
set -euo pipefail

ROOT="${MLLM_SEG_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
POLYFORMER_DIR="${POLYFORMER_DIR:-${ROOT}/code/third_party/polyformer}"
export PYTHONPATH="${POLYFORMER_DIR}/fairseq:${POLYFORMER_DIR}:${SCRIPT_DIR}:${PYTHONPATH:-}"
WEIGHTS_ROOT="${POLYFORMER_WEIGHTS_ROOT:-${ROOT}/models/freeref_missing_methods}"
CHECKPOINT="${POLYFORMER_CHECKPOINT:-${WEIGHTS_ROOT}/polyformer/polyformer_l_refcoco.pt}"
BERT_DIR="${POLYFORMER_BERT_DIR:-${WEIGHTS_ROOT}/shared/bert-base-uncased}"
REFER_ROOT="${POLYFORMER_REFER_ROOT:-${ROOT}/data/lisa_paper_refer_seg}"
IMAGE_ROOT="${POLYFORMER_IMAGE_ROOT:-${REFER_ROOT}/images/mscoco/images/train2014}"
CONDA_ENV="${POLYFORMER_CONDA_ENV:-polyformer-freeref}"
REFINE_ENV="${FREEREF_CONDA_ENV:-STAMP}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
DATASET="${POLYFORMER_DATASET:-refcoco}"
SPLIT_BY="${POLYFORMER_SPLIT_BY:-}"
SPLIT="${POLYFORMER_SPLIT:-testA}"
LIMIT="${POLYFORMER_LIMIT:-64}"
OFFSET="${POLYFORMER_OFFSET:-0}"
BATCH_SIZE="${POLYFORMER_BATCH_SIZE:-8}"
MIN_FREE_MB="${POLYFORMER_MIN_FREE_MB:-30000}"
if [[ -z "${SPLIT_BY}" ]]; then
  if [[ "${DATASET}" == refcocog ]]; then SPLIT_BY=umd; else SPLIT_BY=unc; fi
fi
if [[ -z "${POLYFORMER_CHECKPOINT:-}" ]]; then
  case "${DATASET}" in
    refcoco) CHECKPOINT="${WEIGHTS_ROOT}/polyformer/polyformer_l_refcoco.pt" ;;
    refcoco+) CHECKPOINT="${WEIGHTS_ROOT}/polyformer/polyformer_l_refcoco+.pt" ;;
    refcocog) CHECKPOINT="${WEIGHTS_ROOT}/polyformer/polyformer_l_refcocog.pt" ;;
    *) echo "ERROR: unsupported PolyFormer dataset: ${DATASET}" >&2; exit 2 ;;
  esac
fi
SAFE_DATASET="${DATASET//+/plus}"
RUN_NAME="${SAFE_DATASET}_${SPLIT}"
OUTPUT_ROOT="${POLYFORMER_OUTPUT_ROOT:-${ROOT}/outputs/polyformer_freeref_smoke_n${LIMIT}_o${OFFSET}}"
PAPER_MIOU="${POLYFORMER_PAPER_MIOU:-}"
if [[ -z "${PAPER_MIOU}" ]]; then
  case "${DATASET}_${SPLIT}" in
    refcoco_val) PAPER_MIOU=76.94 ;;
    refcoco_testA) PAPER_MIOU=78.49 ;;
    refcoco_testB) PAPER_MIOU=74.83 ;;
    refcoco+_val) PAPER_MIOU=72.15 ;;
    refcoco+_testA) PAPER_MIOU=75.71 ;;
    refcoco+_testB) PAPER_MIOU=66.73 ;;
    refcocog_val) PAPER_MIOU=71.15 ;;
    refcocog_test) PAPER_MIOU=71.17 ;;
  esac
fi

for required in \
  "${POLYFORMER_DIR}/evaluate.py" \
  "${CHECKPOINT}" \
  "${BERT_DIR}/vocab.txt" \
  "${REFER_ROOT}/${DATASET}/instances.json" \
  "${REFER_ROOT}/${DATASET}/refs(${SPLIT_BY}).p"; do
  if [[ ! -f "${required}" ]]; then
    echo "ERROR: PolyFormer prerequisite is missing: ${required}" >&2
    exit 1
  fi
done
if [[ ! -d "${IMAGE_ROOT}" ]]; then
  echo "ERROR: COCO train2014 directory is missing: ${IMAGE_ROOT}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}/data" "${OUTPUT_ROOT}/official" "${OUTPUT_ROOT}/freeref"
TSV="${OUTPUT_ROOT}/data/${RUN_NAME}.tsv"
BASE_DIR="${OUTPUT_ROOT}/official"
REFINE_DIR="${OUTPUT_ROOT}/freeref"

echo "PolyFormer-L + FreeRef paired evaluation"
echo "dataset=${DATASET} split_by=${SPLIT_BY} split=${SPLIT} limit=${LIMIT} offset=${OFFSET} gpu=${CUDA_DEVICE}"
echo "output=${OUTPUT_ROOT}"

CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" conda run --no-capture-output -n "${CONDA_ENV}" \
  python -m universal_freeref.prepare_polyformer_eval_data \
  --polyformer-code-dir "${POLYFORMER_DIR}" \
  --refer-root "${REFER_ROOT}" \
  --image-root "${IMAGE_ROOT}" \
  --dataset "${DATASET}" --split-by "${SPLIT_BY}" --split "${SPLIT}" \
  --output "${TSV}" --limit "${LIMIT}" --offset "${OFFSET}"

while true; do
  FREE_MB="$(nvidia-smi -i "${CUDA_DEVICE}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -dc '0-9')"
  if [[ -n "${FREE_MB}" && "${FREE_MB}" -ge "${MIN_FREE_MB}" ]]; then
    break
  fi
  echo "GPU ${CUDA_DEVICE} free: ${FREE_MB:-unknown} MiB; waiting 10 seconds..."
  sleep 10
done

MODEL_OVERRIDES="{\"data\":\"${TSV}\",\"bpe_dir\":\"${POLYFORMER_DIR}/utils/BPE\",\"selected_cols\":\"0,5,6,2,4,3\"}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" conda run --no-capture-output -n "${CONDA_ENV}" \
  python -m universal_freeref.export_polyformer_masks \
  --freeref-polyformer-code-dir "${POLYFORMER_DIR}" \
  --freeref-output-dir "${BASE_DIR}" \
  --freeref-tsv "${TSV}" \
  --freeref-bert-dir "${BERT_DIR}" \
  --freeref-method PolyFormer-L-official \
  --freeref-split "${DATASET}_${SPLIT}" \
  -- \
  "${TSV}" \
  --path "${CHECKPOINT}" \
  --user-dir "${POLYFORMER_DIR}/polyformer_module" \
  --task refcoco \
  --batch-size "${BATCH_SIZE}" \
  --log-format simple --log-interval 10 \
  --seed 7 --gen-subset "${DATASET}_${SPLIT}" \
  --results-path "${BASE_DIR}/official_metrics" \
  --result_dir "${BASE_DIR}/official_batches" \
  --vis_dir "${BASE_DIR}/visualizations" \
  --no-repeat-ngram-size 3 --fp16 --num-workers 0 --num-bins 64 \
  --distributed-world-size 1 \
  --model-overrides "${MODEL_OVERRIDES}"

conda run --no-capture-output -n "${REFINE_ENV}" python -m universal_freeref.evaluate \
  --manifest "${BASE_DIR}/manifest.jsonl" \
  --output-dir "${REFINE_DIR}" \
  --save-visualizations 8

SUMMARY_ARGS=(
  --export-summary "${BASE_DIR}/export_summary.json"
  --freeref-summary "${REFINE_DIR}/eval_summary.json"
  --output "${OUTPUT_ROOT}/comparison.md"
)
[[ -n "${PAPER_MIOU}" ]] && SUMMARY_ARGS+=(--paper-miou "${PAPER_MIOU}")
conda run --no-capture-output -n "${REFINE_ENV}" python -m universal_freeref.summarize_polyformer \
  "${SUMMARY_ARGS[@]}"

echo "PolyFormer evaluation completed: ${OUTPUT_ROOT}/comparison.md"
