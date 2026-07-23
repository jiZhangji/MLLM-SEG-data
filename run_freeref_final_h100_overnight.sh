#!/usr/bin/env bash
set -euo pipefail

ROOT="${MLLM_SEG_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_ROOT="${FINAL_OUTPUT_ROOT:-${ROOT}/outputs/freeref_final_h100_overnight}"
STAMP_ENV_PATH="${STAMP_ENV_PATH:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/my_global_cache/conda/envs/STAMP}"
TEXT4SEG_CONDA_ENV="${TEXT4SEG_CONDA_ENV:-text4seg-tf}"
CPU_JOBS="${FINAL_ACCURACY_PARALLEL_JOBS:-8}"
MIN_FREE_MB="${FINAL_MIN_FREE_MB:-60000}"
FORCE="${FINAL_FORCE:-0}"
RUN_GREFCOCO="${FINAL_RUN_GREFCOCO:-1}"
PREFLIGHT_ONLY="${FINAL_PREFLIGHT_ONLY:-0}"
TIMING_SAMPLES="${FINAL_TIMING_SAMPLES:-500}"
TIMING_WARMUP="${FINAL_TIMING_WARMUP:-20}"
TIMING_SEED="${FINAL_TIMING_SEED:-0}"
TIMING_K_VALUES_TEXT="${FINAL_TIMING_K_VALUES:-500 1024 2000 4000 8000}"

STAMP_CODE_DIR="${STAMP_CODE_DIR:-${ROOT}/code/STAMP}"
TEXT4SEG_CODE_DIR="${TEXT4SEG_DIR:-${ROOT}/code/Text4Seg}"
STAMP2B_MODEL="${STAMP2B_MODEL_PATH:-${ROOT}/models/STAMP-2B-uni}"
STAMP7B_MODEL="${STAMP7B_MODEL_PATH:-${ROOT}/models/STAMP-7B-lora}"
TEXT4SEG_MODEL="${TEXT4SEG_P24_MODEL_PATH:-lmc22/text4seg-llava-7b-p24}"
VISION_TOWER="${TEXT4SEG_VISION_TOWER:-${ROOT}/models/freeref_missing_methods/shared/clip-vit-large-patch14-336}"
SAM_B_PATH="${SAM_VIT_B_PATH:-${ROOT}/models/SAM/sam_vit_b_01ec64.pth}"
SAM_H_PATH="${SAM_PATH:-${ROOT}/models/SAM/sam_vit_h_4b8939.pth}"
EVAL_JSON="${FINAL_TIMING_EVAL_JSON:-${STAMP_CODE_DIR}/playground/data/json_eval_baseline/refcoco_testA.json}"

if [[ -n "${FREEREF_H100_GPUS:-}" ]]; then
  read -r -a GPUS <<<"${FREEREF_H100_GPUS}"
else
  mapfile -t GPUS < <(
    nvidia-smi --query-gpu=index,name --format=csv,noheader |
      awk -F, 'tolower($2) ~ /h100/ {gsub(/ /,"",$1); print $1}' | head -n 2
  )
fi
if (( ${#GPUS[@]} != 2 )); then
  echo "ERROR: exactly two H100 GPUs are required; detected: ${GPUS[*]:-none}" >&2
  exit 1
fi
GPU0="${GPUS[0]}"
GPU1="${GPUS[1]}"
for gpu in "${GPUS[@]}"; do
  name="$(nvidia-smi -i "${gpu}" --query-gpu=name --format=csv,noheader | head -n1)"
  if [[ "${name,,}" != *h100* ]]; then
    echo "ERROR: physical GPU ${gpu} is ${name}, not H100." >&2
    exit 1
  fi
done
if ! [[ "${CPU_JOBS}" =~ ^[1-9][0-9]*$ && "${TIMING_SAMPLES}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: parallel jobs and timing samples must be positive integers." >&2
  exit 1
fi
read -r -a TIMING_K_VALUES <<<"${TIMING_K_VALUES_TEXT}"

STAMP_PYTHON="${STAMP_ENV_PATH}/bin/python"
for path in \
  "${STAMP_PYTHON}" "${STAMP_CODE_DIR}" "${TEXT4SEG_CODE_DIR}" \
  "${STAMP2B_MODEL}" "${STAMP7B_MODEL}" "${VISION_TOWER}" \
  "${SAM_B_PATH}" "${SAM_H_PATH}" "${EVAL_JSON}"; do
  if [[ ! -e "${path}" ]]; then
    echo "ERROR: required input is missing: ${path}" >&2
    exit 1
  fi
done
if (( $(stat -c '%s' "${SAM_B_PATH}") < 300000000 )); then
  echo "ERROR: SAM ViT-B checkpoint appears incomplete: ${SAM_B_PATH}" >&2
  exit 1
fi
if (( $(stat -c '%s' "${SAM_H_PATH}") < 2000000000 )); then
  echo "ERROR: SAM ViT-H checkpoint appears incomplete: ${SAM_H_PATH}" >&2
  exit 1
fi
if ! "${STAMP_PYTHON}" -c \
  'import pydensecrf.densecrf, scipy, skimage' \
  >/dev/null 2>&1; then
  echo "ERROR: STAMP baseline dependencies are incomplete." >&2
  echo "Run bash prepare_freeref_baseline_env.sh before launching this suite." >&2
  exit 1
fi
if ! conda run --no-capture-output -n "${TEXT4SEG_CONDA_ENV}" python -c \
  'import pydensecrf.densecrf, scipy, skimage' \
  >/dev/null 2>&1; then
  echo "ERROR: Text4Seg baseline dependencies are incomplete." >&2
  echo "Run bash prepare_freeref_baseline_env.sh before launching this suite." >&2
  exit 1
fi

export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false STAMP_DISABLE_CUDNN="${STAMP_DISABLE_CUDNN:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}" NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
mkdir -p "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/combined"
exec > >(tee -a "${OUTPUT_ROOT}/overnight.log") 2>&1

if command -v flock >/dev/null 2>&1; then
  exec 9>"${ROOT}/outputs/.freeref_final_h100_overnight.lock"
  if ! flock -n 9; then
    echo "Another final H100 overnight suite is already running."
    exit 0
  fi
fi

declare -a SPLITS=(
  refcoco_val refcoco_testA refcoco_testB
  refcoco+_val refcoco+_testA refcoco+_testB
  refcocog_val refcocog_test
)
declare -A EXPECTED=(
  [refcoco_val]=10834 [refcoco_testA]=5657 [refcoco_testB]=5095
  [refcoco+_val]=10758 [refcoco+_testA]=5726 [refcoco+_testB]=4889
  [refcocog_val]=4896 [refcocog_test]=9602
)

stage() {
  echo "$1" >"${OUTPUT_ROOT}/current_stage.txt"
  printf '\n===== %s | %s =====\n' "$(date -u '+%F %T UTC')" "$1"
}

split_slug() {
  printf '%s' "${1//+/plus}"
}

model_slug() {
  printf '%s' "${1,,}" | tr -d '-'
}

dump_name() {
  local model="$1" split="$2"
  case "${model}|${split}" in
    STAMP-2B\|refcoco_val) echo refcoco_val_full_stamp2b ;;
    STAMP-2B\|refcoco_testA) echo refcoco_testA_full_stamp2b ;;
    STAMP-2B\|refcoco_testB) echo refcoco_testB_full_stamp2b ;;
    STAMP-2B\|refcoco+_val) echo refcocoplus_val_full_stamp2b ;;
    STAMP-2B\|refcoco+_testA) echo refcocoplus_testA_full_stamp2b ;;
    STAMP-2B\|refcoco+_testB) echo refcocoplus_testB_full_stamp2b ;;
    STAMP-2B\|refcocog_val) echo refcocog_val_full ;;
    STAMP-2B\|refcocog_test)
      if [[ -d "${ROOT}/outputs/refine_stamp_dumps/refcocog_test_full_stamp2b" ]]; then
        echo refcocog_test_full_stamp2b
      else
        echo refcocog_test_full
      fi
      ;;
    STAMP-7B\|refcoco_val) echo refcoco_val_full_stamp7b ;;
    STAMP-7B\|refcoco_testA) echo refcoco_testA_full_stamp7b ;;
    STAMP-7B\|refcoco_testB) echo refcoco_testB_full_stamp7b ;;
    STAMP-7B\|refcoco+_val) echo refcocoplus_val_full_stamp7b ;;
    STAMP-7B\|refcoco+_testA) echo refcocoplus_testA_full_stamp7b ;;
    STAMP-7B\|refcoco+_testB) echo refcocoplus_testB_full_stamp7b ;;
    STAMP-7B\|refcocog_val) echo refcocog_val_full_stamp7b ;;
    STAMP-7B\|refcocog_test) echo refcocog_test_full_stamp7b ;;
    *) echo "ERROR: no dump mapping for ${model} ${split}" >&2; return 1 ;;
  esac
}

complete_samples() {
  local summary="$1" expected="$2"
  [[ "${FORCE}" != "1" && -f "${summary}" ]] && \
    python -c 'import json,sys; d=json.load(open(sys.argv[1])); raise SystemExit(0 if int(d.get("samples",0))==int(sys.argv[2]) else 1)' \
      "${summary}" "${expected}"
}

has_samples() {
  local summary="$1" expected="$2"
  [[ -f "${summary}" ]] && python -c \
    'import json,sys; d=json.load(open(sys.argv[1])); raise SystemExit(0 if int(d.get("samples",0))==int(sys.argv[2]) else 1)' \
    "${summary}" "${expected}"
}

complete_sam() {
  local summary="$1" expected="$2" protocol="$3"
  [[ "${FORCE}" != "1" && -f "${summary}" ]] && \
    python -c 'import json,sys; d=json.load(open(sys.argv[1])); ok=int(d.get("samples",0))==int(sys.argv[2]) and d.get("protocol")==sys.argv[3]; raise SystemExit(0 if ok else 1)' \
      "${summary}" "${expected}" "${protocol}"
}

has_sam() {
  local summary="$1" expected="$2" protocol="$3"
  [[ -f "${summary}" ]] && python -c \
    'import json,sys; d=json.load(open(sys.argv[1])); ok=int(d.get("samples",0))==int(sys.argv[2]) and d.get("protocol")==sys.argv[3]; raise SystemExit(0 if ok else 1)' \
    "${summary}" "${expected}" "${protocol}"
}

wait_for_gpu() {
  local gpu="$1" free
  while true; do
    free="$(nvidia-smi -i "${gpu}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -dc '0-9')"
    if [[ -n "${free}" ]] && (( free >= MIN_FREE_MB )); then
      return
    fi
    echo "GPU ${gpu}: ${free:-unknown} MiB free; waiting for ${MIN_FREE_MB} MiB."
    sleep 30
  done
}

run_sam_job() {
  local model="$1" split="$2" sam_type="$3" gpu="$4"
  local expected="${EXPECTED[${split}]}" checkpoint protocol output log dump
  dump="${ROOT}/outputs/refine_stamp_dumps/$(dump_name "${model}" "${split}")"
  if [[ ! -d "${dump}" ]]; then
    echo "ERROR: missing full dump directory: ${dump}" >&2
    return 1
  fi
  if [[ "${sam_type}" == vit_b ]]; then
    checkpoint="${SAM_B_PATH}"
    protocol="stamp_released_prompting_frozen_sam_vit_b_v1"
    output="${OUTPUT_ROOT}/sam_b/$(model_slug "${model}")/$(split_slug "${split}")"
  else
    checkpoint="${SAM_H_PATH}"
    protocol="stamp_official_frozen_sam_h_v1"
    output="${ROOT}/outputs/stamp_official_samh_${model,,}_$(split_slug "${split}")"
  fi
  log="${OUTPUT_ROOT}/logs/sam_${sam_type}_$(model_slug "${model}")_$(split_slug "${split}").log"
  if complete_sam "${output}/eval_summary.json" "${expected}" "${protocol}"; then
    echo "SKIP complete SAM ${sam_type}: ${model} ${split}"
    return
  fi
  wait_for_gpu "${gpu}"
  mkdir -p "${output}"
  echo "RUN SAM ${sam_type}: ${model} ${split} on GPU ${gpu}; log=${log}"
  if [[ "${sam_type}" == vit_b ]]; then
    env CUDA_VISIBLE_DEVICES="${gpu}" "${STAMP_PYTHON}" \
      -m training_free_refine.eval_stamp_sam_variant_only \
      --input-dir "${dump}" --output-dir "${output}" \
      --stamp-code-dir "${STAMP_CODE_DIR}" --sam-path "${checkpoint}" \
      --sam-model-type "${sam_type}" --model-label "${model}" --split-name "${split}" \
      --limit "${expected}" >"${log}" 2>&1
  else
    env CUDA_VISIBLE_DEVICES="${gpu}" "${STAMP_PYTHON}" \
      -m training_free_refine.eval_stamp_sam_h \
      --input-dir "${dump}" --output-dir "${output}" \
      --stamp-code-dir "${STAMP_CODE_DIR}" --sam-path "${checkpoint}" \
      --sam-model-type "${sam_type}" --model-label "${model}" --split-name "${split}" \
      --limit "${expected}" --save-visualizations 0 >"${log}" 2>&1
  fi
  if ! has_sam "${output}/eval_summary.json" "${expected}" "${protocol}"; then
    echo "ERROR: incomplete SAM summary after ${model} ${split} ${sam_type}" >&2
    return 1
  fi
  echo "DONE SAM ${sam_type}: ${model} ${split}"
}

run_gpu_pairs() {
  local index=0 entry model split sam_type pid0 pid1 failed=0
  while (( index < ${#SAM_JOBS[@]} )); do
    entry="${SAM_JOBS[index]}"; IFS='|' read -r model split sam_type <<<"${entry}"
    run_sam_job "${model}" "${split}" "${sam_type}" "${GPU0}" & pid0=$!
    ((index+=1))
    pid1=""
    if (( index < ${#SAM_JOBS[@]} )); then
      entry="${SAM_JOBS[index]}"; IFS='|' read -r model split sam_type <<<"${entry}"
      run_sam_job "${model}" "${split}" "${sam_type}" "${GPU1}" & pid1=$!
      ((index+=1))
    fi
    wait "${pid0}" || failed=1
    [[ -z "${pid1}" ]] || wait "${pid1}" || failed=1
    (( failed == 0 )) || return 1
  done
}

run_output_job() {
  local name="$1" summary="$2" expected="$3"
  shift 3
  local log="${OUTPUT_ROOT}/logs/${name}.log"
  if complete_samples "${summary}" "${expected}"; then
    echo "SKIP complete ${name}"
    return
  fi
  mkdir -p "$(dirname "${summary}")"
  echo "RUN ${name}; log=${log}"
  if "$@" >"${log}" 2>&1; then
    if ! has_samples "${summary}" "${expected}"; then
      echo "ERROR ${name} produced an incomplete summary; log=${log}" >&2
      return 1
    fi
    echo "DONE ${name}"
  else
    local code="$?"
    echo "ERROR ${name} exited ${code}; log=${log}" >&2
    tail -n 80 "${log}" >&2 || true
    return "${code}"
  fi
}

declare -a CPU_PIDS=()
declare -a CPU_NAMES=()
CPU_FAILED=0

wait_first_cpu() {
  local pid="${CPU_PIDS[0]}" name="${CPU_NAMES[0]}"
  if ! wait "${pid}"; then
    echo "ERROR parallel accuracy job failed: ${name}" >&2
    CPU_FAILED=1
  fi
  CPU_PIDS=("${CPU_PIDS[@]:1}")
  CPU_NAMES=("${CPU_NAMES[@]:1}")
}

schedule_cpu() {
  local name="$1"
  shift
  while (( ${#CPU_PIDS[@]} >= CPU_JOBS )); do
    wait_first_cpu
  done
  "$@" &
  CPU_PIDS+=("$!")
  CPU_NAMES+=("${name}")
}

finish_cpu_queue() {
  while (( ${#CPU_PIDS[@]} > 0 )); do
    wait_first_cpu
  done
  (( CPU_FAILED == 0 ))
}

schedule_stamp_study() {
  local study="$1" setting="$2" input="$3" expected="$4"
  shift 4
  local output="${OUTPUT_ROOT}/studies/stamp7b/${study}/${setting}"
  local name="study_stamp7b_${study}_${setting}"
  schedule_cpu "${name}" run_output_job "${name}" "${output}/eval_summary.json" "${expected}" \
    "${STAMP_PYTHON}" -m training_free_refine.eval_stamp_dumps \
    --input-dir "${input}" --output-dir "${output}" --limit "${expected}" \
    --save-visualizations 0 "$@"
}

schedule_text4seg_study() {
  local study="$1" setting="$2" manifest="$3" expected="$4"
  shift 4
  local output="${OUTPUT_ROOT}/studies/text4seg_p24/${study}/${setting}"
  local name="study_text4seg_${study}_${setting}"
  schedule_cpu "${name}" run_output_job "${name}" "${output}/eval_summary.json" "${expected}" \
    conda run --no-capture-output -n "${TEXT4SEG_CONDA_ENV}" \
    python -m training_free_refine.eval_text4seg_outputs \
    --manifest "${manifest}" --output-dir "${output}" --limit "${expected}" \
    --save-visualizations 0 "$@"
}

schedule_postprocess() {
  local model="$1" split="$2" expected="$3" slug output name
  slug="$(split_slug "${split}")"
  output="${OUTPUT_ROOT}/postprocess/${model}/${slug}"
  name="postprocess_${model}_${slug}"
  if [[ "${model}" == stamp7b ]]; then
    local input="${ROOT}/outputs/refine_stamp_dumps/$(dump_name STAMP-7B "${split}")"
    schedule_cpu "${name}" run_output_job "${name}" "${output}/eval_summary.json" "${expected}" \
      "${STAMP_PYTHON}" -m training_free_refine.eval_postprocess_baselines \
      --source stamp --input-dir "${input}" --output-dir "${output}" \
      --model-label STAMP-7B --split-name "${split}" --limit "${expected}" \
      --freeref-backend cpu --methods base densecrf guided_filter fast_bilateral_solver slic_average freeref
  else
    local manifest="${ROOT}/outputs/text4seg_official_${split}/manifest.jsonl"
    if [[ ! -f "${manifest}" ]]; then
      echo "ERROR: missing Text4Seg full manifest: ${manifest}" >&2
      return 1
    fi
    schedule_cpu "${name}" run_output_job "${name}" "${output}/eval_summary.json" "${expected}" \
      conda run --no-capture-output -n "${TEXT4SEG_CONDA_ENV}" \
      python -m training_free_refine.eval_postprocess_baselines \
      --source text4seg --manifest "${manifest}" --output-dir "${output}" \
      --model-label Text4Seg-p24 --split-name "${split}" --limit "${expected}" \
      --freeref-backend cpu --methods base densecrf guided_filter fast_bilateral_solver slic_average freeref
  fi
}

stage "Input preflight"
repair_needed=0
for specification in \
  "refcoco_testA|refcoco_testA_full_stamp2b|5657" \
  "refcoco+_testB|refcocoplus_testB_full_stamp2b|4889"; do
  IFS='|' read -r split dump_name_value expected <<<"${specification}"
  dump="${ROOT}/outputs/refine_stamp_dumps/${dump_name_value}"
  count=0
  [[ -d "${dump}" ]] && count="$(find "${dump}" -maxdepth 1 -type f -name '*.pt' | wc -l)"
  if (( count != expected )); then
    repair_needed=1
  fi
done
if (( repair_needed )); then
  stage "Repairing the two known STAMP-2B missing samples"
  env STAMP2B_REPAIR_GPU="${GPU0}" STAMP2B_REPAIR_SAMH_PARALLEL_JOBS=1 \
    STAMP2B_REPAIR_SAMH_MIN_FREE_MB="${MIN_FREE_MB}" \
    bash "${SCRIPT_DIR}/repair_stamp2b_missing_samples.sh"
  stage "Input preflight after STAMP-2B repair"
fi
gref_generic="${ROOT}/outputs/refine_stamp_dumps/refcocog_test_full"
gref_stamp2b="${ROOT}/outputs/refine_stamp_dumps/refcocog_test_full_stamp2b"
gref_count=0
if [[ -d "${gref_stamp2b}" ]]; then
  gref_count="$(find "${gref_stamp2b}" -maxdepth 1 -type f -name '*.pt' | wc -l)"
elif [[ -d "${gref_generic}" ]]; then
  gref_count="$(find "${gref_generic}" -maxdepth 1 -type f -name '*.pt' | wc -l)"
fi
if (( gref_count != 9602 )); then
  stage "Repairing STAMP-2B RefCOCOg test to 9602 samples"
  env CUDA_DEVICE="${GPU0}" STAMP2B_OTHER_SPLITS=refcocog_test EMPTY_ON_FAILURE=1 \
    STAMP2B_OTHER_COMBINED_OUTPUT="${OUTPUT_ROOT}/stamp2b_refcocog_repair" \
    bash "${SCRIPT_DIR}/run_training_free_stamp2b_refcoco_family_eval.sh"
  stage "Input preflight after STAMP-2B RefCOCOg repair"
fi
for model in STAMP-2B STAMP-7B; do
  for split in "${SPLITS[@]}"; do
    dump="${ROOT}/outputs/refine_stamp_dumps/$(dump_name "${model}" "${split}")"
    expected="${EXPECTED[${split}]}"
    count=0
    [[ -d "${dump}" ]] && count="$(find "${dump}" -maxdepth 1 -type f -name '*.pt' | wc -l)"
    if (( count != expected )); then
      echo "ERROR: ${model} ${split} dumps ${count}/${expected}: ${dump}" >&2
      exit 1
    fi
  done
done
for split in "${SPLITS[@]}"; do
  manifest="${ROOT}/outputs/text4seg_official_${split}/manifest.jsonl"
  expected="${EXPECTED[${split}]}"
  count=0
  [[ -f "${manifest}" ]] && count="$(grep -cve '^[[:space:]]*$' "${manifest}")"
  if (( count != expected )); then
    echo "ERROR: Text4Seg ${split} manifest ${count}/${expected}: ${manifest}" >&2
    exit 1
  fi
done
echo "All 16 STAMP dump sets and 8 Text4Seg manifests are complete."
if [[ "${PREFLIGHT_ONLY}" == "1" ]]; then
  stage "PREFLIGHT COMPLETE"
  exit 0
fi

stage "1/5 Frozen SAM-B full evaluation and SAM-H completeness repair (two H100s)"
declare -a SAM_JOBS=()
for sam_type in vit_b vit_h; do
  for model in STAMP-2B STAMP-7B; do
    for split in "${SPLITS[@]}"; do
      SAM_JOBS+=("${model}|${split}|${sam_type}")
    done
  done
done
run_gpu_pairs
"${STAMP_PYTHON}" -m training_free_refine.summarize_sam_full \
  --input-root "${OUTPUT_ROOT}/sam_b" --output-dir "${OUTPUT_ROOT}/combined/sam_b" \
  --sam-model-type vit_b
"${STAMP_PYTHON}" -m training_free_refine.summarize_sam_h \
  --results-root "${ROOT}/outputs" \
  --output-dir "${ROOT}/outputs/stamp_official_samh_full_comparison"

stage "2/5 Full component ablations, hyperparameters, and eight-split post-processors"
STAMP_TESTA="${ROOT}/outputs/refine_stamp_dumps/refcoco_testA_full_stamp7b"
STAMP_VAL="${ROOT}/outputs/refine_stamp_dumps/refcoco_val_full_stamp7b"
TEXT4SEG_VAL="${ROOT}/outputs/text4seg_official_refcoco_val/manifest.jsonl"
for path in "${STAMP_TESTA}" "${STAMP_VAL}" "${TEXT4SEG_VAL}"; do
  [[ -e "${path}" ]] || { echo "ERROR: required study input missing: ${path}" >&2; exit 1; }
done

schedule_stamp_study ablation full "${STAMP_TESTA}" 5657
schedule_stamp_study ablation uniform_anchoring "${STAMP_TESTA}" 5657 --no-uncertainty-aware-anchoring
schedule_stamp_study ablation unweighted_graph "${STAMP_TESTA}" 5657 --no-appearance-weighted-graph
schedule_stamp_study ablation global_replacement "${STAMP_TESTA}" 5657 --no-selective-fusion

for value in 0.125 0.25 0.5 1 2 4 8; do
  schedule_stamp_study graph_lambda "lambda_${value//./p}" "${STAMP_VAL}" 10834 --graph-lambda "${value}"
done
for value in 0.5 1 2 4 8; do
  schedule_stamp_study confidence_power "gamma_${value//./p}" "${STAMP_VAL}" 10834 --confidence-power "${value}"
done
for value in 0.25 0.5 0.75 1 1.5 2 4; do
  schedule_stamp_study fusion_power "beta_${value//./p}" "${STAMP_VAL}" 10834 --fusion-power "${value}"
done
for value in 0 10 25 50 75 100 200; do
  schedule_stamp_study seed_strength "kappa_${value}" "${STAMP_VAL}" 10834 --seed-strength "${value}"
done
declare -a SEED_PAIRS=("0.4 0.6" "0.3 0.7" "0.2 0.8" "0.1 0.9" "0.05 0.95" "0.01 0.99")
for pair in "${SEED_PAIRS[@]}"; do
  read -r background foreground <<<"${pair}"
  setting="tau_${background//./p}_${foreground//./p}"
  schedule_stamp_study seed_thresholds "${setting}" "${STAMP_VAL}" 10834 \
    --background-seed "${background}" --foreground-seed "${foreground}"
done
declare -a ALL_K_VALUES=(250 500 1000 1024 1500 2000 2500 3000 4000 5000 6000 8000 10000 12000)
for value in "${ALL_K_VALUES[@]}"; do
  schedule_stamp_study n_segments "k_${value}" "${STAMP_VAL}" 10834 --n-segments "${value}"
done
for value in 2 4 8 16 32; do
  schedule_text4seg_study boundary_sigma "sigma_${value}" "${TEXT4SEG_VAL}" 10834 --boundary-sigma "${value}"
done
for split in "${SPLITS[@]}"; do
  schedule_postprocess stamp7b "${split}" "${EXPECTED[${split}]}"
  schedule_postprocess text4seg_p24 "${split}" "${EXPECTED[${split}]}"
done

GREF_PID=""
if [[ "${RUN_GREFCOCO}" == "1" ]]; then
  stage "2/5 Parallel side task: standalone gRefCOCO generalized evaluation"
  env CUDA_DEVICE="${GPU1}" SPECIAL_ONLY_DATASETS=grefcoco \
    SPECIAL_MIN_FREE_GPU_MB="${MIN_FREE_MB}" \
    SPECIAL_COMBINED_OUTPUT="${OUTPUT_ROOT}/grefcoco" \
    bash "${SCRIPT_DIR}/run_training_free_special_datasets_serial.sh" \
    >"${OUTPUT_ROOT}/logs/grefcoco.log" 2>&1 &
  GREF_PID="$!"
fi

finish_cpu_queue
if [[ -n "${GREF_PID}" ]]; then
  wait "${GREF_PID}"
fi
"${STAMP_PYTHON}" -m training_free_refine.summarize_paper_studies \
  --input-root "${OUTPUT_ROOT}/studies" --output-dir "${OUTPUT_ROOT}/combined/studies"
"${STAMP_PYTHON}" -m training_free_refine.summarize_full_postprocess \
  --input-root "${OUTPUT_ROOT}/postprocess" --output-dir "${OUTPUT_ROOT}/combined/postprocess"

stage "3/5 Accuracy complete; preparing strictly serial H100 timing"
complete_timing() {
  local summary="$1" expected_k="$2"
  [[ "${FORCE}" != "1" && -f "${summary}" ]] && python -c \
    'import json,sys; d=json.load(open(sys.argv[1])); k=None if sys.argv[4]=="base" else int(sys.argv[4]); ok=int(d.get("samples",0))==int(sys.argv[2]) and int(d.get("warmup",-1))==int(sys.argv[3]) and "H100" in str(d.get("device","")) and d.get("n_segments")==k; raise SystemExit(0 if ok else 1)' \
    "${summary}" "${TIMING_SAMPLES}" "${TIMING_WARMUP}" "${expected_k}"
}

run_timing() {
  local model="$1" variant="$2" k="$3" output log
  output="${OUTPUT_ROOT}/timing/${model}/${variant}"
  log="${OUTPUT_ROOT}/logs/timing_${model}_${variant}.log"
  if complete_timing "${output}/summary.json" "${k}"; then
    echo "SKIP complete timing ${model}/${variant}"
    return
  fi
  wait_for_gpu "${GPU0}"
  mkdir -p "${output}"
  echo "RUN strictly serial timing ${model}/${variant} on GPU ${GPU0}; log=${log}"
  if [[ "${model}" == stamp2b || "${model}" == stamp7b ]]; then
    local model_path label scale
    if [[ "${model}" == stamp2b ]]; then model_path="${STAMP2B_MODEL}"; label=STAMP-2B; else model_path="${STAMP7B_MODEL}"; label=STAMP-7B; fi
    local args=(
      "${STAMP_PYTHON}" -m efficiency_benchmark.run_stamp
      --root "${ROOT}" --stamp-code-dir "${STAMP_CODE_DIR}" --model "${model_path}"
      --method-label "${label}" --eval-json "${EVAL_JSON}" --sam-path "${SAM_H_PATH}"
      --output-dir "${output}" --warmup "${TIMING_WARMUP}" --samples "${TIMING_SAMPLES}"
      --seed "${TIMING_SEED}" --allow-other-gpu
    )
    if [[ "${variant}" == base ]]; then args+=(--variant base); else args+=(--variant freeref_gpu --n-segments "${k}"); fi
    env CUDA_VISIBLE_DEVICES="${GPU0}" "${args[@]}" >"${log}" 2>&1
  else
    local args=(
      conda run --no-capture-output -n "${TEXT4SEG_CONDA_ENV}"
      python -m efficiency_benchmark.run_text4seg
      --root "${ROOT}" --text4seg-code-dir "${TEXT4SEG_CODE_DIR}"
      --model-path "${TEXT4SEG_MODEL}" --vision-tower "${VISION_TOWER}"
      --eval-json "${EVAL_JSON}" --sam-path "${SAM_H_PATH}"
      --output-dir "${output}" --warmup "${TIMING_WARMUP}" --samples "${TIMING_SAMPLES}"
      --seed "${TIMING_SEED}" --allow-other-gpu
    )
    if [[ "${variant}" == base ]]; then args+=(--variant base); else args+=(--variant freeref_gpu --n-segments "${k}"); fi
    env CUDA_VISIBLE_DEVICES="${GPU0}" "${args[@]}" >"${log}" 2>&1
  fi
  echo "DONE timing ${model}/${variant}"
}

stage "4/5 Strictly serial end-to-end timing on H100 ${GPU0}"
for model in stamp2b stamp7b text4seg_p24; do
  run_timing "${model}" base base
  for value in "${TIMING_K_VALUES[@]}"; do
    run_timing "${model}" "k_${value}" "${value}"
  done
done
"${STAMP_PYTHON}" -m efficiency_benchmark.summarize_k_sweep \
  --input-root "${OUTPUT_ROOT}/timing" --output-dir "${OUTPUT_ROOT}/combined/timing" \
  --k-values "${TIMING_K_VALUES[@]}"

stage "5/5 COMPLETE"
touch "${OUTPUT_ROOT}/COMPLETE"
echo "SAM-B: ${OUTPUT_ROOT}/combined/sam_b/sam_full.md"
echo "Ablations/hyperparameters: ${OUTPUT_ROOT}/combined/studies/paper_studies.md"
echo "Full post-processing: ${OUTPUT_ROOT}/combined/postprocess/postprocess_full.md"
echo "Serial timing: ${OUTPUT_ROOT}/combined/timing/k_timing.md"
if [[ "${RUN_GREFCOCO}" == "1" ]]; then
  echo "gRefCOCO: ${OUTPUT_ROOT}/grefcoco/combined_summary.md"
fi
