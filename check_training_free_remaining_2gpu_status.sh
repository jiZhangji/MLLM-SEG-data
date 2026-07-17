#!/usr/bin/env bash
set -euo pipefail

ROOT="${MLLM_SEG_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG}"
STAMP_ENV="${STAMP_ENV_PATH:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/my_global_cache/conda/envs/STAMP}"
PYTHON_BIN="${TRAINING_FREE_PYTHON:-${STAMP_ENV}/bin/python}"
JSON_DIR="${ROOT}/code/STAMP/playground/data/json_eval_baseline"
SPLITS=(refcoco_val refcoco_testA refcoco_testB refcoco+_val refcoco+_testA refcoco+_testB)

if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="$(command -v python)"
fi

echo "===== GPUs ====="
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader

for scale in 7b 2b; do
  echo
  echo "===== STAMP-${scale^^} RefCOCO family ====="
  for split in "${SPLITS[@]}"; do
    safe_split="${split//+/plus}"
    json_path="${JSON_DIR}/${split}.json"
    dump_dir="${ROOT}/outputs/refine_stamp_dumps/${safe_split}_full_stamp${scale}"
    result_dir="${ROOT}/outputs/training_free_refine_stamp${scale}_${safe_split}_full"
    if [[ -f "${json_path}" ]]; then
      expected="$("${PYTHON_BIN}" -c 'import json,sys; print(len(json.load(open(sys.argv[1], encoding="utf-8"))))' "${json_path}")"
    else
      expected="json-missing"
    fi
    dumps="$(find "${dump_dir}" -maxdepth 1 -type f -name '*.pt' 2>/dev/null | wc -l || true)"
    summary="$([[ -f "${result_dir}/eval_summary.json" ]] && echo yes || echo no)"
    echo "${split}: dumps ${dumps}/${expected} | summary ${summary}"
  done
  combined="${ROOT}/outputs/training_free_refine_stamp${scale}_refcoco_family_full_comparison/combined_summary.json"
  echo "Combined summary: $([[ -f "${combined}" ]] && echo yes || echo no)"
done

echo
echo "===== Active related processes ====="
if ! pgrep -af 'run_training_free_stamp(7b|2b)_refcoco_family_eval|export_stamp_refinement_dumps|training_free_refine.eval_stamp_dumps'; then
  echo "none"
fi
