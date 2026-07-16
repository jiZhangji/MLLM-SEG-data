#!/usr/bin/env bash
set -euo pipefail

ROOT="${MLLM_SEG_ROOT:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG}"
STAMP_ENV="${STAMP_ENV_PATH:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/my_global_cache/conda/envs/STAMP}"
PYTHON_BIN="${TRAINING_FREE_PYTHON:-${STAMP_ENV}/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="$(command -v python)"
fi
EVAL_LIMIT="${STAMP7B_OTHER_EVAL_LIMIT:-0}"
SPLITS_TEXT="${STAMP7B_OTHER_SPLITS:-refcoco_val refcoco_testA refcoco_testB refcoco+_val refcoco+_testA refcoco+_testB}"
JSON_DIR="${ROOT}/code/STAMP/playground/data/json_eval_baseline"
COMBINED_OUTPUT="${STAMP7B_OTHER_COMBINED_OUTPUT:-${ROOT}/outputs/training_free_refine_stamp7b_refcoco_family_full_comparison}"

read -r -a SPLITS <<< "${SPLITS_TEXT}"
if [[ "${EVAL_LIMIT}" == "0" ]]; then
  run_suffix="full"
else
  run_suffix="limit${EVAL_LIMIT}"
fi

for split in "${SPLITS[@]}"; do
  safe_split="${split//+/plus}"
  json_path="${JSON_DIR}/${split}.json"
  dump_dir="${ROOT}/outputs/refine_stamp_dumps/${safe_split}_${run_suffix}_stamp7b"
  result_dir="${ROOT}/outputs/training_free_refine_stamp7b_${safe_split}_${run_suffix}"
  if [[ -f "${json_path}" ]]; then
    expected="$("${PYTHON_BIN}" -c 'import json,sys; n=len(json.load(open(sys.argv[1], encoding="utf-8"))); limit=int(sys.argv[2]); print(min(n, limit) if limit else n)' "${json_path}" "${EVAL_LIMIT}")"
  else
    expected="json-missing"
  fi
  dumps="$(find "${dump_dir}" -maxdepth 1 -type f -name '*.pt' 2>/dev/null | wc -l || true)"
  summary="$([[ -f "${result_dir}/eval_summary.json" ]] && echo yes || echo no)"
  echo "${split}: dumps ${dumps}/${expected} | summary ${summary}"
done

echo "Combined summary: $([[ -f "${COMBINED_OUTPUT}/combined_summary.json" ]] && echo yes || echo no)"
echo
echo "Active related processes:"
if ! pgrep -af 'run_training_free_stamp7b_refcoco_family_eval|export_stamp_refinement_dumps|training_free_refine.eval_stamp_dumps'; then
  echo "none"
fi
