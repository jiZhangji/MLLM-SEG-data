#!/usr/bin/env bash
set -euo pipefail

python -m training_free_refine.eval_stamp_dumps "$@"
