# Local Refinement Experiment Summary

This note records the current local-refinement idea, recovered experiment
results, and the code locations needed to reproduce or extend the runs.

## Core Idea

The refinement stage keeps the original STAMP prediction as a coarse global
mask and adds a sparse local correction module. Instead of refining every token
or every image region, it selects a small set of token-grid patches and trains a
lightweight local refiner on RGB crops plus the corresponding mask-token
features.

The main selector idea is uncertainty-guided refinement:

- STAMP produces foreground/background logits for each mask token.
- Foreground probability close to 0.5 means the model is uncertain.
- The selector prioritizes these uncertain tokens because they are more likely
  to lie near ambiguous object boundaries, thin structures, or hard foreground
  regions.
- A boundary-aware signal can be added so that the selector also emphasizes
  coarse-mask transition areas.

The implemented selector variants are:

- `random`: random patch selection baseline.
- `uncertainty`: select patches with highest mask-token uncertainty.
- `boundary`: select patches on coarse-mask boundaries.
- `hybrid`: uncertainty score plus a weighted boundary score.

The intended contribution is not a larger segmentation backbone. It is a cheap
and targeted correction stage: spend local computation only where the mask is
uncertain or structurally fragile.

## Current Training/Evaluation Setup

Available exported dumps on the server include:

- `outputs/refine_stamp_dumps/refcocog_train_1000`
- `outputs/refine_stamp_dumps/refcocog_val_200`
- `outputs/refine_stamp_dumps/refcocog_val_20`
- `outputs/refine_stamp_dumps/refcocog_val_0`

Five-epoch local refiners trained from 1,000 RefCOCOg train dumps are available
under:

- `outputs/refine_stamp_refiner/refcocog_train_1000_random/local_refiner.pt`
- `outputs/refine_stamp_refiner/refcocog_train_1000_uncertainty/local_refiner.pt`
- `outputs/refine_stamp_refiner/refcocog_train_1000_boundary/local_refiner.pt`
- `outputs/refine_stamp_refiner/refcocog_train_1000_hybrid/local_refiner.pt`

Each corresponding `train_summary.json` records 5 epochs, 1,000 paths, 800 train
samples, and 200 validation samples.

## Previously Completed 200-Dump Evaluation

Earlier evaluation used the `refcocog_val_200_*` local refiner checkpoints and
reported 40 evaluated samples per selector:

| Selector | Coarse mIoU | Refined mIoU | Delta | Coarse cIoU | Refined cIoU | cIoU Delta |
|---|---:|---:|---:|---:|---:|---:|
| random | 0.699477 | 0.712273 | +0.012796 | 0.706484 | 0.716498 | +0.010014 |
| uncertainty | 0.699477 | 0.707201 | +0.007724 | 0.706484 | 0.713777 | +0.007293 |
| boundary | 0.699477 | 0.710359 | +0.010882 | 0.706484 | 0.714523 | +0.008038 |
| hybrid | 0.699477 | 0.708373 | +0.008896 | 0.706484 | 0.712686 | +0.006201 |

These results show that all sparse local-refinement selectors improved over the
coarse STAMP mask in the small validation setting.

## 1K-Train to Val-200 Evaluation

The current compatibility evaluation uses the 5-epoch checkpoints trained on
`refcocog_train_1000` and evaluates them on `refcocog_val_200`:

| Selector | Samples | Top-K | Coarse IoU | Refined IoU | Delta | Patch IoU |
|---|---:|---:|---:|---:|---:|---:|
| random | 200 | 64 | 0.747454 | 0.756996 | +0.009541 | 0.094708 |
| uncertainty | 200 | 64 | 0.747454 | 0.753297 | +0.005843 | 0.528971 |
| boundary | 200 | 64 | 0.747454 | 0.755468 | +0.008014 | 0.409094 |
| hybrid | 200 | 64 | 0.747454 | 0.752907 | +0.005453 | 0.489394 |

All four selectors improve over the coarse mask. In this compatibility run,
`random` gives the largest full-mask IoU gain, followed by `boundary`, while
`uncertainty` and `hybrid` still provide consistent positive improvements.

The low `patch_iou` for `random` but strong full-mask IoU suggests that patch
quality and final stitched-mask quality are not perfectly aligned. Random
selection may spread corrections more broadly, while uncertainty-based methods
focus on harder local regions. For a paper-style claim, full-mask metrics should
be reported alongside boundary metrics and visualizations.

## Code Locations

Recovered legacy experiment scaffold:

- `offline_rstamp/refine_stamp_src/refine_stamp/scripts/train_refiner_from_dumps.py`
- `offline_rstamp/refine_stamp_src/refine_stamp/scripts/eval_refiner_from_dumps.py`
- `offline_rstamp/refine_stamp_src/refine_stamp/scripts/eval_selector_quality.py`
- `offline_rstamp/refine_stamp_src/refine_stamp/scripts/visualize_selector.py`
- `offline_rstamp/run/80_train_refine_stamp_refiner_debug.sh`
- `offline_rstamp/run/81_eval_refine_stamp_refiner_debug.sh`
- `offline_rstamp/run/83_train_refine_stamp_refiner_official.sh`
- `offline_rstamp/run/84_eval_refine_stamp_refiner_official_split.sh`
- `offline_rstamp/run/85_run_refine_stamp_official_1000_all_selectors.sh`

Current neutral-name compatibility code:

- `local_refine/train_from_dumps.py`
- `local_refine/eval_from_dumps.py`
- `run_local_refine_train.sh`
- `run_local_refine_eval.sh`

The compatibility code supports old checkpoints that store weights in
`model_state_dict`, handles non-square token grids via dump-provided `grid_hw`,
and falls back from broken cuDNN initialization to native CUDA convolution.

## Reproduction Commands

Evaluate the 5-epoch 1K-trained checkpoints on val-200:

```bash
cd /inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG/MLLM-SEG-data

for S in random uncertainty boundary hybrid; do
  INPUT_DIR="../outputs/refine_stamp_dumps/refcocog_val_200" \
  CHECKPOINT="../outputs/refine_stamp_refiner/refcocog_train_1000_${S}/local_refiner.pt" \
  OUTPUT_DIR="../outputs/refine_stamp_refiner_eval_train1000_on_val200/refcocog_train_1000_${S}_on_val_200" \
  SELECTOR="$S" \
  TOP_K=64 \
  DEVICE=cuda \
  bash run_local_refine_eval.sh
done
```

Summarize the compatibility evaluation:

```bash
python - <<'PY'
import json
from pathlib import Path

root = Path("../outputs/refine_stamp_refiner_eval_train1000_on_val200")
print("selector,samples,top_k,coarse_iou,refined_iou,delta,patch_iou")
for p in sorted(root.glob("*/eval_summary.json")):
    d = json.loads(p.read_text())
    delta = d["refined_iou"] - d["coarse_iou"]
    print(f'{d["selector"]},{d["samples"]},{d["top_k"]},{d["coarse_iou"]:.6f},{d["refined_iou"]:.6f},{delta:.6f},{d["patch_iou"]:.6f}')
PY
```

