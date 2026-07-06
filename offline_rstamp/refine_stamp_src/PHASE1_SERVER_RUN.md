# Refine-STAMP Phase-1 Server Run

## What This Runs

This is the first real-data diagnosis for Refine-STAMP.

It does not train the local refiner yet. It compares whether these selectors choose useful patches on real RefCOCOg samples:

```text
random
uncertainty
boundary
hybrid
```

The main report answers:

> Do uncertainty/boundary selected patches overlap GT boundary and STAMP coarse-mask errors more often than random patches?

## Step 0: Pull Latest Tool Repo

```bash
export ROOT="/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/MLLM-SEG"
cd "$ROOT/MLLM-SEG-data"
git pull
```

## Step 1: Smoke Test the Refine-STAMP Utilities

```bash
bash offline_rstamp/run/70_test_refine_stamp_mvp.sh
```

Optional synthetic demo:

```bash
bash offline_rstamp/run/74_eval_refine_stamp_selector_quality_demo.sh
```

This only checks the utility code. It is not a dataset result.

## Step 2: Run Real RefCOCOg Phase-1 Diagnosis

```bash
SPLIT=refcocog_val \
EVAL_LIMIT=20 \
CUDA_VISIBLE_DEVICES=0 \
bash offline_rstamp/run/76_refine_stamp_phase1_refcocog.sh
```

Outputs:

```text
$ROOT/outputs/refine_stamp_dumps/refcocog_val_20/
$ROOT/outputs/refine_stamp_phase1_selector_quality/refcocog_val_20/
```

Main report:

```bash
cat "$ROOT/outputs/refine_stamp_phase1_selector_quality/refcocog_val_20/selector_quality_summary.md"
```

Visualizations:

```text
$ROOT/outputs/refine_stamp_phase1_selector_quality/refcocog_val_20/visualizations/
```

## Required STAMP Export API

The export script calls `GenerativeSegmenter` and looks for one of these methods:

```text
generate_with_refinement_outputs(image, query)
generate_with_refinement(image, query)
export_refinement_outputs(image, query)
forward_for_refinement(images=[...], texts=[...])
```

The method must return:

```python
{
  "mask_logits": Tensor[B, N, 2],
  "mask_hidden": Tensor[B, N, D],
  "grid_hw": (grid_h, grid_w),
}
```

If none exists, the script will fail early with a clear error. Then patch STAMP Phase 2 to expose:

```text
z_mask / mask hidden states
mask logits from the mask classifier
dynamic grid_hw
```

## How to Interpret the Report

The most important columns are:

```text
GT Boundary Hit
Error Hit
```

Good sign:

```text
hybrid > random on GT Boundary Hit
hybrid > random on Error Hit
boundary > random on GT Boundary Hit
uncertainty > random on Error Hit
```

If this holds, proceed to Stage 2: train a small frozen-STAMP local refiner.

If random is similar to hybrid, redesign the selector before training.

