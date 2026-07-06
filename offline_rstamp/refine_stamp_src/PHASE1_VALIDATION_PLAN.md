# Refine-STAMP Small-Sample Validation Plan

## Goal

Before training a local refiner, validate whether STAMP's own uncertainty and coarse-boundary signals can identify useful refinement regions.

This phase does not claim final mask improvement yet. It answers a simpler question:

> Are the patches selected by uncertainty/boundary cues closer to true object boundaries and STAMP error regions than random patches?

If the selector cannot beat random selection, training a refiner is unlikely to help.

## Stage 1: No-Training Selector Diagnosis

### Inputs

Each sample should be exported from STAMP Phase 2 as a `.pt` file containing:

```python
{
  "mask_logits": Tensor[1, N, 2],
  "mask_hidden": Tensor[1, N, D],  # optional for Stage 1
  "grid_hw": (grid_h, grid_w),
  "image_path": "...",
  "mask_path": "...",             # GT mask
  "query": "..."                   # optional
}
```

Stage 1 only requires:

```text
mask_logits
grid_hw
mask_path
```

`image_path` is needed only for visualization.

### Compared Selectors

```text
random
uncertainty
boundary
hybrid = uncertainty + boundary_weight * boundary
```

### Metrics

For each selector, report:

```text
selected_gt_boundary_rate
selected_error_rate
selected_fg_rate
selected_gt_fg_rate
score_mean_on_selected
```

Definitions:

- `selected_gt_boundary_rate`: selected patches overlapping GT boundary.
- `selected_error_rate`: selected patches overlapping coarse-mask error.
- `selected_fg_rate`: selected patches predicted foreground by STAMP.
- `selected_gt_fg_rate`: selected patches overlapping GT foreground.

Baseline coarse mask metrics are also reported:

```text
coarse_iou
coarse_boundary_iou
coarse_fg_ratio
gt_fg_ratio
```

### Expected Result

Useful selectors should satisfy:

```text
hybrid selected_gt_boundary_rate > random
hybrid selected_error_rate > random
boundary selected_gt_boundary_rate > random
uncertainty selected_error_rate > random
```

If uncertainty mainly selects background, add foreground-neighborhood constraints before training the refiner.

## Stage 2: Small Refiner Training

Only after Stage 1 succeeds:

```text
train: 200-500 samples
val: 100 samples
STAMP: frozen
trainable: local refiner only
```

Compare:

```text
STAMP baseline without SAM
STAMP + random-K refiner
STAMP + uncertainty-only refiner
STAMP + boundary-only refiner
STAMP + hybrid refiner
```

Primary metrics:

```text
cIoU / gIoU
Boundary IoU
Boundary F-score
small-object IoU
latency
GPU memory
```

## Go / No-Go

Continue if:

```text
selector beats random on boundary/error targeting
oracle selector improves Boundary IoU after refiner training
hybrid refiner improves Boundary IoU without reducing cIoU
```

Redesign if:

```text
random selector performs the same as hybrid
selected patches mostly hit irrelevant background
coordinate alignment is unstable
```

