# Refine-STAMP Design Review

## Verdict

This direction is more reasonable than continuing prompt-level structured prior tricks for the current stage.

Why:

- Official-aligned RefCOCOg already shows that STAMP-2B is strong and that simple text-only prompt wrapping is essentially tied with the baseline.
- The earlier oracle-prior gain used GT geometry and is best treated as an upper-bound diagnostic, not a fair deployable result.
- STAMP's remaining SAM-free weakness is more likely in mask granularity, boundaries, small objects and thin structures than in standard RefCOCOg prompt triggering.
- A frozen-STAMP local refiner preserves the original `<SEG>` trigger, All-mask Prediction path, dialogue ability and baseline stability.

## What This MVP Tests

The MVP tests whether STAMP's own outputs contain enough signal to choose useful refinement regions:

```text
mask logits -> uncertainty
coarse mask -> boundary
mask hidden states -> query-conditioned local semantics
RGB crops -> local high-resolution edge/texture
```

The first useful result is not necessarily a large cIoU gain. The first important signals are:

- selected patches concentrate around target boundaries;
- oracle boundary selection improves Boundary IoU;
- uncertainty/boundary selection beats random selection;
- local refinement improves small/thin/boundary subsets without damaging overall cIoU;
- latency stays well below STAMP + SAM.

## Main Risks

1. Coordinate mismatch among STAMP image preprocessing, patch grid, RGB crops and GT masks.
2. The selector may choose uncertain background instead of useful object boundaries.
3. Local patch replacement may introduce block artifacts.
4. Official RefCOCOg may be too saturated for visible full-set gains.
5. Extracting `mask_hidden` and `grid_hw` from official STAMP may require careful Phase-2 patching.

## Go / No-Go Criteria

Continue if any one of these holds:

```text
Boundary IoU improves by >= 3 points
cIoU improves by >= 1 point
hard/small/thin subset improves clearly with <= 20% extra latency
performance approaches STAMP + SAM with much lower overhead
```

Stop or redesign if:

```text
oracle selector cannot improve Boundary IoU
random selection matches uncertainty/boundary selection
coordinate checks reveal unstable grid/crop alignment
latency approaches or exceeds SAM
```

## Recommended First Server Run

1. Patch or hook official STAMP to export `mask_logits`, `mask_hidden`, and `grid_hw`.
2. Run `inspect_stamp_outputs.py` on a few saved tensor dumps.
3. Visualize uncertainty/boundary/selected patches before training.
4. Train only `LocalPatchRefiner` with oracle selector first.
5. Compare random, uncertainty, boundary and hybrid selectors.

