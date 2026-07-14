# Training-Free Uncertainty Graph Refinement

This package refines STAMP mask-token probabilities without training an
adapter. It consumes only the RGB image, STAMP mask logits, and the token-grid
shape. Ground-truth masks are used by the evaluator only and never by the
refinement algorithm.

## Method

1. Reconstruct the STAMP foreground probability with the existing softmax and
   bilinear upsampling protocol.
2. Compute uncertainty as `1 - abs(2 * p - 1)`.
3. Build an RGB/Lab-aware SLIC superpixel adjacency graph.
4. Solve the sparse graph-Laplacian energy
   `sum C_k(q_k-p_k)^2 + lambda * sum w_kl(q_k-q_l)^2`, where
   `C_k=(1-u_k)^2+eps` and confident foreground/background superpixels are
   strengthened as seeds.
5. Fuse the graph solution only where STAMP is uncertain:
   `p_refined=(1-u)*p + u*q`.

There are no learned parameters, checkpoints, backpropagation steps, or uses of
STAMP `mask_hidden`.

## Evaluation

```bash
python -m training_free_refine.eval_stamp_dumps \
  --input-dir ../outputs/refine_stamp_dumps/refcocog_val_200 \
  --output-dir ../outputs/training_free_refine_on_val200
```

The evaluator writes `eval_summary.json`, `eval_rows.csv`, and optional
visualizations. Each visualization contains, from left to right: RGB image, GT
overlay, coarse STAMP overlay, uncertainty, and refined overlay.

Run the full validation split before test. Fix the configuration after
validation and evaluate test only once to preserve the training-free protocol.

To run unit tests, complete RefCOCOg val, complete test, and the final combined
comparison sequentially with one script:

```bash
bash run_training_free_refine_full_eval.sh
```

The combined report is written to
`../outputs/training_free_refine_refcocog_full_comparison/combined_summary.md`.
