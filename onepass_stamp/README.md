# OnePass-STAMP

This package implements the final query-driven OnePass-STAMP design without
the historical refinement adapter or uncertainty branch.

## Model path

```text
image + prompt + task query + semantic <|seg|> query + spatial <|mask|> queries
  -> one STAMP Transformer forward with hybrid mask attention
  -> shared linear mask classifier
  -> patch mask
```

Spatial query initialization is:

```text
Q_i = E_STAMP(<|mask|>) + delta_mask + W_v V_i + P_2d(i)
```

`W_v` starts as identity, all query deltas and the continuous normalized 2D
position projection start at zero, and the linear classifier starts from the
released STAMP checkpoint. The pretrained STAMP backbone is frozen.

By default, small in-place LoRA matrices are trained on the text Transformer's
`q/k/v/o` and MLP projections so the pretrained backbone can adapt to the SEG
query changing from an output routing token to an input semantic query. The
original projection weights stay frozen.

## Important properties

- `generate()` is never called.
- No KV cache slicing or second model invocation is used.
- No original mask logits are consumed as features.
- No residual correction, uncertainty gate, or SAM decoder is included.
- Training uses direct BCE plus Dice loss.
- Checkpoints contain only query parameters, the direct mask classifier, and
  optional optimizer/scheduler state. They do not duplicate the 2B backbone.

Use `python -m onepass_stamp.train --help` and
`python -m onepass_stamp.eval --help` for all command-line options.
