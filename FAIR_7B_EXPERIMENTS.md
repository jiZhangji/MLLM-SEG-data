# Fair STAMP-7B vs OnePass-7B

This runner trains two models sequentially from the same plain
`Qwen/Qwen2-VL-7B-Instruct` initialization.

1. `stamp7b_native`: teacher-forces `<|seg|>`, then appends one `<|mask|>`
   query per merged image token and performs STAMP's second classification pass.
2. `onepass7b`: inserts `<|seg|>` and all spatial mask queries before the
   transformer and predicts the mask in one Hybrid-Attention forward pass.

Both runs share the dataset, epochs, image resolution, LoRA configuration,
learning rate, effective global batch size, scheduler, warmup, and seed. The
existing `STAMP-7B-lora` adapter is deliberately not used for initialization.
The native STAMP objective includes language-model and mask losses; OnePass has
no autoregressive generation objective and uses its mask loss.

Run `python run_fair_7b_experiments.py --help` for all options. Checkpoints are
written after every epoch and at `--save-steps`, so either branch can resume
with `--resume-stamp` or `--resume-onepass`.

