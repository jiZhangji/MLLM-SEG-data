#!/usr/bin/env python3
"""Patch STAMP GenerativeSegmenter to export Phase-2 refinement tensors.

This appends a non-invasive monkey patch to `segment_predictor.py`. The original
`generate_with_segmentation()` API is left unchanged. The new method is:

    generate_with_refinement_outputs(image, prompt)

and returns:

    {
        "mask_logits": Tensor[1, N, 2],
        "mask_hidden": Tensor[1, N, D],
        "grid_hw": (grid_h, grid_w),
        "response_text": str,
    }

STAMP's current `bi_logits` is a single binary foreground logit. For selector
code expecting two-class logits, we export `[-logit, +logit]`; the foreground
threshold remains equivalent to `bi_logits > 0`.
"""

from __future__ import annotations

import argparse
from pathlib import Path


PATCH_MARKER = "# >>> Refine-STAMP Phase-2 export patch"
PATCH_END_MARKER = "# <<< Refine-STAMP Phase-2 export patch"


PATCH_CODE = r'''

# >>> Refine-STAMP Phase-2 export patch
@torch.no_grad()
def _refine_stamp_generate_with_refinement_outputs(self, image: Image.Image, prompt: str):
    messages = [{"role": "user", "content": [{"image": image}, {"text": prompt}]}]
    text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = self.processor(text=[text], images=[image], return_tensors="pt")
    merge_size = self.processor.image_processor.merge_size

    inputs = {k: v.to(self.device) for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]
    image_grid_thw = inputs.get("image_grid_thw").to(self.device)
    attention_mask_raw = inputs["attention_mask"].to(self.device)

    outputs = self.model.generate(
        **inputs,
        max_new_tokens=1024,
        use_cache=True,
        return_dict_in_generate=True,
        eos_token_id=self.eos_token_id,
        pad_token_id=self.tokenizer.pad_token_id,
    )

    sequence = outputs.sequences[0]
    full_past_key_values = outputs.past_key_values
    seg_indices = torch.where(sequence == self.seg_token_id)[0].tolist()

    generated_ids = sequence[prompt_len:]
    response_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
    if not seg_indices:
        raise RuntimeError("No <|seg|> token generated; cannot export refinement tensors.")

    num_patches = find_image_patch_info(self.image_pad_id, inputs["input_ids"])

    # Export the first segmentation task. Multi-<seg> support can be added later
    # by returning a list, but Refine-STAMP Phase 1 uses single-object samples.
    idx = seg_indices[0]
    sliced_len = idx + 1
    attention_mask = attention_mask_raw[:, :sliced_len]
    legacy_cache = full_past_key_values.to_legacy_cache()
    past_key_values_sliced = tuple(
        (
            key_layer[:, :, :sliced_len, :],
            value_layer[:, :, :sliced_len, :],
        )
        for key_layer, value_layer in legacy_cache
    )
    past_key_values_sliced = DynamicCache.from_legacy_cache(past_key_values_sliced)

    mask_query_ids = torch.full((1, num_patches), self.mask_token_id, dtype=torch.long, device=self.device)
    mask_query_attention_mask = torch.ones(
        (1, num_patches + sliced_len - attention_mask[0].sum()),
        dtype=torch.long,
        device=self.device,
    )
    mask_query_attention_mask = torch.cat((attention_mask, mask_query_attention_mask), dim=1)
    mask_grid_thw = image_grid_thw[-1].clone().unsqueeze(0)

    mask_pre_ids = sequence.clone().unsqueeze(0)
    mask_ids = torch.cat([mask_pre_ids[0, :sliced_len], mask_query_ids[0]], dim=0).unsqueeze(0)

    seg_forward_outputs = self.model(
        input_ids=mask_ids,
        attention_mask=mask_query_attention_mask,
        image_grid_thw=image_grid_thw,
        pixel_values=inputs["pixel_values"],
        past_key_values=past_key_values_sliced,
        return_dict=True,
        do_classification=True,
        output_hidden_states=True,
    )

    binary_logits = seg_forward_outputs.bi_logits[:, -num_patches:]
    if binary_logits.ndim == 3 and binary_logits.shape[-1] == 1:
        binary_logits = binary_logits.squeeze(-1)
    mask_logits = torch.stack([-binary_logits, binary_logits], dim=-1)

    hidden_states = getattr(seg_forward_outputs, "hidden_states", None)
    if hidden_states is not None and len(hidden_states) > 0:
        mask_hidden = hidden_states[-1][:, -num_patches:, :]
        mask_hidden_is_fallback = False
    elif hasattr(seg_forward_outputs, "last_hidden_state"):
        mask_hidden = seg_forward_outputs.last_hidden_state[:, -num_patches:, :]
        mask_hidden_is_fallback = False
    else:
        # Phase-1 selector quality does not use mask_hidden, but the dump format
        # keeps the field present. If this fallback appears, patch the model
        # forward to return hidden states before training the local refiner.
        mask_hidden = binary_logits.unsqueeze(-1)
        mask_hidden_is_fallback = True

    h_grid, w_grid = mask_grid_thw[0, 1:]
    h_grid, w_grid = int(h_grid / merge_size), int(w_grid / merge_size)

    return {
        "mask_logits": mask_logits,
        "mask_hidden": mask_hidden,
        "grid_hw": (h_grid, w_grid),
        "response_text": response_text,
        "num_patches": int(num_patches),
        "mask_hidden_is_fallback": mask_hidden_is_fallback,
    }


GenerativeSegmenter.generate_with_refinement_outputs = _refine_stamp_generate_with_refinement_outputs
# <<< Refine-STAMP Phase-2 export patch
'''


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stamp-code-dir", required=True, type=Path)
    parser.add_argument("--target", choices=["segment_predictor", "segment_predictor_cache", "both"], default="segment_predictor")
    args = parser.parse_args()

    root = args.stamp_code_dir.expanduser().resolve()
    targets = []
    if args.target in {"segment_predictor", "both"}:
        targets.append(root / "segment_predictor.py")
    if args.target in {"segment_predictor_cache", "both"}:
        targets.append(root / "segment_predictor_cache.py")

    patched = []
    for path in targets:
        if not path.exists():
            print(f"[SKIP] missing: {path}")
            continue
        text = path.read_text(encoding="utf-8")
        if PATCH_MARKER in text:
            start = text.index(PATCH_MARKER)
            end = text.find(PATCH_END_MARKER, start)
            if end == -1:
                raise SystemExit(f"Found patch start but not end marker in {path}")
            end += len(PATCH_END_MARKER)
            text = text[:start].rstrip() + PATCH_CODE + text[end:].lstrip("\n")
            path.write_text(text.rstrip() + "\n", encoding="utf-8")
            print(f"[OK] refreshed existing patch: {path}")
            patched.append(str(path))
            continue
        path.write_text(text.rstrip() + PATCH_CODE + "\n", encoding="utf-8")
        print(f"[OK] patched: {path}")
        patched.append(str(path))

    if not patched:
        raise SystemExit("No files patched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
