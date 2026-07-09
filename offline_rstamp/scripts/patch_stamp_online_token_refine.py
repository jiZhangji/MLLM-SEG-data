#!/usr/bin/env python3
"""Patch official STAMP for online calibrated token refinement training.

This patch targets the public STAMP layout:

- train/main_uni.py
- train/seg_trainer.py
- model/qwen_changes.py

It keeps STAMP's batched segmentation training path. Instead of exporting
dumps, `SegmentationSFTTrainer.compute_loss()` uses the normal batched
`do_classification=True` forward, obtains `bi_logits` and `hidden_states`, and
optionally passes them through `model.online_token_refiner`.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


MARKER_MAIN = "# >>> Online token refine patch"
MARKER_TRAINER = "# >>> Online token refine trainer patch"
MARKER_QWEN = "# >>> Online token refine qwen_changes patch"


def backup_once(path: Path) -> None:
    backup = path.with_suffix(path.suffix + ".bak_online_token_refine")
    if not backup.exists():
        backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")


def replace_once(text: str, old: str, new: str, path: Path) -> str:
    if old not in text:
        raise RuntimeError(f"Could not find expected block in {path}:\n{old[:500]}")
    return text.replace(old, new, 1)


def copy_module(tool_repo: Path, stamp_code_dir: Path) -> None:
    src = tool_repo / "online_token_refine"
    dst = stamp_code_dir / "online_token_refine"
    if not src.exists():
        raise FileNotFoundError(src)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    print(f"[OK] copied {src} -> {dst}")


def patch_qwen_changes(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if MARKER_QWEN in text:
        print(f"[SKIP] already patched: {path}")
        return
    backup_once(path)
    old = """            return CustomModelOutput(
                bi_logits=logits,
                # hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
"""
    new = f"""            return CustomModelOutput(
                bi_logits=logits,
                hidden_states=outputs.hidden_states,  {MARKER_QWEN}
                attentions=outputs.attentions,
            )
"""
    text = replace_once(text, old, new, path)
    path.write_text(text, encoding="utf-8")
    print(f"[OK] patched qwen hidden_states: {path}")


def patch_main_uni(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if MARKER_MAIN in text:
        print(f"[SKIP] already patched: {path}")
        return
    backup_once(path)
    old = """        self.model.mask_token_id = mask_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|mask|>")
"""
    new = f"""        self.model.mask_token_id = mask_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|mask|>")

        {MARKER_MAIN}
        if os.environ.get("STAMP_ONLINE_TOKEN_REFINE", "0") == "1":
            from online_token_refine import OnlineCalibratedResidualRefiner
            hidden_size = getattr(self.model.config, "hidden_size", None)
            if hidden_size is None:
                hidden_size = self.model.config.text_config.hidden_size
            hidden_size = int(hidden_size)
            self.model.online_token_refiner = OnlineCalibratedResidualRefiner(
                token_dim=hidden_size,
                hidden_size=int(os.environ.get("STAMP_REFINE_HIDDEN_SIZE", "128")),
                use_uncertainty_gate=os.environ.get("STAMP_REFINE_USE_GATE", "0") == "1",
                trainable_logit_calibration=os.environ.get("STAMP_REFINE_CALIBRATE", "1") == "1",
            )
            if os.environ.get("STAMP_FREEZE_FOR_ONLINE_REFINE", "1") == "1":
                for name, param in self.model.named_parameters():
                    param.requires_grad = name.startswith("online_token_refiner.")
            if IS_MAIN_PROCESS:
                trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                total = sum(p.numel() for p in self.model.parameters())
                print(f"--- Online token refiner enabled: trainable={{trainable}} / total={{total}} ---")
        # <<< Online token refine patch
"""
    text = replace_once(text, old, new, path)
    path.write_text(text, encoding="utf-8")
    print(f"[OK] patched main trainer setup: {path}")


def patch_seg_trainer(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if MARKER_TRAINER in text:
        print(f"[SKIP] already patched: {path}")
        return
    backup_once(path)
    if "from online_token_refine import token_refine_loss" not in text:
        text = text.replace(
            "import torchvision.transforms as T\n",
            "import torchvision.transforms as T\nfrom online_token_refine import token_refine_loss\n",
            1,
        )

    old_forward = """            seg_logits = model(input_ids=batch_input_ids, attention_mask=batch_attn_mask, pixel_values=batch_pixel_values,
                  image_grid_thw=batch_grid_thw, output_hidden_states=True, do_classification=True).bi_logits
            mask_preds = seg_logits[batch_idx_mask]
"""
    new_forward = f"""            {MARKER_TRAINER}
            seg_outputs = model(
                input_ids=batch_input_ids,
                attention_mask=batch_attn_mask,
                pixel_values=batch_pixel_values,
                image_grid_thw=batch_grid_thw,
                output_hidden_states=True,
                do_classification=True,
            )
            seg_logits = seg_outputs.bi_logits
            mask_preds = seg_logits[batch_idx_mask]
            mask_hidden = None
            if hasattr(model, "online_token_refiner"):
                if getattr(seg_outputs, "hidden_states", None) is None:
                    raise RuntimeError("Online token refinement requires seg_outputs.hidden_states; patch model/qwen_changes.py first.")
                mask_hidden = seg_outputs.hidden_states[-1][batch_idx_mask]
            # <<< Online token refine trainer patch
"""
    text = replace_once(text, old_forward, new_forward, path)

    old_loss = """                    loss_fn_weighted = WeightedDiceBCELoss(alpha=0.3, beta=0.7)
                    loss_seg_ = loss_fn_weighted(mask_pred.float(), binary_gt_labels.unsqueeze(1).float())
                    img_show = T.ToTensor()(all_images[i][-1]).permute(1, 2, 0).cpu()
                    loss_seg += loss_seg_
                    num_gt += 1
"""
    new_loss = """                    if hasattr(model, "online_token_refiner"):
                        hidden_slice = mask_hidden[start_p - num_p: start_p].unsqueeze(0)
                        logits_slice = mask_pred.float().unsqueeze(0)
                        two_class_logits = torch.cat([-logits_slice, logits_slice], dim=-1)
                        refine_outputs = model.online_token_refiner(hidden_slice, two_class_logits)
                        loss_seg_ = token_refine_loss(
                            refine_outputs["refined_logits"],
                            binary_gt_labels.unsqueeze(0),
                            uncertainty=refine_outputs["uncertainty"],
                            delta_logits=refine_outputs["delta_logits"],
                            uncertainty_loss_weight=float(os.environ.get("STAMP_REFINE_UNCERTAINTY_WEIGHT", "2.0")),
                            delta_reg_weight=float(os.environ.get("STAMP_REFINE_DELTA_REG", "0.01")),
                        )
                    else:
                        loss_fn_weighted = WeightedDiceBCELoss(alpha=0.3, beta=0.7)
                        loss_seg_ = loss_fn_weighted(mask_pred.float(), binary_gt_labels.unsqueeze(1).float())
                    img_show = T.ToTensor()(all_images[i][-1]).permute(1, 2, 0).cpu()
                    loss_seg += loss_seg_
                    num_gt += 1
"""
    text = replace_once(text, old_loss, new_loss, path)
    path.write_text(text, encoding="utf-8")
    print(f"[OK] patched online refine loss: {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool-repo", required=True, type=Path)
    parser.add_argument("--stamp-code-dir", required=True, type=Path)
    args = parser.parse_args()

    tool_repo = args.tool_repo.expanduser().resolve()
    stamp_code_dir = args.stamp_code_dir.expanduser().resolve()
    copy_module(tool_repo, stamp_code_dir)
    patch_qwen_changes(stamp_code_dir / "model" / "qwen_changes.py")
    patch_main_uni(stamp_code_dir / "train" / "main_uni.py")
    patch_seg_trainer(stamp_code_dir / "train" / "seg_trainer.py")
    print("[DONE] Online token refinement patch applied.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
