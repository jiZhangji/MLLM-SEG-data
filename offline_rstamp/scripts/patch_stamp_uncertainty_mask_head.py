#!/usr/bin/env python3
"""Patch official STAMP with an uncertainty-aware mask head.

This is the clean head-level variant:

    mask_hidden -> uncertainty-aware mask head -> final bi_logits

The patch does not feed STAMP's previous `bi_logits` into the new head. Instead,
it initializes a base classifier from STAMP's original linear classifier and
adds uncertainty/residual branches on top of mask hidden states.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


MARKER_MAIN = "# >>> Uncertainty-aware mask head patch"
MARKER_QWEN = "# >>> Uncertainty-aware mask head qwen patch"


def backup_once(path: Path) -> None:
    backup = path.with_suffix(path.suffix + ".bak_uncertainty_mask_head")
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
        if os.environ.get("STAMP_UNCERTAINTY_MASK_HEAD", "0") == "1":
            from online_token_refine import UncertaintyAwareMaskHead
            hidden_size = getattr(self.model.config, "hidden_size", None)
            if hidden_size is None:
                hidden_size = self.model.config.text_config.hidden_size
            hidden_size = int(hidden_size)
            self.model.uncertainty_mask_head = UncertaintyAwareMaskHead(
                token_dim=hidden_size,
                hidden_size=int(os.environ.get("STAMP_REFINE_HIDDEN_SIZE", "128")),
                use_uncertainty_gate=os.environ.get("STAMP_REFINE_USE_GATE", "0") == "1",
            )
            self.model.uncertainty_mask_head.initialize_from_classifier(self.model.classifier)
            if os.environ.get("STAMP_FREEZE_FOR_UNCERTAINTY_HEAD", "1") == "1":
                for name, param in self.model.named_parameters():
                    param.requires_grad = name.startswith("uncertainty_mask_head.")
            if IS_MAIN_PROCESS:
                trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                total = sum(p.numel() for p in self.model.parameters())
                print(f"--- Uncertainty-aware mask head enabled: trainable={{trainable}} / total={{total}} ---")
        # <<< Uncertainty-aware mask head patch
"""
    text = replace_once(text, old, new, path)
    path.write_text(text, encoding="utf-8")
    print(f"[OK] patched uncertainty head setup: {path}")


def patch_qwen_changes(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if MARKER_QWEN in text:
        print(f"[SKIP] already patched: {path}")
        return
    backup_once(path)
    old = """            last_hidden_state = outputs.hidden_states[-1]
            logits = self.classifier(last_hidden_state)

            return CustomModelOutput(
                bi_logits=logits,
                # hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
"""
    new = f"""            last_hidden_state = outputs.hidden_states[-1]
            if hasattr(self, "uncertainty_mask_head"):
                head_outputs = self.uncertainty_mask_head(last_hidden_state)
                logits = head_outputs["refined_binary_logits"]
            else:
                logits = self.classifier(last_hidden_state)

            return CustomModelOutput(
                bi_logits=logits,
                hidden_states=outputs.hidden_states,  {MARKER_QWEN}
                attentions=outputs.attentions,
            )
"""
    text = replace_once(text, old, new, path)
    path.write_text(text, encoding="utf-8")
    print(f"[OK] patched qwen uncertainty head forward: {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool-repo", required=True, type=Path)
    parser.add_argument("--stamp-code-dir", required=True, type=Path)
    args = parser.parse_args()

    tool_repo = args.tool_repo.expanduser().resolve()
    stamp_code_dir = args.stamp_code_dir.expanduser().resolve()
    copy_module(tool_repo, stamp_code_dir)
    patch_main_uni(stamp_code_dir / "train" / "main_uni.py")
    patch_qwen_changes(stamp_code_dir / "model" / "qwen_changes.py")
    print("[DONE] Uncertainty-aware mask head patch applied.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
