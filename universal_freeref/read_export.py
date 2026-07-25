from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .export_utils import (
    artifact_paths,
    atomic_save_logits,
    atomic_save_mask,
    atomic_write_json,
    atomic_write_jsonl,
)


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _batch_metadata(value: Any) -> list[Any]:
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        return value[0]
    return value if isinstance(value, list) else []


class READOfficialExporter:
    """Flatten READ's image-grouped teacher-forced validation outputs."""

    def __init__(self, output_dir: Path, method: str, split: str) -> None:
        self.output_dir = output_dir.expanduser().resolve()
        self.method = method
        self.split = split
        self.rows: list[dict[str, Any]] = []
        self.image_batches = 0
        self.empty_predictions = 0
        self.no_target_samples = 0
        self.intersection = 0
        self.union = 0
        self.per_sample_ious: list[float] = []

    @staticmethod
    def _mask_stack(value: Any) -> np.ndarray:
        array = _numpy(value)
        if array.ndim == 4 and array.shape[1] == 1:
            array = array[:, 0]
        if array.ndim == 2:
            array = array[None]
        if array.ndim != 3:
            raise ValueError(f"READ masks must have shape [N,H,W], received {array.shape}.")
        return array

    def record(self, input_dict: dict[str, Any], output_dict: dict[str, Any]) -> None:
        image_paths = input_dict.get("image_paths")
        if not isinstance(image_paths, list) or len(image_paths) != 1:
            raise ValueError(f"READ exporter expects one image path, received {image_paths!r}.")
        image = Path(str(image_paths[0])).expanduser().resolve()
        if not image.is_file():
            raise FileNotFoundError(f"READ source image is missing: {image}")
        pred_batches = output_dict["pred_masks"]
        gt_batches = output_dict["gt_masks"]
        if len(pred_batches) != 1 or len(gt_batches) != 1:
            raise ValueError("READ official validation must use image batch size 1.")
        logits = self._mask_stack(pred_batches[0]).astype(np.float32)
        targets = self._mask_stack(gt_batches[0]) > 0
        if logits.shape != targets.shape:
            raise ValueError(f"READ mask stack mismatch: {logits.shape} vs {targets.shape}.")

        queries = input_dict.get("conversation_list")
        queries = queries if isinstance(queries, list) else []
        ref_ids = _batch_metadata(input_dict.get("ref_ids"))
        sent_ids = _batch_metadata(input_dict.get("sent_ids"))
        image_batch = self.image_batches
        self.image_batches += 1
        for expression_index, (logit, target) in enumerate(zip(logits, targets)):
            prediction = logit > 0
            intersection = int(np.logical_and(prediction, target).sum())
            union = int(np.logical_or(prediction, target).sum())
            self.intersection += intersection
            self.union += union
            self.per_sample_ious.append(1.0 if union == 0 else intersection / union)
            no_target = not bool(target.any())
            self.empty_predictions += int(not prediction.any())
            self.no_target_samples += int(no_target)

            index = len(self.rows)
            paths = artifact_paths(self.output_dir, index)
            atomic_save_logits(paths["logits"], logit)
            atomic_save_mask(paths["mask"], prediction)
            atomic_save_mask(paths["gt"], target)
            ref_id = ref_ids[expression_index] if expression_index < len(ref_ids) else image_batch
            sent_id = sent_ids[expression_index] if expression_index < len(sent_ids) else expression_index
            self.rows.append(
                {
                    "name": f"read_{index:08d}",
                    "method": self.method,
                    "split": self.split,
                    "instance_id": f"{ref_id}:{sent_id}",
                    "image": str(image),
                    "gt_mask": str(paths["gt"]),
                    "prediction": str(paths["logits"]),
                    "prediction_kind": "logits",
                    "threshold": 0.5,
                    "no_target": no_target,
                    "query": str(queries[expression_index]) if expression_index < len(queries) else "",
                    "protocol": "read_official_teacher_forced_seg_token_sasp_mask",
                }
            )

    def finalize(self) -> dict[str, Any]:
        if not self.rows:
            raise ValueError("READ official validation produced no masks.")
        manifest = self.output_dir / "manifest.jsonl"
        atomic_write_jsonl(manifest, self.rows)
        report = {
            "source": "read_official_teacher_forced_validation",
            "samples": len(self.rows),
            "image_batches": self.image_batches,
            "split": self.split,
            "method": self.method,
            "empty_predictions": self.empty_predictions,
            "no_target_samples": self.no_target_samples,
            "official_cIoU": self.intersection / max(self.union, 1),
            "official_gIoU": float(np.mean(self.per_sample_ious)),
            "manifest": str(manifest.resolve()),
        }
        atomic_write_json(self.output_dir / "export_summary.json", report)
        return report
