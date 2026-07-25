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


class GSVAOfficialExporter:
    """Collect final masks while GSVA's unmodified official validation runs."""

    def __init__(self, output_dir: Path, method: str, split: str) -> None:
        self.output_dir = output_dir.expanduser().resolve()
        self.method = method
        self.split = split
        self.rows: list[dict[str, Any]] = []
        self.image_batches = 0
        self.empty_predictions = 0
        self.no_target_samples = 0

    def record(self, input_dict: dict[str, Any], output_dict: dict[str, Any]) -> None:
        image_paths = input_dict.get("image_paths")
        if not isinstance(image_paths, list) or len(image_paths) != 1:
            raise ValueError(f"GSVA exporter expects one image path, received {image_paths!r}.")
        image = Path(str(image_paths[0])).expanduser().resolve()
        if not image.is_file():
            raise FileNotFoundError(f"GSVA source image is missing: {image}")
        pred_batches = output_dict["pred_masks"]
        gt_batches = output_dict["gt_masks"]
        if len(pred_batches) != 1 or len(gt_batches) != 1:
            raise ValueError(
                "GSVA official validation uses batch size 1; received "
                f"{len(pred_batches)} prediction and {len(gt_batches)} target batches."
            )
        logits = _numpy(pred_batches[0]).astype(np.float32)
        predictions = logits > 0
        targets = _numpy(gt_batches[0]) > 0
        if predictions.shape != targets.shape or predictions.ndim != 3:
            raise ValueError(
                f"GSVA mask stack mismatch: {predictions.shape} vs {targets.shape}."
            )

        image_batch = self.image_batches
        self.image_batches += 1
        for expression_index, (logit, prediction, target) in enumerate(
            zip(logits, predictions, targets)
        ):
            index = len(self.rows)
            paths = artifact_paths(self.output_dir, index)
            atomic_save_logits(paths["logits"], logit)
            atomic_save_mask(paths["mask"], prediction)
            atomic_save_mask(paths["gt"], target)
            no_target = not bool(target.any())
            self.empty_predictions += int(not prediction.any())
            self.no_target_samples += int(no_target)
            self.rows.append(
                {
                    "name": f"gsva_{index:08d}",
                    "method": self.method,
                    "split": self.split,
                    "instance_id": f"{image.name}:{image_batch}:{expression_index}",
                    "image": str(image),
                    "gt_mask": str(paths["gt"]),
                    "prediction": str(paths["logits"]),
                    "prediction_kind": "logits",
                    "threshold": 0.5,
                    "no_target": no_target,
                    "protocol": "gsva_official_validate_final_sam_mask_hook",
                }
            )

    def finalize(self, metrics: Any = None) -> dict[str, Any]:
        if not self.rows:
            raise ValueError("GSVA official validation produced no masks.")
        manifest = self.output_dir / "manifest.jsonl"
        atomic_write_jsonl(manifest, self.rows)
        metric_values: list[float] = []
        if isinstance(metrics, tuple):
            metric_values = [float(value) for value in metrics]
        report = {
            "source": "gsva_official_validate",
            "samples": len(self.rows),
            "image_batches": self.image_batches,
            "split": self.split,
            "method": self.method,
            "empty_predictions": self.empty_predictions,
            "no_target_samples": self.no_target_samples,
            "official_return_metrics": metric_values,
            "manifest": str(manifest.resolve()),
        }
        atomic_write_json(self.output_dir / "export_summary.json", report)
        return report
