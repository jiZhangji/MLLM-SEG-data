from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .export_utils import artifact_paths, atomic_save_mask, atomic_write_json, atomic_write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert the per-expression prediction file saved by ReLA's official "
            "ReferEvaluator into a paired FreeRef manifest."
        )
    )
    parser.add_argument("--input-pth", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--method", default="ReLA-Swin-B-official")
    parser.add_argument(
        "--ignore-pred-nt",
        action="store_true",
        help=(
            "Keep the raw mask even when ReLA predicts the no-target class. By default "
            "a predicted no-target sample is exported as an empty final mask."
        ),
    )
    return parser.parse_args()


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except (RuntimeError, ValueError):
            pass
    return value


def normalize_image_id(value: Any) -> str:
    value = _scalar(value)
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    text = str(value)
    if text.isdigit():
        return str(int(text))
    return text


def image_candidates(image_root: Path, image_id: str) -> list[Path]:
    raw = Path(image_id)
    candidates: list[Path] = []
    if raw.suffix:
        candidates.extend((image_root / raw, image_root / raw.name))
    if image_id.isdigit():
        number = int(image_id)
        candidates.extend(
            (
                image_root / f"COCO_train2014_{number:012d}.jpg",
                image_root / f"{number:012d}.jpg",
                image_root / f"{number}.jpg",
            )
        )
    candidates.extend((image_root / image_id, image_root / raw.name))
    return list(dict.fromkeys(candidates))


def resolve_image(image_root: Path, image_id: str) -> Path:
    for candidate in image_candidates(image_root, image_id):
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Cannot resolve ReLA image_id={image_id!r} below {image_root}; "
        f"tried {[str(path) for path in image_candidates(image_root, image_id)]}."
    )


def prediction_rows(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        for key in ("predictions", "results", "outputs"):
            if isinstance(value.get(key), list):
                value = value[key]
                break
    if not isinstance(value, list):
        raise TypeError(
            "ReLA prediction file must contain the list produced by ReferEvaluator "
            f"(received {type(value).__name__})."
        )
    if not all(isinstance(row, dict) for row in value):
        raise TypeError("Every ReLA prediction entry must be a dictionary.")
    return value


def convert_predictions(
    predictions: Iterable[dict[str, Any]],
    *,
    output_dir: Path,
    image_root: Path,
    split: str,
    method: str,
    respect_pred_nt: bool,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    empty_predictions = 0
    predicted_no_target = 0
    ground_truth_no_target = 0

    for index, sample in enumerate(predictions):
        missing = {"img_id", "pred_mask", "gt_mask"} - set(sample)
        if missing:
            raise KeyError(f"ReLA prediction {index} is missing keys: {sorted(missing)}.")
        image_id = normalize_image_id(sample["img_id"])
        image = resolve_image(image_root, image_id)
        prediction = _as_numpy(sample["pred_mask"]).squeeze().astype(bool)
        target = _as_numpy(sample["gt_mask"]).squeeze().astype(bool)
        if prediction.ndim != 2 or target.ndim != 2:
            raise ValueError(
                f"ReLA prediction {index} masks must be 2-D; "
                f"received {prediction.shape} and {target.shape}."
            )
        if prediction.shape != target.shape:
            raise ValueError(
                f"ReLA prediction {index} mask shape mismatch: "
                f"{prediction.shape} vs {target.shape}."
            )

        pred_nt = bool(_scalar(sample.get("pred_nt", False)))
        gt_nt = bool(_scalar(sample.get("gt_nt", False)))
        if respect_pred_nt and pred_nt:
            prediction = np.zeros_like(prediction, dtype=bool)
        predicted_no_target += int(pred_nt)
        ground_truth_no_target += int(gt_nt)
        empty_predictions += int(not prediction.any())

        paths = artifact_paths(output_dir, index)
        atomic_save_mask(paths["mask"], prediction)
        atomic_save_mask(paths["gt"], target)
        sentence = sample.get("sent", "")
        sent_info = sample.get("sent_info")
        instance_id = f"{image_id}:{index}"
        if isinstance(sent_info, dict):
            for key in ("sent_id", "id"):
                if sent_info.get(key) is not None:
                    instance_id = f"{image_id}:{sent_info[key]}"
                    break
        rows.append(
            {
                "name": f"rela_{index:08d}",
                "method": method,
                "split": split,
                "instance_id": instance_id,
                "image": str(image),
                "gt_mask": str(paths["gt"]),
                "prediction": str(paths["mask"]),
                "prediction_kind": "mask",
                "threshold": 0.5,
                "no_target": gt_nt,
                "query": str(sentence),
                "predicted_no_target": pred_nt,
                "protocol": "rela_official_refer_evaluator_prediction_import",
            }
        )

    if not rows:
        raise ValueError("No ReLA predictions were converted.")
    manifest = output_dir / "manifest.jsonl"
    atomic_write_jsonl(manifest, rows)
    report = {
        "source": "rela_official_refer_evaluator",
        "samples": len(rows),
        "split": split,
        "method": method,
        "respect_predicted_no_target": respect_pred_nt,
        "predicted_no_target": predicted_no_target,
        "ground_truth_no_target": ground_truth_no_target,
        "empty_predictions": empty_predictions,
        "manifest": str(manifest.resolve()),
    }
    atomic_write_json(output_dir / "import_summary.json", report)
    return report


def main() -> int:
    args = parse_args()
    import torch

    input_path = args.input_pth.expanduser().resolve()
    loaded = torch.load(input_path, map_location="cpu", weights_only=False)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report = convert_predictions(
        prediction_rows(loaded),
        output_dir=output_dir,
        image_root=args.image_root.expanduser().resolve(),
        split=args.split,
        method=args.method,
        respect_pred_nt=not args.ignore_pred_nt,
    )
    print(json.dumps(report, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
