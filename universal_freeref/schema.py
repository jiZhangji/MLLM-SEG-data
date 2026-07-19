from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PREDICTION_KINDS = {"mask", "probability", "logits"}


@dataclass(frozen=True)
class ManifestItem:
    name: str
    method: str
    split: str
    image: Path
    gt_mask: Path
    prediction: Path
    prediction_kind: str = "mask"
    ignore_mask: Path | None = None
    uncertainty: Path | None = None
    array_key: str | None = None
    uncertainty_key: str | None = None
    foreground_channel: int = 1
    threshold: float = 0.5
    no_target: bool = False
    instance_id: str | None = None

    @classmethod
    def from_mapping(cls, value: dict[str, Any], manifest_path: Path, line_number: int) -> "ManifestItem":
        def required(key: str) -> Any:
            if key not in value or value[key] in (None, ""):
                raise ValueError(f"Manifest line {line_number} is missing {key!r}.")
            return value[key]

        def resolve(raw: Any) -> Path:
            path = Path(str(raw)).expanduser()
            return path if path.is_absolute() else (manifest_path.parent / path).resolve()

        prediction_value = value.get("prediction")
        prediction_kind = str(value.get("prediction_kind", "")).lower()
        if prediction_value in (None, ""):
            if value.get("pred_mask") not in (None, ""):
                prediction_value = value["pred_mask"]
                prediction_kind = prediction_kind or "mask"
            elif value.get("probability") not in (None, ""):
                prediction_value = value["probability"]
                prediction_kind = prediction_kind or "probability"
        if prediction_value in (None, ""):
            raise ValueError(
                f"Manifest line {line_number} needs prediction, pred_mask, or probability."
            )
        prediction_kind = prediction_kind or "mask"
        if prediction_kind not in PREDICTION_KINDS:
            raise ValueError(
                f"Manifest line {line_number} has prediction_kind={prediction_kind!r}; "
                f"expected one of {sorted(PREDICTION_KINDS)}."
            )
        threshold = float(value.get("threshold", 0.5))
        if not 0.0 < threshold < 1.0:
            raise ValueError(f"Manifest line {line_number} threshold must lie in (0, 1).")

        image_value = required("image")
        gt_value = value.get("gt_mask", value.get("mask"))
        if gt_value in (None, ""):
            raise ValueError(f"Manifest line {line_number} is missing 'gt_mask'.")
        ignore_value = value.get("ignore_mask")
        uncertainty_value = value.get("uncertainty")
        return cls(
            name=str(value.get("name") or f"sample_{line_number - 1:08d}"),
            method=str(value.get("method") or "unknown"),
            split=str(value.get("split") or "unknown"),
            image=resolve(image_value),
            gt_mask=resolve(gt_value),
            prediction=resolve(prediction_value),
            prediction_kind=prediction_kind,
            ignore_mask=resolve(ignore_value) if ignore_value not in (None, "") else None,
            uncertainty=resolve(uncertainty_value) if uncertainty_value not in (None, "") else None,
            array_key=str(value["array_key"]) if value.get("array_key") not in (None, "") else None,
            uncertainty_key=(
                str(value["uncertainty_key"])
                if value.get("uncertainty_key") not in (None, "")
                else None
            ),
            foreground_channel=int(value.get("foreground_channel", 1)),
            threshold=threshold,
            no_target=bool(value.get("no_target", False)),
            instance_id=str(value["instance_id"]) if value.get("instance_id") is not None else None,
        )


def load_manifest(path: Path) -> list[ManifestItem]:
    path = path.expanduser().resolve()
    items: list[ManifestItem] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"Manifest line {line_number} is not a JSON object.")
        items.append(ManifestItem.from_mapping(value, path, line_number))
    if not items:
        raise ValueError(f"Manifest is empty: {path}")
    return items
