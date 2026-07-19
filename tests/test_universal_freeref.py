from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from universal_freeref import build_manifest, evaluate, summarize
from universal_freeref.io import load_probability
from universal_freeref.schema import ManifestItem, load_manifest


def save_rgb(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(values.astype(np.uint8), mode="RGB").save(path)


def save_mask(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(values.astype(np.uint8) * 255, mode="L").save(path)


def make_sample(root: Path, name: str = "sample") -> dict[str, Path]:
    height = width = 32
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :16] = (40, 160, 60)
    image[:, 16:] = (180, 50, 50)
    target = np.zeros((height, width), dtype=bool)
    target[6:26, 4:15] = True
    prediction = np.zeros_like(target)
    prediction[8:24, 6:15] = True
    paths = {
        "image": root / "images" / f"{name}.png",
        "target": root / "gt" / f"{name}.png",
        "prediction": root / "pred" / f"{name}_pred.png",
    }
    save_rgb(paths["image"], image)
    save_mask(paths["target"], target)
    save_mask(paths["prediction"], prediction)
    return paths


def test_schema_accepts_legacy_prediction_keys(tmp_path: Path) -> None:
    paths = make_sample(tmp_path)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "name": "legacy",
                "method": "test",
                "split": "val",
                "image": str(paths["image"]),
                "gt_mask": str(paths["target"]),
                "pred_mask": str(paths["prediction"]),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    item = load_manifest(manifest)[0]
    assert item.prediction_kind == "mask"
    assert item.prediction == paths["prediction"]


def test_two_channel_logits_are_converted_to_probability(tmp_path: Path) -> None:
    paths = make_sample(tmp_path)
    logits = np.zeros((2, 16, 16), dtype=np.float32)
    logits[0] = -2.0
    logits[1] = 2.0
    logits_path = tmp_path / "logits.npy"
    np.save(logits_path, logits)
    item = ManifestItem(
        name="soft",
        method="test",
        split="val",
        image=paths["image"],
        gt_mask=paths["target"],
        prediction=logits_path,
        prediction_kind="logits",
        foreground_channel=1,
    )
    probability, uncertainty = load_probability(item, (32, 32))
    assert probability.shape == (32, 32)
    assert float(probability.min()) > 0.95
    assert uncertainty is None


def test_build_manifest_and_paired_evaluation(tmp_path: Path, monkeypatch) -> None:
    paths = make_sample(tmp_path)
    manifest = tmp_path / "manifests" / "test.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_manifest",
            "--method",
            "Synthetic",
            "--split",
            "val",
            "--prediction-root",
            str(paths["prediction"].parent),
            "--image-root",
            str(paths["image"].parent),
            "--gt-root",
            str(paths["target"].parent),
            "--prediction-glob",
            "*_pred.png",
            "--strip-suffix",
            "_pred",
            "--image-template",
            "{stem}.png",
            "--gt-template",
            "{stem}.png",
            "--output",
            str(manifest),
        ],
    )
    assert build_manifest.main() == 0
    assert load_manifest(manifest)[0].method == "Synthetic"

    output = tmp_path / "evaluation"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate",
            "--manifest",
            str(manifest),
            "--output-dir",
            str(output),
            "--n-segments",
            "16",
            "--save-visualizations",
            "1",
            "--bootstrap-samples",
            "20",
        ],
    )
    assert evaluate.main() == 0
    summary = json.loads((output / "eval_summary.json").read_text(encoding="utf-8"))
    assert summary["samples"] == 1
    assert 0.0 <= summary["coarse_mean_iou"] <= 1.0
    assert 0.0 <= summary["refined_mean_iou"] <= 1.0
    assert summary["coarse_gIoU"] == summary["coarse_mean_iou"]
    assert (output / "visualizations" / "00000000_sample.png").is_file()
    assert (output / "refined_masks" / "00000000_sample.png").is_file()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "summarize",
            "--summary",
            f"Synthetic-val={output / 'eval_summary.json'}",
            "--output-dir",
            str(tmp_path / "combined"),
        ],
    )
    assert summarize.main() == 0
    assert "Synthetic-val" in (tmp_path / "combined" / "comparison.md").read_text(encoding="utf-8")


def test_empty_hard_prediction_is_preserved(tmp_path: Path, monkeypatch) -> None:
    paths = make_sample(tmp_path, "empty")
    save_mask(paths["prediction"], np.zeros((32, 32), dtype=bool))
    manifest = tmp_path / "empty.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "name": "empty",
                "method": "Synthetic",
                "split": "val",
                "image": str(paths["image"]),
                "gt_mask": str(paths["target"]),
                "prediction": str(paths["prediction"]),
                "prediction_kind": "mask",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "empty_eval"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate",
            "--manifest",
            str(manifest),
            "--output-dir",
            str(output),
            "--n-segments",
            "16",
            "--save-visualizations",
            "0",
            "--bootstrap-samples",
            "0",
        ],
    )
    assert evaluate.main() == 0
    coarse = np.asarray(Image.open(output / "coarse_masks" / "00000000_empty.png")) > 0
    refined = np.asarray(Image.open(output / "refined_masks" / "00000000_empty.png")) > 0
    assert not coarse.any()
    assert np.array_equal(coarse, refined)
