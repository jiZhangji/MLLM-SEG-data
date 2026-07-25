from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from universal_freeref.import_rela_outputs import (
    convert_predictions,
    image_candidates,
    normalize_image_id,
    prediction_rows,
)
from universal_freeref.schema import load_manifest


ROOT = Path(__file__).resolve().parents[1]


def test_normalize_image_id_and_candidates(tmp_path: Path) -> None:
    assert normalize_image_id(np.int64(42)) == "42"
    paths = image_candidates(tmp_path, "42")
    assert tmp_path / "COCO_train2014_000000000042.jpg" in paths


def test_prediction_rows_accepts_official_list_and_wrappers() -> None:
    rows = [{"img_id": 1}]
    assert prediction_rows(rows) is rows
    assert prediction_rows({"predictions": rows}) is rows


def test_convert_rela_predictions_builds_valid_manifest(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    image_path = image_root / "COCO_train2014_000000000042.jpg"
    Image.new("RGB", (4, 4), color=(20, 40, 60)).save(image_path)
    output_dir = tmp_path / "converted"
    predictions = [
        {
            "img_id": 42,
            "sent": "the object",
            "sent_info": {"raw": "the object", "sent_id": 9},
            "pred_nt": False,
            "gt_nt": False,
            "pred_mask": np.asarray([[0, 1], [0, 1]], dtype=np.uint8),
            "gt_mask": np.asarray([[0, 1], [1, 1]], dtype=np.uint8),
        },
        {
            "img_id": 42,
            "sent": "nothing",
            "pred_nt": True,
            "gt_nt": True,
            "pred_mask": np.ones((2, 2), dtype=np.uint8),
            "gt_mask": np.zeros((2, 2), dtype=np.uint8),
        },
    ]
    report = convert_predictions(
        predictions,
        output_dir=output_dir,
        image_root=image_root,
        split="refcoco_val",
        method="ReLA-test",
        respect_pred_nt=True,
    )

    assert report["samples"] == 2
    assert report["predicted_no_target"] == 1
    assert report["empty_predictions"] == 1
    items = load_manifest(output_dir / "manifest.jsonl")
    assert len(items) == 2
    assert items[0].instance_id == "42:9"
    assert items[1].no_target is True
    saved = np.asarray(Image.open(items[1].prediction))
    assert int(saved.sum()) == 0
    summary = json.loads((output_dir / "import_summary.json").read_text())
    assert summary["source"] == "rela_official_refer_evaluator"


def test_rela_hook_repairs_official_distributed_evaluator_import() -> None:
    hook = (ROOT / "universal_freeref" / "rela_hook" / "sitecustomize.py").read_text()
    assert "refer_evaluation.itertools = itertools" in hook
