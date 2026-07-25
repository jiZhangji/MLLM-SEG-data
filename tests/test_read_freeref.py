from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from universal_freeref.read_export import READOfficialExporter
from universal_freeref.schema import load_manifest


ROOT = Path(__file__).resolve().parents[1]


def test_read_exporter_saves_logits_and_metrics(tmp_path: Path) -> None:
    image = tmp_path / "image.jpg"
    Image.new("RGB", (5, 4), color=(30, 20, 10)).save(image)
    exporter = READOfficialExporter(tmp_path / "out", "READ-test", "refcoco_testA")
    exporter.record(
        {
            "image_paths": [str(image)],
            "conversation_list": ["query one", "query two"],
            "ref_ids": [[11, 12]],
            "sent_ids": [[1, 2]],
        },
        {
            "pred_masks": [
                np.asarray(
                    [
                        [[[2.0, -1.0], [2.0, -1.0]]],
                        [[[-2.0, -2.0], [-2.0, -2.0]]],
                    ]
                )
            ],
            "gt_masks": [
                np.asarray(
                    [
                        [[1, 0], [1, 1]],
                        [[0, 0], [0, 0]],
                    ]
                )
            ],
        },
    )
    report = exporter.finalize()

    assert report["samples"] == 2
    assert report["official_cIoU"] == 2 / 3
    assert report["official_gIoU"] == (2 / 3 + 1.0) / 2
    items = load_manifest(tmp_path / "out" / "manifest.jsonl")
    assert items[0].prediction_kind == "logits"
    assert items[0].instance_id == "11:1"
    assert items[1].no_target is True
    with np.load(items[0].prediction) as archive:
        assert archive["logits"].shape == (2, 2)
    summary = json.loads((tmp_path / "out" / "export_summary.json").read_text())
    assert summary["source"] == "read_official_teacher_forced_validation"


def test_read_adapter_forces_public_checkpoint_to_local_vision_tower() -> None:
    adapter = (ROOT / "universal_freeref" / "export_read_masks.py").read_text()
    assert "model_config.vision_tower = str(vision_tower_path)" in adapter
    assert "model_config.mm_vision_tower = str(vision_tower_path)" in adapter
    assert "config=model_config" in adapter
