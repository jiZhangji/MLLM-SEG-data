from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from universal_freeref.gsva_export import GSVAOfficialExporter
from universal_freeref.schema import load_manifest


ROOT = Path(__file__).resolve().parents[1]


def test_gsva_exporter_flattens_per_image_expressions(tmp_path: Path) -> None:
    image = tmp_path / "image.jpg"
    Image.new("RGB", (4, 3), color=(10, 20, 30)).save(image)
    exporter = GSVAOfficialExporter(tmp_path / "out", "GSVA-test", "refcoco_val")
    exporter.record(
        {"image_paths": [str(image)]},
        {
            "pred_masks": [
                np.asarray(
                    [
                        [[-1.0, 2.0], [-1.0, 2.0]],
                        [[-2.0, -2.0], [-2.0, -2.0]],
                    ]
                )
            ],
            "gt_masks": [
                np.asarray(
                    [
                        [[0, 1], [1, 1]],
                        [[0, 0], [0, 0]],
                    ]
                )
            ],
        },
    )
    report = exporter.finalize((0.7, 0.8))

    assert report["samples"] == 2
    assert report["empty_predictions"] == 1
    assert report["no_target_samples"] == 1
    assert report["official_return_metrics"] == [0.7, 0.8]
    items = load_manifest(tmp_path / "out" / "manifest.jsonl")
    assert len(items) == 2
    assert items[0].image == image.resolve()
    assert items[1].no_target is True
    saved = np.asarray(Image.open(tmp_path / "out" / "pred_masks" / "00000000.png"))
    assert saved.tolist() == [[0, 255], [0, 255]]
    assert items[0].prediction_kind == "logits"
    summary = json.loads((tmp_path / "out" / "export_summary.json").read_text())
    assert summary["source"] == "gsva_official_validate"


def test_gsva_runners_require_atomic_legacy_merge_marker() -> None:
    merge = (ROOT / "prepare_gsva_llava_legacy.sh").read_text()
    split = (ROOT / "run_gsva_freeref_split.sh").read_text()
    orchestrator = (ROOT / "run_remaining_six_experiments_4gpu.sh").read_text()
    assert 'touch "${MERGE_MARKER}"' in merge
    assert "${MODEL_PATH}/.freeref_merge_complete" in split
    assert 'PRECISION="${GSVA_PRECISION:-fp32}"' in split
    assert "${gsva_model}/.freeref_merge_complete" in orchestrator


def test_gsva_data_prep_includes_auxiliary_grefcoco_required_by_official_main() -> None:
    prep = (ROOT / "prepare_gsva_freeref_data.sh").read_text()
    assert "GSVA_GREF_SOURCE_ROOT" in prep
    assert "grefs(unc).p" in prep and "grefs(unc).json" in prep
    assert '"${TARGET}/grefcoco"' in prep
