from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from universal_freeref import summarize_polyformer
from universal_freeref.export_polyformer_masks import (
    _ordered_rows,
    _polygons_from_generation,
    _rasterize,
)


ROOT = Path(__file__).resolve().parents[1]


def test_polyformer_generation_is_rasterized_without_coordinate_quantization() -> None:
    generation = [0.1, 0.1, 0.9, 0.9, 0.2, 0.2, 0.8, 0.2, 0.8, 0.8, 0.2, 0.8, -1]
    box, polygons = _polygons_from_generation(generation, 100)
    assert np.allclose(box, [0.1, 0.1, 0.9, 0.9])
    assert len(polygons) == 1
    assert np.allclose(polygons[0][0], [20.0, 20.0])
    mask = _rasterize(polygons, (100, 100))
    assert mask[50, 50]
    assert not mask[5, 5]


def test_polyformer_manifest_follows_official_tsv_order(tmp_path: Path) -> None:
    tsv = tmp_path / "split.tsv"
    tsv.write_text("b\tdata\na\tdata\n", encoding="utf-8")
    rows = [{"instance_id": "a"}, {"instance_id": "b"}]
    assert [row["instance_id"] for row in _ordered_rows(rows, tsv)] == ["b", "a"]


def test_polyformer_summary_checks_paired_baseline_identity(tmp_path: Path, monkeypatch) -> None:
    export_path = tmp_path / "export.json"
    refine_path = tmp_path / "refine.json"
    output = tmp_path / "comparison.md"
    export_path.write_text(
        json.dumps({"samples": 2, "coarse_mean_iou": 0.5}), encoding="utf-8"
    )
    refine_path.write_text(
        json.dumps(
            {
                "samples": 2,
                "coarse_mean_iou": 0.5,
                "refined_mean_iou": 0.55,
                "coarse_cIoU": 0.52,
                "refined_cIoU": 0.57,
                "coarse_boundary_iou": 0.1,
                "refined_boundary_iou": 0.2,
                "improved_samples": 1,
                "degraded_samples": 1,
                "unchanged_samples": 0,
                "delta_ci95_low": -0.01,
                "delta_ci95_high": 0.11,
                "wilcoxon_p": 0.5,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "summarize_polyformer",
            "--export-summary",
            str(export_path),
            "--freeref-summary",
            str(refine_path),
            "--output",
            str(output),
            "--paper-miou",
            "78.49",
        ],
    )
    assert summarize_polyformer.main() == 0
    text = output.read_text(encoding="utf-8")
    assert "50.00 | 55.00 | +5.00" in text
    assert "Paper full-split mIoU reference: 78.49" in text


def test_polyformer_download_and_inference_are_separate() -> None:
    asset_script = (ROOT / "prepare_polyformer_freeref_assets.sh").read_text(encoding="utf-8")
    run_script = (ROOT / "run_polyformer_freeref_smoke.sh").read_text(encoding="utf-8")
    assert "download_missing_method_weights.sh" in asset_script
    assert "prepare_polyformer_freeref_env.sh" in asset_script
    assert "69fc728b2ec6a2b3595ec34db64074badcb19151" in asset_script
    assert "download_missing_method_weights.sh" not in run_script
    assert "prepare_polyformer_freeref_env.sh" not in run_script
    assert "polyformer_l_refcoco.pt" in run_script
    assert "--paper-miou 78.49" in run_script
