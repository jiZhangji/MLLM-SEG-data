from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from universal_freeref import build_eval_json_manifest, summarize_paper_suite
from universal_freeref.export_pixellm_masks import preprocess_square
from universal_freeref.export_utils import merge_binary_logits
from universal_freeref.import_segagent_outputs import reported_iou, select_prediction
from universal_freeref.run_segagent_official import FullEvaluationList


ROOT = Path(__file__).resolve().parents[1]


def _save_image(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def test_probabilistic_union_preserves_spatial_union() -> None:
    logits = np.full((2, 4, 4), -10.0, dtype=np.float32)
    logits[0, :2, :2] = 10.0
    logits[1, 2:, 2:] = 10.0
    merged = merge_binary_logits(logits)
    assert merged.shape == (4, 4)
    assert np.all(merged[:2, :2] > 0)
    assert np.all(merged[2:, 2:] > 0)
    assert merged[0, 3] < 0


def test_pixellm_square_preprocess_pads_without_resizing() -> None:
    image = np.zeros((3, 5, 7), dtype=np.float32)
    result = preprocess_square(__import__("torch").from_numpy(image), 8)
    assert tuple(result.shape) == (3, 8, 8)


def test_segagent_selection_matches_declared_protocols() -> None:
    sample = {
        "pred_list": [
            {"mask": "first", "outputs": "Current IOU: 0.10, Next"},
            {"mask": "second", "outputs": "Current IOU: 0.90, Finish"},
            {"mask": "final", "outputs": None},
        ]
    }
    assert select_prediction(sample, "final", -1) == ("final", 2)
    assert select_prediction(sample, "official-reported-best", -1) == ("first", 0)
    assert reported_iou("Current IOU: 0.75, Positive point") == 0.75


def test_segagent_wrapper_removes_official_5000_item_cap() -> None:
    values = FullEvaluationList(list(range(6001)))
    assert len(values[:5000]) == 6001
    limited = FullEvaluationList(list(range(100)), offset=7, limit=11)
    assert limited[:5000] == list(range(7, 18))


def test_external_prediction_manifest_aligns_flat_eval_json(tmp_path: Path, monkeypatch) -> None:
    image = tmp_path / "image.png"
    target = tmp_path / "target.png"
    prediction = tmp_path / "predictions" / "00000000.png"
    _save_image(image, np.zeros((8, 8, 3), dtype=np.uint8))
    _save_image(target, np.pad(np.ones((4, 4), dtype=np.uint8) * 255, 2))
    _save_image(prediction, np.pad(np.ones((3, 3), dtype=np.uint8) * 255, ((2, 3), (2, 3))))
    eval_json = tmp_path / "eval.json"
    eval_json.write_text(
        json.dumps(
            [
                {
                    "id": "sample",
                    "image": str(image),
                    "mask": str(target),
                    "messages": [{"role": "user", "content": 'Please segment "object".'}],
                }
            ]
        ),
        encoding="utf-8",
    )
    output = tmp_path / "import"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_eval_json_manifest",
            "--method",
            "External",
            "--split",
            "val",
            "--eval-json",
            str(eval_json),
            "--data-root",
            str(tmp_path),
            "--prediction-root",
            str(prediction.parent),
            "--output-dir",
            str(output),
        ],
    )
    assert build_eval_json_manifest.main() == 0
    row = json.loads((output / "manifest.jsonl").read_text(encoding="utf-8"))
    assert row["method"] == "External"
    assert Path(row["gt_mask"]).is_file()


def test_paper_summary_requires_complete_split_counts(tmp_path: Path, monkeypatch) -> None:
    summary_dir = tmp_path / "outputs" / "training_free_refine_stamp7b_refcoco_val_full"
    summary_dir.mkdir(parents=True)
    (summary_dir / "eval_summary.json").write_text(
        json.dumps(
            {
                "samples": 10834,
                "coarse_mean_iou": 0.7,
                "refined_mean_iou": 0.72,
                "mean_iou_delta": 0.02,
                "coarse_cIoU": 0.71,
                "refined_cIoU": 0.73,
                "cIoU_delta": 0.02,
                "source": "synthetic",
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "paper"
    monkeypatch.setattr(
        sys,
        "argv",
        ["summarize_paper_suite", "--root", str(tmp_path), "--output-dir", str(output)],
    )
    assert summarize_paper_suite.main() == 0
    text = (output / "paper_results.md").read_text(encoding="utf-8")
    assert "STAMP-7B | Qwen2-7B | + FreeRef | 72.00" in text
    assert "STAMP-7B | refcoco_val | 10834 | 10834 | complete" in text


def test_paper_scripts_keep_download_and_inference_separate() -> None:
    asset_script = (ROOT / "prepare_freeref_paper_assets.sh").read_text(encoding="utf-8")
    run_script = (ROOT / "run_freeref_paper_suite.sh").read_text(encoding="utf-8")
    assert "download_missing_method_weights.sh" in asset_script
    assert "download_missing_method_weights.sh" not in run_script
    assert "run_pixellm_freeref_full_eval.sh" in run_script
    assert "run_segagent_freeref_full_eval.sh" in run_script


def test_segagent_environment_setup_is_isolated_from_inference() -> None:
    script = (ROOT / "prepare_segagent_freeref_env.sh").read_text(encoding="utf-8")
    assert "--clone" in script
    assert "opencv-python-headless==4.10.0.84" in script
    assert "segment-anything==1.0" in script
    assert "run_segagent_freeref_full_eval.sh" not in script
    assert "download_missing_method_weights.sh" not in script
