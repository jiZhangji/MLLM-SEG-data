from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from training_free_refine.data import ReferringSegDataset
from universal_freeref.export_lisa_masks import (
    _artifact_paths,
    _atomic_save_logits,
    _atomic_save_mask,
    _manifest_row,
    chunked,
    group_record_positions,
    lisa_question,
)
from universal_freeref.export_lisa_freeref_prompt import (
    probability_to_sam_mask_input,
)
from universal_freeref.evaluate_lisa_freeref_prompt import summarize_rows
from universal_freeref.io import load_probability
from universal_freeref.schema import ManifestItem


ROOT = Path(__file__).resolve().parents[1]


def _save_image(path: Path, rgb: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (12, 8), rgb).save(path)


def _save_mask(path: Path, left: int) -> None:
    values = np.zeros((8, 12), dtype=np.uint8)
    values[2:7, left : left + 3] = 255
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(values, mode="L").save(path)


def _dataset(tmp_path: Path) -> ReferringSegDataset:
    image_a = tmp_path / "images" / "a.jpg"
    image_b = tmp_path / "images" / "b.jpg"
    _save_image(image_a, (10, 120, 20))
    _save_image(image_b, (120, 10, 20))
    rows = []
    for index, (image, expression) in enumerate(
        ((image_a, "left object"), (image_b, "middle object"), (image_a, "right object"))
    ):
        target = tmp_path / "masks" / f"{index}.png"
        _save_mask(target, 1 + index)
        rows.append(
            {
                "id": f"item-{index}",
                "images": [str(image)],
                "masks": [str(target)],
                "conversations": [
                    {
                        "from": "human",
                        "content": f'Please segment the object this sentence describes: "{expression}"',
                    }
                ],
            }
        )
    path = tmp_path / "eval.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    return ReferringSegDataset(path, data_root=tmp_path)


def test_lisa_groups_repeated_images_and_chunks_without_reordering(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    assert group_record_positions(dataset) == [[0, 2], [1]]
    assert list(chunked([0, 2, 4], 2)) == [[0, 2], [4]]
    assert list(chunked([0, 2, 4], 0)) == [[0, 2, 4]]
    assert lisa_question(" Red CAR ") == (
        "What is red car in this image? Please output segmentation mask."
    )


def test_lisa_manifest_preserves_soft_logits_for_freeref(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    output = tmp_path / "lisa"
    paths = _artifact_paths(output, 0)
    logits = np.full((8, 12), -4.0, dtype=np.float32)
    logits[2:7, 1:4] = 4.0
    _atomic_save_logits(paths["logits"], logits)
    _atomic_save_mask(paths["mask"], logits > 0)
    _atomic_save_mask(paths["gt"], dataset[0].mask.numpy())

    row = _manifest_row(dataset, 0, output, "LISA-7B-v1", "refcoco_testA")
    assert row["prediction_kind"] == "logits"
    assert row["array_key"] == "logits"
    item = ManifestItem.from_mapping(row, tmp_path / "manifest.jsonl", 1)
    probability, uncertainty = load_probability(item, (8, 12))
    assert probability.shape == (8, 12)
    assert float(probability[3, 2]) > 0.98
    assert float(probability[0, 0]) < 0.02
    assert uncertainty is None


def test_lisa_runner_is_local_resumable_and_invokes_paired_evaluation() -> None:
    text = (ROOT / "run_lisa_freeref_eval.sh").read_text(encoding="utf-8")
    assert "HF_HUB_OFFLINE=1" in text
    assert "TRANSFORMERS_OFFLINE=1" in text
    assert "models/freeref_missing_methods" in text
    assert "universal_freeref.export_lisa_masks" in text
    assert "universal_freeref.evaluate" in text
    assert "universal_freeref.summarize" in text
    assert "refcoco_testA" in text


class _IdentityTransform:
    def apply_image(self, image: np.ndarray) -> np.ndarray:
        return image


def test_lisa_freeref_probability_becomes_padded_sam_logit_prompt() -> None:
    probability = np.asarray([[0.25, 0.5, 0.75], [0.1, 0.9, 0.6]], dtype=np.float32)
    prompt = probability_to_sam_mask_input(
        probability,
        _IdentityTransform(),
        (4, 5),
    )
    assert prompt.shape == (4, 5)
    assert np.allclose(prompt[:2, :3], np.log(probability / (1.0 - probability)))
    assert np.count_nonzero(prompt[2:, :]) == 0
    assert np.count_nonzero(prompt[:, 3:]) == 0


def test_lisa_freeref_prompt_summary_reports_paired_metrics_and_timing() -> None:
    rows = [
        {
            "baseline_iou": 0.5,
            "prompted_iou": 0.7,
            "iou_delta": 0.2,
            "baseline_boundary_iou": 0.2,
            "prompted_boundary_iou": 0.4,
            "base_seconds": 0.25,
            "freeref_seconds": 0.10,
            "second_decoder_seconds": 0.02,
            "total_seconds": 0.38,
            "_baseline_intersection": 5,
            "_baseline_union": 10,
            "_prompted_intersection": 7,
            "_prompted_union": 10,
        },
        {
            "baseline_iou": 0.8,
            "prompted_iou": 0.7,
            "iou_delta": -0.1,
            "baseline_boundary_iou": 0.5,
            "prompted_boundary_iou": 0.6,
            "base_seconds": 0.35,
            "freeref_seconds": 0.20,
            "second_decoder_seconds": 0.04,
            "total_seconds": 0.60,
            "_baseline_intersection": 8,
            "_baseline_union": 10,
            "_prompted_intersection": 7,
            "_prompted_union": 10,
        },
    ]
    summary = summarize_rows(rows, bootstrap_samples=0, seed=0)
    assert summary["samples"] == 2
    assert np.isclose(summary["baseline_mean_iou"], 0.65)
    assert np.isclose(summary["prompted_mean_iou"], 0.7)
    assert np.isclose(summary["baseline_cIoU"], 0.65)
    assert np.isclose(summary["prompted_cIoU"], 0.7)
    assert summary["improved_samples"] == 1
    assert summary["degraded_samples"] == 1
    assert np.isclose(summary["base_seconds_per_sample"], 0.3)
    assert np.isclose(summary["total_seconds_per_sample"], 0.49)


def test_lisa_freeref_prompt_runner_is_offline_and_defaults_to_smoke_test() -> None:
    text = (ROOT / "run_lisa_freeref_prompt_eval.sh").read_text(encoding="utf-8")
    assert "HF_HUB_OFFLINE=1" in text
    assert "TRANSFORMERS_OFFLINE=1" in text
    assert "LISA_PROMPT_LIMIT:-16" in text
    assert "universal_freeref.export_lisa_freeref_prompt" in text
    assert "universal_freeref.evaluate_lisa_freeref_prompt" in text
