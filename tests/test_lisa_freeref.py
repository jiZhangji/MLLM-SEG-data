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
