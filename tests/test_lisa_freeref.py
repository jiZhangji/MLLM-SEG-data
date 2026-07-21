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
from universal_freeref.eval_lisa_official_freeref_sam import (
    PROTOCOL as LISA_OFFICIAL_FREEREF_SAM_PROTOCOL,
    summarize_rows as summarize_official_freeref_sam_rows,
)
from universal_freeref.eval_lisa_paper_protocol import (
    EXPECTED_EXPRESSIONS,
    PAPER_CIOU,
    official_lisa_metric_totals,
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
    assert row["data_protocol"] == "stamp_flat_json_not_paper_reproduction"
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


def test_lisa_official_metric_matches_foreground_ciou_and_mean_iou() -> None:
    predictions = [
        np.asarray([[1, 1], [0, 0]], dtype=bool),
        np.asarray([[0, 0], [0, 0]], dtype=bool),
    ]
    targets = [
        np.asarray([[1, 0], [0, 0]], dtype=bool),
        np.asarray([[0, 0], [0, 0]], dtype=bool),
    ]
    totals = official_lisa_metric_totals(predictions, targets)
    assert totals["intersection"] == 1
    assert totals["union"] == 2
    assert np.isclose(totals["giou_sum"] / totals["count"], 0.75)
    assert np.isclose(totals["intersection"] / totals["union"], 0.5)


def test_lisa_paper_protocol_tracks_the_reported_rows_and_full_counts() -> None:
    assert PAPER_CIOU["base"]["refcoco|unc|testA"] == 76.5
    assert PAPER_CIOU["finetuned_referseg"]["refcoco|unc|testA"] == 79.1
    assert EXPECTED_EXPRESSIONS["refcoco|unc|testA"] == 5657
    assert EXPECTED_EXPRESSIONS["refcocog|umd|test"] == 9602


def test_lisa_paper_runner_is_a_baseline_only_reproduction_gate() -> None:
    text = (ROOT / "run_lisa_paper_reproduction.sh").read_text(encoding="utf-8")
    assert "eval_lisa_paper_protocol" in text
    assert "LISA_REQUIRE_PAPER_MATCH:-1" in text
    assert "HF_HUB_OFFLINE=1" in text
    assert "universal_freeref.evaluate" not in text
    assert "FreeRef is intentionally disabled" in text


def test_lisa_paper_data_preparation_uses_links_without_downloads() -> None:
    text = (ROOT / "prepare_lisa_paper_data.sh").read_text(encoding="utf-8")
    assert "ln -s" in text
    assert "data/lisa_paper_refer_seg" in text
    assert "curl " not in text
    assert "wget " not in text


def test_lisa_paper_freeref_runner_keeps_the_reproduction_gate() -> None:
    text = (ROOT / "run_lisa_paper_freeref_eval.sh").read_text(encoding="utf-8")
    assert "LISA_REQUIRE_PAPER_MATCH=1" in text
    assert "paper_match" in text
    assert "universal_freeref.evaluate" in text
    assert "waiting 10 seconds" in text


def test_four_gpu_runner_assigns_parallel_work_by_gpu_type() -> None:
    text = (ROOT / "run_freeref_remaining_4gpu.sh").read_text(encoding="utf-8")
    assert "H200_GPUS" in text
    assert "H100_GPUS" in text
    assert "TEXT4SEG_H200_PARALLEL_JOBS:-5" in text
    assert "STAMP_H200_PARALLEL_JOBS:-6" in text
    assert "LISA_H100_PARALLEL_JOBS:-2" in text
    assert "prepare_lisa_paper_data.sh" in text


def test_all_experiment_status_script_covers_the_three_model_families() -> None:
    text = (ROOT / "check_freeref_all_status.sh").read_text(encoding="utf-8")
    assert "check_frozen_samh_status.sh" in text
    assert "check_text4seg_public_p24_status.sh" in text
    assert "check_lisa_paper_freeref_status.sh" in text
    assert "check_lisa_official_freeref_sam_status.sh" in text


def test_lisa_official_freeref_sam_summary_contains_all_four_paired_branches() -> None:
    rows = []
    for baseline, freeref, baseline_sam, freeref_sam in (
        (0.5, 0.6, 0.7, 0.8),
        (0.4, 0.5, 0.6, 0.7),
    ):
        row = {}
        for branch, iou in (
            ("baseline", baseline),
            ("freeref", freeref),
            ("baseline_sam", baseline_sam),
            ("freeref_sam", freeref_sam),
        ):
            row[f"{branch}_iou"] = iou
            row[f"{branch}_boundary_iou"] = iou
            row[f"_{branch}_intersection"] = int(iou * 10)
            row[f"_{branch}_union"] = 10
        rows.append(row)
    summary = summarize_official_freeref_sam_rows(rows, bootstrap_samples=0, seed=0)
    assert summary["samples"] == 2
    assert np.isclose(summary["baseline_mean_iou"], 0.45)
    assert np.isclose(summary["freeref_sam_mean_iou"], 0.75)
    assert summary["freeref_sam_improved_samples"] == 2


def test_lisa_h100_runner_uses_official_refer_and_freeref_before_second_sam() -> None:
    runner = (ROOT / "run_lisa_official_freeref_sam_h100.sh").read_text(encoding="utf-8")
    evaluator = (ROOT / "universal_freeref/eval_lisa_official_freeref_sam.py").read_text(
        encoding="utf-8"
    )
    assert "LISA_H100_PARALLEL_PER_GPU:-2" in runner
    assert "eval_lisa_official_freeref_sam" in runner
    assert "ValDataset" in evaluator
    assert "baseline_sam" in evaluator
    assert "freeref_sam" in evaluator
    assert LISA_OFFICIAL_FREEREF_SAM_PROTOCOL in evaluator
