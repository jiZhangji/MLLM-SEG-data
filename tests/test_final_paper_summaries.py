from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from efficiency_benchmark.summarize_k_sweep import main as summarize_k_main
from training_free_refine.summarize_full_postprocess import main as summarize_postprocess_main
from training_free_refine.summarize_sam_full import main as summarize_sam_main


SPLITS = (
    ("refcoco_val", 10834),
    ("refcoco_testA", 5657),
    ("refcoco_testB", 5095),
    ("refcoco+_val", 10758),
    ("refcoco+_testA", 5726),
    ("refcoco+_testB", 4889),
    ("refcocog_val", 4896),
    ("refcocog_test", 9602),
)


class FinalPaperSummaryTests(unittest.TestCase):
    def test_full_sam_summary_requires_all_models_and_splits(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for model in ("stamp2b", "stamp7b"):
                for split, samples in SPLITS:
                    output = root / "input" / model / split.replace("+", "plus")
                    output.mkdir(parents=True)
                    value = {
                        "protocol": "stamp_released_prompting_frozen_sam_vit_b_v1",
                        "samples": samples,
                        "coarse_cIoU": 0.70,
                        "sam_cIoU": 0.75,
                        "sam_mean_iou": 0.74,
                        "sam_boundary_iou": 0.30,
                    }
                    (output / "eval_summary.json").write_text(json.dumps(value), encoding="utf-8")
            argv = [
                "summarize_sam_full",
                "--input-root",
                str(root / "input"),
                "--output-dir",
                str(root / "output"),
                "--sam-model-type",
                "vit_b",
            ]
            with patch("sys.argv", argv):
                self.assertEqual(summarize_sam_main(), 0)
            rows = json.loads((root / "output" / "sam_full.json").read_text())["rows"]
            self.assertEqual(len(rows), 16)

    def test_full_postprocess_summary_macro_averages_eight_splits(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            methods = (
                "base",
                "densecrf",
                "guided_filter",
                "fast_bilateral_solver",
                "slic_average",
                "freeref",
            )
            for model in ("stamp7b", "text4seg_p24"):
                for split, samples in SPLITS:
                    output = root / "input" / model / split.replace("+", "plus")
                    output.mkdir(parents=True)
                    values = {}
                    for index, method in enumerate(methods):
                        values[method] = {
                            "mIoU": 0.70 + index * 0.01,
                            "delta_mIoU": index * 0.01,
                            "cIoU": 0.71 + index * 0.01,
                            "delta_cIoU": index * 0.01,
                            "bIoU": 0.20 + index * 0.01,
                            "delta_bIoU": index * 0.01,
                        }
                    (output / "eval_summary.json").write_text(
                        json.dumps({"samples": samples, "methods": values}), encoding="utf-8"
                    )
            argv = [
                "summarize_full_postprocess",
                "--input-root",
                str(root / "input"),
                "--output-dir",
                str(root / "output"),
            ]
            with patch("sys.argv", argv):
                self.assertEqual(summarize_postprocess_main(), 0)
            rows = json.loads((root / "output" / "postprocess_full.json").read_text())[
                "macro_rows"
            ]
            self.assertEqual(len(rows), 12)
            freeref = next(
                row for row in rows if row["model"] == "STAMP-7B" and row["method"] == "FreeRef"
            )
            self.assertAlmostEqual(freeref["delta_bIoU"], 5.0)

    def test_k_timing_summary_contains_base_and_each_configuration(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for model in ("stamp2b", "stamp7b", "text4seg_p24"):
                for variant, k in (("base", None), ("k_500", 500), ("k_1024", 1024)):
                    output = root / "input" / model / variant
                    output.mkdir(parents=True)
                    (output / "summary.json").write_text(
                        json.dumps(
                            {
                                "device": "NVIDIA H100 80GB HBM3",
                                "samples": 10,
                                "n_segments": k,
                                "e2e_mean_seconds": 1.0,
                                "e2e_median_seconds": 0.9,
                                "e2e_p95_seconds": 1.2,
                                "fps": 1.0,
                            }
                        ),
                        encoding="utf-8",
                    )
            argv = [
                "summarize_k_sweep",
                "--input-root",
                str(root / "input"),
                "--output-dir",
                str(root / "output"),
                "--k-values",
                "500",
                "1024",
            ]
            with patch("sys.argv", argv):
                self.assertEqual(summarize_k_main(), 0)
            rows = json.loads((root / "output" / "k_timing.json").read_text())["rows"]
            self.assertEqual(len(rows), 9)


if __name__ == "__main__":
    unittest.main()
