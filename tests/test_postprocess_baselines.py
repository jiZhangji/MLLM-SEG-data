from __future__ import annotations

import json
import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
from PIL import Image

from training_free_refine.eval_postprocess_baselines import main as evaluate_main
from training_free_refine.postprocess_baselines import (
    PostprocessBaselineConfig,
    _validate_inputs,
    fast_bilateral_solver_probability,
    guided_filter_probability,
    hard_mask_probability,
    slic_region_average_probability,
)
from training_free_refine.plot_postprocess_baselines import main as plot_main
from training_free_refine.summarize_postprocess_baselines import (
    assert_paired_base,
    main as summarize_main,
)


class PostprocessBaselineTests(unittest.TestCase):
    def test_densecrf_image_buffer_is_writable_and_contiguous(self):
        source = np.zeros((8, 10, 3), dtype=np.uint8)
        readonly = np.frombuffer(source.tobytes(), dtype=np.uint8).reshape(source.shape)
        self.assertFalse(readonly.flags.writeable)
        image, probability = _validate_inputs(
            readonly, np.full(source.shape[:2], 0.5, dtype=np.float32)
        )
        self.assertTrue(image.flags.writeable)
        self.assertTrue(image.flags.c_contiguous)
        self.assertEqual(probability.shape, source.shape[:2])

    def test_hard_mask_probability_uses_symmetric_confidence(self):
        mask = np.asarray([[False, True], [True, False]])
        np.testing.assert_allclose(
            hard_mask_probability(mask, 0.9),
            [[0.1, 0.9], [0.9, 0.1]],
            atol=1e-6,
        )

    def test_guided_filter_preserves_constant_probability(self):
        image = np.zeros((24, 32, 3), dtype=np.uint8)
        image[:, :16] = (30, 80, 210)
        image[:, 16:] = (220, 70, 40)
        probability = np.full((24, 32), 0.7, dtype=np.float32)
        output = guided_filter_probability(
            image,
            probability,
            PostprocessBaselineConfig(guided_radius=3),
        )
        np.testing.assert_allclose(output, probability, atol=1e-5)

    def test_fast_bilateral_solver_preserves_constant_probability(self):
        image = np.zeros((24, 32, 3), dtype=np.uint8)
        image[:, :16] = (30, 80, 210)
        image[:, 16:] = (220, 70, 40)
        probability = np.full((24, 32), 0.7, dtype=np.float32)
        output = fast_bilateral_solver_probability(
            image,
            probability,
            PostprocessBaselineConfig(
                fbs_sigma_spatial=8.0,
                fbs_iterations=50,
            ),
        )
        self.assertEqual(output.shape, probability.shape)
        self.assertTrue(np.isfinite(output).all())
        np.testing.assert_allclose(output, probability, atol=2e-3)

    def test_slic_average_is_constant_within_each_region(self):
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        image[:, :16] = (20, 80, 220)
        image[:, 16:] = (220, 70, 30)
        probability = np.linspace(0, 1, 32, dtype=np.float32)[None, :].repeat(32, axis=0)
        output = slic_region_average_probability(
            image,
            probability,
            PostprocessBaselineConfig(n_segments=8, compactness=8.0),
        )
        self.assertEqual(output.shape, probability.shape)
        self.assertTrue(np.isfinite(output).all())
        self.assertGreaterEqual(float(output.min()), 0.0)
        self.assertLessEqual(float(output.max()), 1.0)
        self.assertLess(len(np.unique(output)), output.size // 4)

    def test_text4seg_paired_evaluator_without_optional_densecrf(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest = root / "manifest.jsonl"
            records = []
            for index in range(2):
                image = np.zeros((32, 40, 3), dtype=np.uint8)
                image[:, :20] = (30, 80, 220)
                image[:, 20:] = (220, 70, 40)
                target = np.zeros((32, 40), dtype=np.uint8)
                target[:, :20] = 255
                coarse = np.zeros((32, 40), dtype=np.uint8)
                coarse[:, : 22 + index] = 255
                image_path = root / f"{index}_image.png"
                target_path = root / f"{index}_target.png"
                coarse_path = root / f"{index}_coarse.png"
                Image.fromarray(image).save(image_path)
                Image.fromarray(target).save(target_path)
                Image.fromarray(coarse).save(coarse_path)
                records.append(
                    {
                        "name": f"sample_{index}",
                        "image": str(image_path),
                        "gt_mask": str(target_path),
                        "pred_mask": str(coarse_path),
                    }
                )
            manifest.write_text(
                "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
            )
            output = root / "output"
            argv = [
                "eval_postprocess_baselines",
                "--source",
                "text4seg",
                "--manifest",
                str(manifest),
                "--output-dir",
                str(output),
                "--model-label",
                "Text4Seg-test",
                "--limit",
                "2",
                "--methods",
                "base",
                "guided_filter",
                "slic_average",
                "freeref",
                "--n-segments",
                "8",
            ]
            with patch("sys.argv", argv):
                self.assertEqual(evaluate_main(), 0)
            summary = json.loads((output / "eval_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["samples"], 2)
            self.assertEqual(summary["input_representation"], "hard_mask")
            self.assertEqual(set(summary["methods"]), set(argv[argv.index("--methods") + 1 : argv.index("--n-segments")]))

    def test_paired_base_check_rejects_different_sam_inputs(self):
        baseline = {"mIoU": 0.7, "cIoU": 0.72, "bIoU": 0.2}
        paired = {
            "coarse_mean_iou": 0.7,
            "coarse_cIoU": 0.72,
            "coarse_boundary_iou": 0.2,
        }
        assert_paired_base("model", baseline, paired)
        paired["coarse_cIoU"] = 0.71
        with self.assertRaisesRegex(ValueError, "not paired"):
            assert_paired_base("model", baseline, paired)

    def test_combined_summary_contains_all_six_refiners_per_model(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_root = root / "input"
            output = root / "combined"
            base_metrics = {
                "mIoU": 0.7,
                "cIoU": 0.72,
                "bIoU": 0.2,
                "delta_mIoU": 0.0,
                "delta_cIoU": 0.0,
                "delta_bIoU": 0.0,
            }
            methods = {}
            timing_methods = {}
            for index, name in enumerate(
                ("base", "densecrf", "guided_filter", "slic_average", "freeref")
            ):
                value = dict(base_metrics)
                if name != "base":
                    value.update(
                        {
                            "mIoU": 0.7 + index * 0.01,
                            "cIoU": 0.72 + index * 0.01,
                            "bIoU": 0.2 + index * 0.01,
                            "delta_mIoU": index * 0.01,
                            "delta_cIoU": index * 0.01,
                            "delta_bIoU": index * 0.01,
                        }
                    )
                methods[name] = value
                timing_methods[name] = {**value, "mean_seconds": 0.1 + index * 0.01}

            for slug in ("stamp7b", "text4seg_p24"):
                accuracy_dir = input_root / "accuracy" / slug
                timing_dir = input_root / "timing" / slug
                (accuracy_dir / "baselines").mkdir(parents=True)
                (timing_dir / "baselines").mkdir(parents=True)
                (accuracy_dir / "sam_h").mkdir(parents=True)
                (timing_dir / "sam_h").mkdir(parents=True)
                (accuracy_dir / "baselines" / "eval_summary.json").write_text(
                    json.dumps({"samples": 2, "methods": methods}), encoding="utf-8"
                )
                (timing_dir / "baselines" / "eval_summary.json").write_text(
                    json.dumps({"samples": 2, "methods": timing_methods}), encoding="utf-8"
                )
                sam_accuracy = {
                    "coarse_mean_iou": 0.7,
                    "coarse_cIoU": 0.72,
                    "coarse_boundary_iou": 0.2,
                    "coarse_sam_mean_iou": 0.76,
                    "coarse_sam_cIoU": 0.78,
                    "coarse_sam_boundary_iou": 0.3,
                }
                (accuracy_dir / "sam_h" / "eval_summary.json").write_text(
                    json.dumps(sam_accuracy), encoding="utf-8"
                )
                rows_path = timing_dir / "sam_h" / "eval_rows.csv"
                with rows_path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(
                        handle, fieldnames=("sam_image_seconds", "sam_prompt_seconds")
                    )
                    writer.writeheader()
                    writer.writerows(
                        [
                            {"sam_image_seconds": 0.2, "sam_prompt_seconds": 0.1},
                            {"sam_image_seconds": 0.3, "sam_prompt_seconds": 0.1},
                        ]
                    )
                (timing_dir / "sam_h" / "eval_summary.json").write_text(
                    json.dumps({"rows_csv": str(rows_path)}), encoding="utf-8"
                )

            argv = [
                "summarize_postprocess_baselines",
                "--input-root",
                str(input_root),
                "--output-dir",
                str(output),
            ]
            with patch("sys.argv", argv):
                self.assertEqual(summarize_main(), 0)
            summary = json.loads(
                (output / "postprocess_comparison.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(summary["rows"]), 12)
            self.assertEqual(
                {row["refiner"] for row in summary["rows"]},
                {"None", "DenseCRF", "Guided Filter", "SLIC Averaging", "SAM-H", "FreeRef"},
            )
            plot_output = root / "plots"
            plot_argv = [
                "plot_postprocess_baselines",
                "--input",
                str(output / "postprocess_comparison.csv"),
                "--output-dir",
                str(plot_output),
                "--dpi",
                "72",
            ]
            with patch("sys.argv", plot_argv):
                self.assertEqual(plot_main(), 0)
            for name in (
                "postprocess_accuracy_gains.png",
                "postprocess_accuracy_gains.pdf",
                "postprocess_accuracy_latency.png",
                "postprocess_accuracy_latency.pdf",
            ):
                self.assertGreater((plot_output / name).stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
