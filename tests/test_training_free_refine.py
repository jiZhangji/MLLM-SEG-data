from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from training_free_refine.eval_text4seg_outputs import main as eval_text4seg_main
from training_free_refine.export_text4seg_masks import decode_prediction
from training_free_refine.refiner import (
    TrainingFreeRefineConfig,
    TrainingFreeUncertaintyRefiner,
    boundary_uncertainty,
    probability_uncertainty,
    stamp_probability,
)


class TrainingFreeRefineTests(unittest.TestCase):
    def test_uncertainty_peaks_at_half_probability(self):
        probability = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0])
        uncertainty = probability_uncertainty(probability)
        np.testing.assert_allclose(uncertainty, [0.0, 0.5, 1.0, 0.5, 0.0])

    def test_stamp_probability_matches_grid_and_output_size(self):
        logits = torch.tensor(
            [
                [[5.0, -5.0], [-5.0, 5.0], [5.0, -5.0], [-5.0, 5.0]],
            ]
        )
        probability = stamp_probability(logits, (2, 2), (12, 10))
        self.assertEqual(tuple(probability.shape), (12, 10))
        self.assertGreater(float(probability.max()), 0.99)
        self.assertLess(float(probability.min()), 0.01)

    def test_boundary_uncertainty_is_localized_around_hard_mask_edge(self):
        mask = np.zeros((64, 64), dtype=bool)
        mask[16:48, 16:48] = True
        uncertainty = boundary_uncertainty(mask, sigma=4.0)
        self.assertEqual(uncertainty.shape, mask.shape)
        self.assertGreater(float(uncertainty[16, 32]), 0.95)
        self.assertLess(float(uncertainty[32, 32]), 0.01)
        self.assertLess(float(uncertainty[0, 0]), 0.01)

    def test_hard_mask_refiner_uses_generic_probability_path(self):
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        image[:, :32] = (40, 90, 210)
        image[:, 32:] = (220, 70, 40)
        mask = np.zeros((64, 64), dtype=bool)
        mask[:, :36] = True
        refiner = TrainingFreeUncertaintyRefiner(
            TrainingFreeRefineConfig(n_segments=64, compactness=8.0, graph_lambda=1.0)
        )
        output = refiner.refine_hard_mask(image, mask, boundary_sigma=6.0)
        self.assertEqual(tuple(output["refined_probability"].shape), (64, 64))
        self.assertTrue(torch.isfinite(output["refined_probability"]).all())
        np.testing.assert_allclose(output["coarse_probability"].numpy(), mask.astype(np.float32))

    def test_refiner_is_finite_and_preserves_confident_regions(self):
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        image[:, :32] = (40, 90, 210)
        image[:, 32:] = (220, 70, 40)
        foreground = torch.full((8, 8), 0.02)
        foreground[:, :4] = 0.98
        foreground[:, 3:5] = torch.tensor([0.8, 0.2])
        logits = torch.stack([torch.log1p(-foreground), torch.log(foreground)], dim=-1).reshape(1, -1, 2)
        refiner = TrainingFreeUncertaintyRefiner(
            TrainingFreeRefineConfig(n_segments=64, compactness=8.0, graph_lambda=1.0)
        )
        output = refiner.refine(image, logits, (8, 8))
        coarse = output["coarse_probability"]
        refined = output["refined_probability"]
        self.assertEqual(tuple(refined.shape), (64, 64))
        self.assertTrue(torch.isfinite(refined).all())
        self.assertGreaterEqual(float(refined.min()), 0.0)
        self.assertLessEqual(float(refined.max()), 1.0)
        confident = (coarse <= 0.05) | (coarse >= 0.95)
        self.assertLess(float((refined[confident] - coarse[confident]).abs().max()), 0.03)
        self.assertGreater(int(output["diagnostics"]["segments"]), 1)

    def test_text4seg_saved_mask_evaluator(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_dir = root / "official"
            output_dir = root / "refined"
            sample_dir = input_dir / "model" / "refcocog_umd_val" / "image_1"
            sample_dir.mkdir(parents=True)
            image = np.zeros((48, 64, 3), dtype=np.uint8)
            image[:, :32] = (40, 90, 210)
            image[:, 32:] = (220, 70, 40)
            target = np.zeros((48, 64), dtype=np.uint8)
            target[:, :32] = 255
            coarse = np.zeros((48, 64), dtype=np.uint8)
            coarse[:, :36] = 255
            Image.fromarray(image).save(sample_dir / "0_image.png")
            Image.fromarray(target).save(sample_dir / "0_gt_mask.png")
            Image.fromarray(coarse).save(sample_dir / "0_pred_mask.png")
            Image.fromarray(target).save(sample_dir / "0_sam_mask.png")
            manifest = root / "manifest.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "name": "sample_0",
                        "image": str(sample_dir / "0_image.png"),
                        "gt_mask": str(sample_dir / "0_gt_mask.png"),
                        "pred_mask": str(sample_dir / "0_pred_mask.png"),
                        "sam_mask": str(sample_dir / "0_sam_mask.png"),
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            argv = [
                "eval_text4seg_outputs",
                "--manifest",
                str(manifest),
                "--output-dir",
                str(output_dir),
                "--n-segments",
                "48",
                "--boundary-sigma",
                "4",
                "--save-visualizations",
                "0",
            ]
            with patch("sys.argv", argv):
                self.assertEqual(eval_text4seg_main(), 0)
            summary = json.loads((output_dir / "eval_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["samples"], 1)
            self.assertAlmostEqual(summary["sam_cIoU"], 1.0)
            self.assertTrue((output_dir / "eval_rows.csv").exists())

    def test_text4seg_descriptor_decode_pads_to_grid(self):
        def fake_decode(value: str) -> str:
            self.assertEqual(value, "encoded")
            return "others|target"

        def fake_translate(value: str) -> list[int]:
            self.assertEqual(value, "others|target")
            return [0, 1]

        prediction = decode_prediction("prefix<seg>encoded</seg>suffix", 2, fake_decode, fake_translate)
        np.testing.assert_array_equal(prediction, [[0, 1], [1, 1]])


if __name__ == "__main__":
    unittest.main()
