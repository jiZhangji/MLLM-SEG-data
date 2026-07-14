from __future__ import annotations

import unittest

import numpy as np
import torch

from training_free_refine.refiner import (
    TrainingFreeRefineConfig,
    TrainingFreeUncertaintyRefiner,
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


if __name__ == "__main__":
    unittest.main()
