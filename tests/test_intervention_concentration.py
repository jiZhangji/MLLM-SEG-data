from __future__ import annotations

import unittest

import numpy as np

from training_free_refine.analyze_intervention_concentration import (
    InterventionAccumulator,
)


class InterventionConcentrationTests(unittest.TestCase):
    def test_low_confidence_changes_and_high_confidence_is_preserved(self):
        coarse = np.full((40, 40), 0.95, dtype=np.float64)
        target = np.ones((40, 40), dtype=bool)
        coarse[15:25, 15:25] = 0.49
        refined = coarse.copy()
        refined[15:25, 15:25] = 0.75

        accumulator = InterventionAccumulator(0.5, (1, 3))
        accumulator.update(coarse, refined, target)
        summary = accumulator.summary()
        confidence = summary["confidence_bins"]

        self.assertGreater(confidence[0]["label_flip_rate"], 0.9)
        self.assertEqual(confidence[-1]["label_flip_rate"], 0.0)
        self.assertGreater(
            confidence[0]["mean_abs_probability_change"],
            confidence[-1]["mean_abs_probability_change"],
        )
        self.assertLess(summary["base_metrics"]["mIoU"], summary["freeref_metrics"]["mIoU"])
        self.assertLess(summary["base_metrics"]["cIoU"], summary["freeref_metrics"]["cIoU"])

    def test_gt_boundary_concentration_reports_share_and_enrichment(self):
        target = np.zeros((80, 80), dtype=bool)
        target[20:60, 20:60] = True
        coarse = np.where(target, 0.9, 0.1).astype(np.float64)
        refined = coarse.copy()
        refined[20:60, 20] = 0.1

        accumulator = InterventionAccumulator(0.5, (1, 3, 5))
        accumulator.update(coarse, refined, target)
        rows = accumulator.summary()["gt_boundary_concentration"]

        self.assertEqual(rows[0]["changed_pixel_share_in_band"], 1.0)
        self.assertGreater(rows[0]["change_enrichment"], 1.0)
        self.assertEqual(rows[-1]["changed_pixel_share_in_band"], 1.0)

    def test_distance_bins_localize_boundary_flips(self):
        target = np.zeros((64, 64), dtype=bool)
        target[16:48, 16:48] = True
        coarse = np.where(target, 0.9, 0.1).astype(np.float64)
        refined = coarse.copy()
        refined[16, 20:44] = 0.1

        accumulator = InterventionAccumulator(0.5, (2,))
        accumulator.update(coarse, refined, target)
        distance = accumulator.summary()["distance_bins"]

        self.assertGreater(distance[0]["label_flip_rate"], 0.0)
        self.assertEqual(distance[-1]["label_flip_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
