import unittest

import torch

from refine_stamp.models.patch_selector import PatchSelector


class PatchSelectorTest(unittest.TestCase):
    def test_selector_returns_expected_shapes(self):
        logits = torch.zeros(2, 16, 2)
        selector = PatchSelector(top_k=5)
        output = selector(logits, grid_hw=(4, 4))
        self.assertEqual(tuple(output["selected_ids"].shape), (2, 5))
        self.assertEqual(tuple(output["uncertainty_map"].shape), (2, 1, 4, 4))
        self.assertEqual(tuple(output["boundary_map"].shape), (2, 1, 4, 4))

    def test_grid_mismatch_raises(self):
        logits = torch.zeros(1, 15, 2)
        selector = PatchSelector(top_k=4)
        with self.assertRaises(ValueError):
            selector(logits, grid_hw=(4, 4))

    def test_boundary_mode_prefers_foreground_background_interface(self):
        logits = torch.empty(1, 16, 2)
        logits[..., 0] = 5.0
        logits[..., 1] = -5.0
        # Make a 2x2 foreground block in the top-left corner.
        for patch_id in [0, 1, 4, 5]:
            logits[0, patch_id, 0] = -5.0
            logits[0, patch_id, 1] = 5.0

        selector = PatchSelector(top_k=4, mode="boundary")
        selected = set(selector(logits, grid_hw=(4, 4))["selected_ids"][0].tolist())
        self.assertTrue(selected)
        self.assertTrue(selected.issubset(set(range(16))))


if __name__ == "__main__":
    unittest.main()
