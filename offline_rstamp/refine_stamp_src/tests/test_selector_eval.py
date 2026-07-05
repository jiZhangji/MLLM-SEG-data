import unittest

import torch

from refine_stamp.utils.selector_eval import (
    logits_to_coarse_mask,
    resize_binary_to_grid,
    selected_patch_rate,
    selector_quality_metrics,
)


class SelectorEvalTest(unittest.TestCase):
    def test_resize_binary_to_grid(self):
        mask = torch.zeros(8, 8)
        mask[:4, :4] = 1
        grid = resize_binary_to_grid(mask, (2, 2))
        expected = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]])
        self.assertTrue(torch.equal(grid, expected))

    def test_logits_to_coarse_mask(self):
        logits = torch.tensor([[[5.0, -5.0], [-5.0, 5.0], [5.0, -5.0], [-5.0, 5.0]]])
        mask = logits_to_coarse_mask(logits, (2, 2))
        expected = torch.tensor([[[[0.0, 1.0], [0.0, 1.0]]]])
        self.assertTrue(torch.equal(mask, expected))

    def test_selected_patch_rate(self):
        selected_ids = torch.tensor([[0, 3]])
        grid_map = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
        rate = selected_patch_rate(selected_ids, grid_map)
        self.assertAlmostEqual(float(rate.item()), 1.0)

    def test_selector_quality_metrics_keys(self):
        selected_ids = torch.tensor([[0, 1]])
        score_map = torch.ones(1, 1, 2, 2)
        coarse = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]])
        gt = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]])
        metrics = selector_quality_metrics(
            selected_ids=selected_ids,
            score_map=score_map,
            coarse_grid_mask=coarse,
            gt_grid_mask=gt,
        )
        self.assertIn("selected_gt_boundary_rate", metrics)
        self.assertIn("selected_error_rate", metrics)
        self.assertIn("score_mean_on_selected", metrics)


if __name__ == "__main__":
    unittest.main()
