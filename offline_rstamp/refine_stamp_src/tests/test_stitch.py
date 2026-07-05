import unittest

import torch

from refine_stamp.utils.stitch import stitch_masks


class StitchTest(unittest.TestCase):
    def test_stitch_shapes_and_range(self):
        coarse_logits = torch.zeros(1, 4, 2)
        local_logits = torch.ones(1, 2, 1, 4, 4)
        selected_ids = torch.tensor([[0, 3]])

        final_prob, final_mask = stitch_masks(
            coarse_logits=coarse_logits,
            local_logits=local_logits,
            selected_ids=selected_ids,
            grid_hw=(2, 2),
            image_hw=(16, 16),
        )

        self.assertEqual(tuple(final_prob.shape), (1, 1, 16, 16))
        self.assertEqual(tuple(final_mask.shape), (1, 1, 16, 16))
        self.assertGreaterEqual(float(final_prob.min().item()), 0.0)
        self.assertLessEqual(float(final_prob.max().item()), 1.0)


if __name__ == "__main__":
    unittest.main()
