import unittest

import torch

from refine_stamp.utils.geometry import patch_ids_to_boxes


class PatchCoordinateTest(unittest.TestCase):
    def test_patch_ids_to_boxes_square_grid(self):
        selected_ids = torch.tensor([[0, 5, 15]])
        boxes = patch_ids_to_boxes(
            selected_ids=selected_ids,
            grid_hw=(4, 4),
            image_hw=(80, 80),
        )
        expected = torch.tensor(
            [
                [
                    [0.0, 0.0, 20.0, 20.0],
                    [20.0, 20.0, 40.0, 40.0],
                    [60.0, 60.0, 80.0, 80.0],
                ]
            ]
        )
        self.assertTrue(torch.allclose(boxes, expected))

    def test_scaled_boxes_are_clamped(self):
        selected_ids = torch.tensor([[0]])
        boxes = patch_ids_to_boxes(
            selected_ids=selected_ids,
            grid_hw=(2, 2),
            image_hw=(100, 100),
            scale=2.0,
        )
        expected = torch.tensor([[[0.0, 0.0, 75.0, 75.0]]])
        self.assertTrue(torch.allclose(boxes, expected))


if __name__ == "__main__":
    unittest.main()
