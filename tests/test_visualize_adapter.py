from __future__ import annotations

import unittest

import numpy as np
from PIL import Image

from token_refine.visualize_adapter_comparison import render_comparison, select_rows


class VisualizeAdapterTest(unittest.TestCase):
    def test_selection_uses_improved_worst_and_neutral_without_duplicates(self) -> None:
        rows = [
            {"name": f"sample_{index}", "path": f"sample_{index}.pt", "original_delta": str(delta)}
            for index, delta in enumerate([-0.3, -0.1, 0.0, 0.2, 0.5])
        ]
        selected = select_rows(rows, num_best=1, num_worst=1, num_neutral=1)
        self.assertEqual([category for category, _ in selected], ["best", "worst", "neutral"])
        self.assertEqual(len({row["path"] for _, row in selected}), 3)
        self.assertEqual(selected[0][1]["name"], "sample_4")
        self.assertEqual(selected[1][1]["name"], "sample_0")
        self.assertEqual(selected[2][1]["name"], "sample_2")

    def test_render_comparison_has_five_stable_panels(self) -> None:
        image = Image.new("RGB", (80, 60), (120, 130, 140))
        gt = np.zeros((60, 80), dtype=bool)
        coarse = np.zeros_like(gt)
        refined = np.zeros_like(gt)
        gt[10:40, 20:60] = True
        coarse[15:45, 25:65] = True
        refined[10:40, 20:60] = True
        output = render_comparison(image, gt, coarse, refined, "sample", panel_size=128)
        self.assertEqual(output.width, 5 * 128 + 4 * 8)
        self.assertEqual(output.height, 72 + 128 + 34)


if __name__ == "__main__":
    unittest.main()
