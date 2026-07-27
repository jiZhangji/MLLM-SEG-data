from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from PIL import Image

from paper_assets.qualitative_comparison.generate_qualitative_figures import (
    BLUE,
    BLUE_EDGE,
    mask_overlay,
    postprocess_score,
    save_grid,
)


class QualitativeFigureTests(unittest.TestCase):
    def test_overlay_uses_shared_blue_prediction_color(self):
        image = np.full((48, 64, 3), 180, dtype=np.uint8)
        mask = np.zeros((48, 64), dtype=bool)
        mask[10:38, 18:50] = True
        rendered = mask_overlay(image, mask, BLUE, BLUE_EDGE, alpha=0.5)
        self.assertGreater(float(rendered[20, 30, 2]), float(rendered[20, 30, 0]))
        self.assertTrue(np.array_equal(rendered[0, 0], image[0, 0]))

    def test_postprocess_score_prefers_freeref_over_competitors(self):
        weak = {
            "base_iou": "0.6",
            "base_boundary_iou": "0.2",
            "densecrf_iou": "0.61",
            "densecrf_boundary_iou": "0.22",
            "guided_filter_iou": "0.60",
            "guided_filter_boundary_iou": "0.21",
            "fast_bilateral_solver_iou": "0.61",
            "fast_bilateral_solver_boundary_iou": "0.23",
            "slic_average_iou": "0.60",
            "slic_average_boundary_iou": "0.20",
            "freeref_iou": "0.62",
            "freeref_boundary_iou": "0.24",
        }
        strong = dict(weak, freeref_iou="0.68", freeref_boundary_iou="0.42")
        self.assertGreater(postprocess_score(strong), postprocess_score(weak))

    def test_grid_exports_png_pdf_and_svg(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            panel = np.full((72, 96, 3), 235, dtype=np.uint8)
            panel[15:60, 25:75] = (40, 120, 200)
            rows = [
                {"sample_id": "1", "prompt": "the blue object", "panels": [panel] * 8},
                {"sample_id": "2", "prompt": "the second object", "panels": [panel] * 8},
            ]
            stem = root / "qualitative"
            save_grid(rows, [f"C{i}" for i in range(8)], stem, 80, {7})
            for suffix in ("png", "pdf", "svg"):
                self.assertTrue(stem.with_suffix(f".{suffix}").is_file())
            with Image.open(stem.with_suffix(".png")) as image:
                self.assertGreater(image.width, 900)
                self.assertGreater(image.height, 150)


if __name__ == "__main__":
    unittest.main()
