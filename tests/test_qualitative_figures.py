from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from PIL import Image

from paper_assets.qualitative_comparison.generate_qualitative_figures import (
    BLUE,
    BLUE_EDGE,
    best_zoom_box,
    binary_panel,
    hard_recovery_score,
    mask_overlay,
    postprocess_score,
    save_binary_zoom_grid,
    save_grid,
    save_grid_pages,
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

    def test_hard_recovery_score_prefers_large_successful_recovery(self):
        modest = [
            {
                "base_iou": 0.68,
                "final_iou": 0.73,
                "iou_gain": 0.05,
                "base_boundary_iou": 0.30,
                "final_boundary_iou": 0.38,
                "boundary_gain": 0.08,
            }
        ] * 3
        strong = [
            {
                "base_iou": 0.58,
                "final_iou": 0.82,
                "iou_gain": 0.24,
                "base_boundary_iou": 0.18,
                "final_boundary_iou": 0.52,
                "boundary_gain": 0.34,
            }
        ] * 3
        self.assertGreater(hard_recovery_score(strong), hard_recovery_score(modest))

    def test_binary_panel_is_strict_black_and_white(self):
        mask = np.zeros((20, 30), dtype=bool)
        mask[4:15, 8:21] = True
        panel = binary_panel(mask)
        self.assertEqual(panel.shape, (20, 30, 3))
        self.assertEqual(set(np.unique(panel).tolist()), {0, 255})

    def test_zoom_box_finds_recovered_region(self):
        target = np.zeros((100, 120), dtype=bool)
        target[25:85, 25:95] = True
        base = target.copy()
        base[55:80, 70:92] = False
        refined = target.copy()
        x0, y0, x1, y1 = best_zoom_box(target, [base], [refined], fraction=0.30)
        self.assertLessEqual(x0, 80)
        self.assertGreaterEqual(x1, 70)
        self.assertLessEqual(y0, 70)
        self.assertGreaterEqual(y1, 55)

    def test_binary_zoom_grid_exports_all_formats(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image = np.full((80, 120, 3), 150, dtype=np.uint8)
            target = np.zeros((80, 120), dtype=bool)
            target[15:70, 30:95] = True
            binary = binary_panel(target)
            rows = [
                {
                    "sample_id": "7",
                    "prompt": "the foreground object",
                    "binary_panels": [image] + [binary] * 7,
                    "zoom_box": (45, 30, 90, 65),
                }
            ]
            stem = root / "binary_zoom"
            save_binary_zoom_grid(rows, [f"C{i}" for i in range(8)], stem, 70, {7})
            for suffix in ("png", "pdf", "svg"):
                self.assertTrue(stem.with_suffix(f".{suffix}").is_file())
            with Image.open(stem.with_suffix(".png")) as rendered:
                self.assertGreater(rendered.width, 900)
                self.assertGreater(rendered.height, 200)

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

    def test_large_candidate_sets_are_paginated(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            panel = np.full((48, 64, 3), 220, dtype=np.uint8)
            rows = [
                {"sample_id": str(index), "prompt": f"object {index}", "panels": [panel] * 8}
                for index in range(5)
            ]
            pages = save_grid_pages(
                rows,
                [f"C{i}" for i in range(8)],
                root / "comparison",
                60,
                {7},
                rows_per_page=2,
            )
            self.assertEqual(len(pages), 3)
            for index in range(1, 4):
                self.assertTrue((root / f"comparison_page_{index:02d}.pdf").is_file())


if __name__ == "__main__":
    unittest.main()
