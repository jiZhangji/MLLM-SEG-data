from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from onepass_qwen7b.visualize_methods import (
    RunSpec,
    error_overlay,
    load_iou_rows,
    mask_iou,
    save_distribution_plot,
    select_cases,
)


class OnePass7BVisualizationTests(unittest.TestCase):
    @staticmethod
    def write_rows(path: Path, values: list[float]) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["index", "name", "iou"])
            writer.writeheader()
            for index, value in enumerate(values):
                writer.writerow({"index": index, "name": f"sample_{index}", "iou": value})

    def test_case_selection_and_distribution_plot(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            values = [
                [0.10, 0.30, 0.55, 0.80, 0.45, 0.90, 0.20, 0.70],
                [0.12, 0.55, 0.50, 0.78, 0.46, 0.91, 0.25, 0.68],
                [0.11, 0.50, 0.75, 0.60, 0.47, 0.92, 0.18, 0.72],
            ]
            paths = [root / f"rows_{index}.csv" for index in range(3)]
            for path, run_values in zip(paths, values):
                self.write_rows(path, run_values)
            rows = [load_iou_rows(path) for path in paths]
            selected = select_cases(rows, count=6)
            self.assertEqual(len(selected), 6)
            self.assertEqual(len({row["index"] for row in selected}), 6)
            self.assertTrue(all("category" in row for row in selected))

            specs = [RunSpec(f"run-{index}", root / "model.pt", path) for index, path in enumerate(paths)]
            output = root / "distribution.png"
            report = save_distribution_plot(output, specs, rows)
            self.assertTrue(output.is_file())
            self.assertEqual(report["common_samples"], 8)
            self.assertEqual(len(report["pairwise"]), 2)

    def test_error_overlay_and_iou(self):
        image = np.full((2, 2, 3), 100, dtype=np.uint8)
        target = np.asarray([[1, 1], [0, 0]], dtype=bool)
        prediction = np.asarray([[1, 0], [1, 0]], dtype=bool)
        overlay = error_overlay(image, prediction, target)
        self.assertEqual(overlay.shape, image.shape)
        self.assertFalse(np.array_equal(overlay[0, 0], image[0, 0]))
        self.assertAlmostEqual(mask_iou(prediction, target), 1.0 / 3.0)


if __name__ == "__main__":
    unittest.main()
