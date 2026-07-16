from __future__ import annotations

import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
from PIL import Image

from training_free_refine.visualize_comparison import main as visualize_comparison_main


class TrainingFreeVisualizationTests(unittest.TestCase):
    def test_publication_visualization_and_cross_model_comparison(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image = np.zeros((48, 64, 3), dtype=np.uint8)
            image[:, :32] = (45, 105, 205)
            image[:, 32:] = (225, 80, 50)
            target = np.zeros((48, 64), dtype=np.uint8)
            target[:, :32] = 255
            coarse = np.zeros((48, 64), dtype=np.uint8)
            coarse[:, :36] = 255
            image_path = root / "image.png"
            target_path = root / "gt.png"
            coarse_path = root / "pred.png"
            Image.fromarray(image).save(image_path)
            Image.fromarray(target).save(target_path)
            Image.fromarray(coarse).save(coarse_path)
            rows_path = root / "eval_rows.csv"
            with rows_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["name", "image", "gt_mask", "pred_mask"])
                writer.writeheader()
                writer.writerow(
                    {
                        "name": "synthetic_boundary",
                        "image": str(image_path),
                        "gt_mask": str(target_path),
                        "pred_mask": str(coarse_path),
                    }
                )

            output_dir = root / "visuals"
            run_argv = [
                "visualize_comparison",
                "run",
                "--kind",
                "text4seg",
                "--rows",
                str(rows_path),
                "--output-dir",
                str(output_dir),
                "--label",
                "SyntheticText4Seg",
                "--n-segments",
                "48",
                "--boundary-sigma",
                "4",
                "--panels-per-group",
                "1",
            ]
            with patch("sys.argv", run_argv):
                self.assertEqual(visualize_comparison_main(), 0)
            overview_path = output_dir / "run_overview.png"
            self.assertTrue(overview_path.exists())
            with Image.open(overview_path) as overview:
                self.assertGreater(overview.width, 1000)
                self.assertGreater(overview.height, 500)
            panels = list((output_dir / "sample_panels").rglob("*.png"))
            self.assertEqual(len(panels), 1)

            combined_dir = root / "combined"
            analysis_rows = output_dir / "visual_analysis_rows.csv"
            compare_argv = [
                "visualize_comparison",
                "compare",
                "--run",
                f"MethodA={analysis_rows}",
                "--run",
                f"MethodB={analysis_rows}",
                "--output-dir",
                str(combined_dir),
            ]
            with patch("sys.argv", compare_argv):
                self.assertEqual(visualize_comparison_main(), 0)
            self.assertTrue((combined_dir / "cross_model_comparison.png").exists())
            self.assertTrue((combined_dir / "cross_model_summary.md").exists())


if __name__ == "__main__":
    unittest.main()
