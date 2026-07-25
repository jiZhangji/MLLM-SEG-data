from __future__ import annotations

import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
from PIL import Image

from paper_assets.framework.generate_framework_figure import main as framework_main
from paper_assets.framework.select_real_framework_sample import (
    main as selector_main,
    preselect_rows,
)


class RealFrameworkFigureTests(unittest.TestCase):
    def test_preselection_prefers_representative_positive_real_rows(self):
        rows = [
            {
                "name": "extreme",
                "pred_mask": "pred.png",
                "image": "image.png",
                "gt_mask": "gt.png",
                "coarse_iou": "0.4",
                "refined_iou": "0.8",
                "iou_delta": "0.4",
                "coarse_boundary_iou": "0.2",
                "refined_boundary_iou": "0.7",
            },
            {
                "name": "representative",
                "pred_mask": "pred.png",
                "image": "image.png",
                "gt_mask": "gt.png",
                "coarse_iou": "0.6",
                "refined_iou": "0.66",
                "iou_delta": "0.06",
                "coarse_boundary_iou": "0.4",
                "refined_boundary_iou": "0.48",
            },
            {
                "name": "failure",
                "pred_mask": "pred.png",
                "image": "image.png",
                "gt_mask": "gt.png",
                "coarse_iou": "0.7",
                "refined_iou": "0.6",
                "iou_delta": "-0.1",
                "coarse_boundary_iou": "0.5",
                "refined_boundary_iou": "0.4",
            },
        ]
        selected = preselect_rows(rows, "text4seg", pool_size=2)
        self.assertEqual([row["name"] for _, row in selected], ["representative", "extreme"])

    def test_real_asset_export_then_framework_render(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            height, width = 54, 72
            yy, xx = np.mgrid[:height, :width]
            image = np.zeros((height, width, 3), dtype=np.uint8)
            image[..., 0] = np.clip(35 + 3 * xx, 0, 255)
            image[..., 1] = np.clip(180 - 2 * yy, 0, 255)
            image[..., 2] = 105
            target = (((xx - 35) / 17) ** 2 + ((yy - 27) / 19) ** 2 <= 1)
            coarse = (((xx - 38) / 18) ** 2 + ((yy - 27) / 19) ** 2 <= 1)
            image_path = root / "image.png"
            target_path = root / "gt.png"
            coarse_path = root / "pred.png"
            Image.fromarray(image).save(image_path)
            Image.fromarray(target.astype(np.uint8) * 255).save(target_path)
            Image.fromarray(coarse.astype(np.uint8) * 255).save(coarse_path)
            rows_path = root / "eval_rows.csv"
            with rows_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "name",
                        "image",
                        "gt_mask",
                        "pred_mask",
                        "coarse_iou",
                        "refined_iou",
                        "iou_delta",
                        "coarse_boundary_iou",
                        "refined_boundary_iou",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "name": "real_pipeline_fixture",
                        "image": str(image_path),
                        "gt_mask": str(target_path),
                        "pred_mask": str(coarse_path),
                        "coarse_iou": "0.75",
                        "refined_iou": "0.78",
                        "iou_delta": "0.03",
                        "coarse_boundary_iou": "0.45",
                        "refined_boundary_iou": "0.50",
                    }
                )

            output_dir = root / "framework"
            selector_argv = [
                "select_real_framework_sample",
                "--kind",
                "text4seg",
                "--rows",
                str(rows_path),
                "--output-dir",
                str(output_dir),
                "--candidate-pool",
                "1",
                "--contact-sheet-count",
                "1",
                "--n-segments",
                "48",
                "--boundary-sigma",
                "4",
                "--dpi",
                "70",
            ]
            with patch("sys.argv", selector_argv):
                self.assertEqual(selector_main(), 0)

            bundle = output_dir / "selected_real_sample.npz"
            self.assertTrue(bundle.exists())
            self.assertTrue((output_dir / "selected_real_sample.json").exists())
            self.assertTrue(
                (output_dir / "framework_candidate_contact_sheet.png").exists()
            )
            with np.load(bundle, allow_pickle=False) as payload:
                self.assertEqual(payload["scene"].shape, (height, width, 3))
                self.assertEqual(payload["superpixels"].shape, (height, width))
                self.assertEqual(str(payload["name"].item()), "real_pipeline_fixture")

            render_argv = [
                "generate_framework_figure",
                "--sample-bundle",
                str(bundle),
                "--output-dir",
                str(output_dir),
                "--stem",
                "real_test",
                "--dpi",
                "80",
            ]
            with patch("sys.argv", render_argv):
                self.assertEqual(framework_main(), 0)
            preview = output_dir / "real_test.png"
            self.assertTrue(preview.exists())
            self.assertTrue((output_dir / "real_test.pdf").exists())
            self.assertTrue((output_dir / "real_test.svg").exists())
            with Image.open(preview) as rendered:
                self.assertGreater(rendered.width, 500)
                self.assertGreater(rendered.height, 250)


if __name__ == "__main__":
    unittest.main()
