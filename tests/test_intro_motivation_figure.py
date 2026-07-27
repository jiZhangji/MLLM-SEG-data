from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from paper_assets.intro_figure.generate_intro_motivation_figure import (
    error_callouts,
    main,
    pair_candidates,
)


class IntroMotivationFigureTests(unittest.TestCase):
    def test_pairing_requires_semantic_localization_and_prefers_boundary_gap(self):
        stamp = [
            {
                "name": "refcoco_val_000003",
                "coarse_iou": "0.80",
                "refined_iou": "0.82",
                "coarse_boundary_iou": "0.20",
                "refined_boundary_iou": "0.32",
            },
            {
                "name": "refcoco_val_000004",
                "coarse_iou": "0.30",
                "refined_iou": "0.70",
                "coarse_boundary_iou": "0.10",
                "refined_boundary_iou": "0.50",
            },
        ]
        pixellm = [
            {
                "instance_id": "3",
                "coarse_iou": "0.76",
                "refined_iou": "0.77",
                "coarse_boundary_iou": "0.24",
                "refined_boundary_iou": "0.31",
            },
            {
                "instance_id": "4",
                "coarse_iou": "0.35",
                "refined_iou": "0.65",
                "coarse_boundary_iou": "0.08",
                "refined_boundary_iou": "0.40",
            },
        ]
        paired = pair_candidates(stamp, pixellm)
        self.assertEqual([candidate.instance_id for candidate in paired], ["3"])

    def test_boundary_callouts_cover_local_false_positive_and_negative_regions(self):
        target = np.zeros((80, 100), dtype=bool)
        target[20:65, 25:75] = True
        coarse = target.copy()
        coarse[20:34, 25:40] = False
        coarse[45:62, 75:87] = True
        callouts = error_callouts(coarse, target, count=2)
        self.assertGreaterEqual(len(callouts), 1)
        self.assertTrue(
            {callout["category"] for callout in callouts}
            & {"Leakage", "Missing detail", "Boundary mismatch"}
        )

    def test_cli_generates_png_pdf_svg_and_selection_record(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            height, width = 96, 128
            yy, xx = np.mgrid[:height, :width]
            image = np.zeros((height, width, 3), dtype=np.uint8)
            image[..., 0] = np.clip(25 + xx, 0, 255)
            image[..., 1] = np.clip(190 - yy, 0, 255)
            image[..., 2] = 105
            target = ((xx - 65) / 29) ** 2 + ((yy - 49) / 28) ** 2 <= 1
            pixellm_probability = np.full((height, width), 0.04, dtype=np.float32)
            pixellm_probability[target] = 0.91
            pixellm_probability[38:57, 90:99] = 0.84

            image_path = root / "image.png"
            target_path = root / "target.png"
            prediction_path = root / "pixellm_logits.npz"
            Image.fromarray(image).save(image_path)
            Image.fromarray(target.astype(np.uint8) * 255).save(target_path)
            np.savez_compressed(prediction_path, logits=pixellm_probability)

            grid_h, grid_w = 8, 8
            grid_target = (
                np.asarray(
                    Image.fromarray(target.astype(np.uint8) * 255).resize(
                        (grid_w, grid_h), Image.Resampling.BILINEAR
                    )
                )
                / 255.0
            )
            foreground_logit = np.log(
                np.clip(grid_target, 1e-3, 1 - 1e-3)
                / np.clip(1.0 - grid_target, 1e-3, 1.0)
            )
            logits = torch.from_numpy(
                np.stack([np.zeros_like(foreground_logit), foreground_logit], axis=-1)
                .reshape(1, grid_h * grid_w, 2)
                .astype(np.float32)
            )
            dump_path = root / "refcoco_val_000000.pt"
            torch.save(
                {
                    "name": "refcoco_val_000000",
                    "index": 0,
                    "query": "Please segment the centered oval in this image.",
                    "image_path": str(image_path),
                    "mask_path": str(target_path),
                    "mask_logits": logits,
                    "mask_hidden": torch.zeros((1, grid_h * grid_w, 2)),
                    "grid_hw": (grid_h, grid_w),
                    "source_item": {"no_target": False},
                },
                dump_path,
            )

            stamp_rows = root / "stamp_rows.csv"
            with stamp_rows.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "name",
                        "dump",
                        "coarse_iou",
                        "refined_iou",
                        "coarse_boundary_iou",
                        "refined_boundary_iou",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "name": "refcoco_val_000000",
                        "dump": str(dump_path),
                        "coarse_iou": "0.78",
                        "refined_iou": "0.82",
                        "coarse_boundary_iou": "0.30",
                        "refined_boundary_iou": "0.44",
                    }
                )

            manifest_path = root / "manifest.jsonl"
            manifest_path.write_text(
                json.dumps(
                    {
                        "name": "pixellm_0",
                        "method": "PixelLM-7B",
                        "split": "refcoco_val",
                        "instance_id": "0",
                        "image": str(image_path),
                        "gt_mask": str(target_path),
                        "prediction": str(prediction_path),
                        "prediction_kind": "probability",
                        "array_key": "logits",
                        "threshold": 0.5,
                        "query": "the centered oval",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            pixellm_rows = root / "pixellm_rows.csv"
            with pixellm_rows.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "instance_id",
                        "coarse_iou",
                        "refined_iou",
                        "coarse_boundary_iou",
                        "refined_boundary_iou",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "instance_id": "0",
                        "coarse_iou": "0.76",
                        "refined_iou": "0.79",
                        "coarse_boundary_iou": "0.28",
                        "refined_boundary_iou": "0.42",
                    }
                )

            output_dir = root / "output"
            argv = [
                "generate_intro_motivation_figure",
                "--stamp-rows",
                str(stamp_rows),
                "--pixellm-rows",
                str(pixellm_rows),
                "--pixellm-manifest",
                str(manifest_path),
                "--output-dir",
                str(output_dir),
                "--candidate-pool",
                "1",
                "--contact-sheet-count",
                "1",
                "--minimum-box-iou",
                "0.35",
                "--n-segments",
                "48",
                "--dpi",
                "80",
            ]
            with patch("sys.argv", argv):
                self.assertEqual(main(), 0)

            for filename in (
                "freeref_intro_motivation.png",
                "freeref_intro_motivation.pdf",
                "freeref_intro_motivation.svg",
                "intro_candidate_contact_sheet.png",
                "intro_candidates.csv",
                "intro_figure_manifest.json",
            ):
                self.assertTrue((output_dir / filename).is_file(), filename)
            with Image.open(output_dir / "freeref_intro_motivation.png") as preview:
                self.assertGreater(preview.width, 700)
                self.assertGreater(preview.height, 450)


if __name__ == "__main__":
    unittest.main()
