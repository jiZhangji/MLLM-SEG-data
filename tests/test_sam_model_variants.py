from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from training_free_refine.eval_stamp_sam_h import sam_protocol_name as stamp_protocol
from training_free_refine.eval_text4seg_sam_h import sam_protocol_name as text4seg_protocol
from training_free_refine.summarize_sam_variant import main as summarize_main


class SamModelVariantTests(unittest.TestCase):
    def test_vit_h_protocol_names_remain_backward_compatible(self):
        self.assertEqual(stamp_protocol("vit_h"), "stamp_official_frozen_sam_h_v1")
        self.assertEqual(
            text4seg_protocol("vit_h"),
            "text4seg_public_p24_paired_frozen_sam_h_v1",
        )

    def test_vit_b_protocol_names_are_explicit(self):
        self.assertEqual(
            stamp_protocol("vit_b"),
            "stamp_released_prompting_frozen_sam_vit_b_v1",
        )
        self.assertEqual(
            text4seg_protocol("vit_b"),
            "text4seg_public_p24_paired_frozen_sam_vit_b_v1",
        )

    def test_sam_variant_summary_contains_paired_accuracy_and_timing(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for slug in ("stamp7b", "text4seg_p24"):
                for stage in ("accuracy", "timing"):
                    output = root / "input" / stage / slug
                    output.mkdir(parents=True)
                    rows_path = output / "eval_rows.csv"
                    with rows_path.open("w", newline="", encoding="utf-8") as handle:
                        writer = csv.DictWriter(
                            handle,
                            fieldnames=(
                                "freeref_seconds",
                                "sam_image_seconds",
                                "sam_prompt_seconds",
                            ),
                        )
                        writer.writeheader()
                        writer.writerows(
                            [
                                {
                                    "freeref_seconds": 0.1,
                                    "sam_image_seconds": 0.2,
                                    "sam_prompt_seconds": 0.1,
                                },
                                {
                                    "freeref_seconds": 0.1,
                                    "sam_image_seconds": 0.3,
                                    "sam_prompt_seconds": 0.1,
                                },
                            ]
                        )
                    summary = {
                        "samples": 2,
                        "sam_model_type": "vit_b",
                        "rows_csv": str(rows_path),
                    }
                    for branch, value in (
                        ("coarse", 0.70),
                        ("freeref", 0.73),
                        ("coarse_sam", 0.76),
                        ("freeref_sam", 0.77),
                    ):
                        summary[f"{branch}_mean_iou"] = value
                        summary[f"{branch}_cIoU"] = value + 0.01
                        summary[f"{branch}_boundary_iou"] = value - 0.4
                    (output / "eval_summary.json").write_text(
                        json.dumps(summary), encoding="utf-8"
                    )
            combined = root / "combined"
            argv = [
                "summarize_sam_variant",
                "--input-root",
                str(root / "input"),
                "--output-dir",
                str(combined),
            ]
            with patch("sys.argv", argv):
                self.assertEqual(summarize_main(), 0)
            value = json.loads(
                (combined / "sam_variant_comparison.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(value["rows"]), 8)
            free_row = next(
                row
                for row in value["rows"]
                if row["model"] == "STAMP-7B" and row["branch"] == "FreeRef"
            )
            self.assertAlmostEqual(free_row["latency_seconds"], 0.1)
            self.assertAlmostEqual(free_row["throughput_samples_per_second"], 10.0)


if __name__ == "__main__":
    unittest.main()
