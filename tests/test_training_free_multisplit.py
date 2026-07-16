from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from training_free_refine.summarize_splits import main as summarize_splits_main


class TrainingFreeMultiSplitTests(unittest.TestCase):
    def test_combines_arbitrary_named_splits(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            paths = []
            for index, split in enumerate(("refcoco_val", "refcoco+_testA")):
                path = root / f"{split}.json"
                path.write_text(
                    json.dumps(
                        {
                            "samples": 10 + index,
                            "coarse_mean_iou": 0.60,
                            "refined_mean_iou": 0.62,
                            "mean_iou_delta": 0.02,
                            "coarse_cIoU": 0.61,
                            "refined_cIoU": 0.63,
                            "cIoU_delta": 0.02,
                            "boundary_iou_delta": 0.05,
                            "improved_samples": 8,
                            "degraded_samples": 2,
                            "unchanged_samples": index,
                            "seconds_per_sample": 0.4,
                        }
                    ),
                    encoding="utf-8",
                )
                paths.append((split, path))
            output_dir = root / "combined"
            argv = ["summarize_splits"]
            for split, path in paths:
                argv.extend(["--summary", f"{split}={path}"])
            argv.extend(["--output-dir", str(output_dir), "--title", "Test summary"])
            with patch("sys.argv", argv):
                self.assertEqual(summarize_splits_main(), 0)
            combined = json.loads((output_dir / "combined_summary.json").read_text(encoding="utf-8"))
            self.assertEqual([row["split"] for row in combined["splits"]], ["refcoco_val", "refcoco+_testA"])
            markdown = (output_dir / "combined_summary.md").read_text(encoding="utf-8")
            self.assertIn("refcoco+_testA", markdown)


if __name__ == "__main__":
    unittest.main()
