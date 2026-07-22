from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from training_free_refine.summarize_splits import main as summarize_splits_main
from training_free_refine.summarize_paper_studies import main as summarize_paper_studies_main


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

    def test_summarizes_paper_study_tree(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            study = root / "stamp7b" / "ablation" / "full"
            study.mkdir(parents=True)
            (study / "eval_summary.json").write_text(
                json.dumps(
                    {
                        "samples": 20,
                        "coarse_mean_iou": 0.60,
                        "refined_mean_iou": 0.63,
                        "mean_iou_delta": 0.03,
                        "coarse_cIoU": 0.61,
                        "refined_cIoU": 0.64,
                        "cIoU_delta": 0.03,
                        "boundary_iou_delta": 0.05,
                        "seconds_per_sample": 0.4,
                        "config": {
                            "n_segments": 1024,
                            "graph_lambda": 1.0,
                            "confidence_power": 2.0,
                            "uncertainty_aware_anchoring": True,
                            "appearance_weighted_graph": True,
                            "selective_fusion": True,
                        },
                    }
                ),
                encoding="utf-8",
            )
            output = root / "combined"
            argv = [
                "summarize_paper_studies",
                "--input-root",
                str(root),
                "--output-dir",
                str(output),
            ]
            with patch("sys.argv", argv):
                self.assertEqual(summarize_paper_studies_main(), 0)
            markdown = (output / "paper_studies.md").read_text(encoding="utf-8")
            self.assertIn("stamp7b | ablation | full", markdown)
            self.assertTrue((output / "paper_studies.csv").is_file())


if __name__ == "__main__":
    unittest.main()
