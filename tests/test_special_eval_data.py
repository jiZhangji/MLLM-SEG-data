from __future__ import annotations

import argparse
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from offline_rstamp.scripts.prepare_special_eval_data import convert_grefcoco, convert_reasonseg

REFINE_SRC = Path(__file__).resolve().parents[1] / "offline_rstamp" / "refine_stamp_src"
sys.path.insert(0, str(REFINE_SRC))
from refine_stamp.scripts.export_stamp_refinement_dumps import export_dumps


class SpecialEvaluationDataTests(unittest.TestCase):
    def test_grefcoco_merges_targets_and_preserves_no_target(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            raw = root / "data" / "annotations" / "grefcoco"
            images = root / "data" / "shared" / "coco" / "train2014"
            raw.mkdir(parents=True)
            images.mkdir(parents=True)
            Image.new("RGB", (16, 12), (80, 120, 160)).save(images / "sample.jpg")
            instances = {
                "images": [{"id": 1, "file_name": "sample.jpg", "height": 12, "width": 16}],
                "annotations": [
                    {"id": 7, "image_id": 1, "segmentation": [[2, 2, 8, 2, 8, 8, 2, 8]]}
                ],
            }
            (raw / "instances.json").write_text(json.dumps(instances), encoding="utf-8")
            refs = [
                {
                    "image_id": 1,
                    "ref_id": 10,
                    "ann_id": [7],
                    "no_target": False,
                    "split": "val",
                    "sentences": [{"sent": "blue object", "sent_id": 1}],
                },
                {
                    "image_id": 1,
                    "ref_id": 11,
                    "ann_id": [-1],
                    "no_target": True,
                    "split": "val",
                    "sentences": [{"sent": "missing object", "sent_id": 2}],
                },
            ]
            (raw / "grefs(unc).json").write_text(json.dumps(refs), encoding="utf-8")
            output = root / "json"
            masks = root / "masks"
            summary = convert_grefcoco(root, ["val"], output, masks)
            self.assertEqual(summary[0]["samples"], 2)
            items = json.loads((output / "grefcoco_val.json").read_text(encoding="utf-8"))
            self.assertFalse(items[0]["no_target"])
            self.assertTrue(items[1]["no_target"])
            self.assertGreater(np.asarray(Image.open(items[0]["masks"][0])).sum(), 0)
            self.assertEqual(np.asarray(Image.open(items[1]["masks"][0])).sum(), 0)

    def test_reasonseg_writes_target_and_ignore_masks(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            split = root / "data" / "datasets" / "reasonseg" / "val"
            split.mkdir(parents=True)
            Image.new("RGB", (20, 16), (100, 100, 100)).save(split / "sample.jpg")
            value = {
                "text": ["thing used for testing"],
                "shapes": [
                    {"label": "target", "shape_type": "polygon", "points": [[2, 2], [12, 2], [12, 12], [2, 12]]},
                    {"label": "ignore", "shape_type": "polygon", "points": [[0, 0], [3, 0], [3, 3], [0, 3]]},
                ],
            }
            (split / "sample.json").write_text(json.dumps(value), encoding="utf-8")
            output = root / "json"
            masks = root / "masks"
            summary = convert_reasonseg(root, ["val"], output, masks)
            self.assertEqual(summary[0]["samples"], 1)
            item = json.loads((output / "reasonseg_val.json").read_text(encoding="utf-8"))[0]
            self.assertTrue(Path(item["masks"][0]).exists())
            self.assertTrue(Path(item["ignore_masks"][0]).exists())

    def test_dump_export_can_record_empty_prediction(self):
        class FailingSegmenter:
            def __init__(self, *args, **kwargs):
                pass

            def generate_with_refinement_outputs(self, image, query):
                raise RuntimeError("no SEG token")

        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image_path = root / "image.png"
            mask_path = root / "mask.png"
            Image.new("RGB", (12, 10), (100, 110, 120)).save(image_path)
            Image.new("L", (12, 10), 0).save(mask_path)
            json_path = root / "eval.json"
            json_path.write_text(
                json.dumps(
                    [
                        {
                            "images": [str(image_path)],
                            "masks": [str(mask_path)],
                            "label": "absent object",
                            "no_target": True,
                        }
                    ]
                ),
                encoding="utf-8",
            )
            output_dir = root / "dumps"
            args = argparse.Namespace(
                root=root,
                stamp_code_dir=root,
                model=root,
                json=json_path,
                output_dir=output_dir,
                split_name="grefcoco_val",
                limit=0,
                offset=0,
                min_pixels=1,
                max_pixels=100,
                prompt_mode="official",
                device_map="cpu",
                overwrite=False,
                fail_fast=False,
                empty_on_failure=True,
            )
            with patch(
                "refine_stamp.scripts.export_stamp_refinement_dumps.load_segmenter_class",
                return_value=FailingSegmenter,
            ), patch(
                "refine_stamp.scripts.export_stamp_refinement_dumps.load_official_question_templates",
                return_value=["Please segment the [class_name]."],
            ):
                report = export_dumps(args)
            self.assertEqual(report["num_empty_predictions"], 1)
            dump = torch.load(next(output_dir.glob("*.pt")), map_location="cpu", weights_only=False)
            self.assertTrue(dump["empty_prediction"])
            self.assertEqual(tuple(dump["grid_hw"]), (1, 1))


if __name__ == "__main__":
    unittest.main()
