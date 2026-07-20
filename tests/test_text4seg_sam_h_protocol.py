from __future__ import annotations

import unittest

import numpy as np

from training_free_refine.eval_text4seg_sam_h import sam_refine, stable_sample_seed


class _IdentityResize:
    def __init__(self, target_length: int):
        self.target_length = target_length

    def apply_image(self, values: np.ndarray) -> np.ndarray:
        return values


class _Predictor:
    def __init__(self, shape: tuple[int, int]):
        self.shape = shape
        self.calls: list[dict[str, object]] = []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        prediction = np.ones((1, *self.shape), dtype=bool)
        low_res_logits = np.ones((1, 256, 256), dtype=np.float32)
        return prediction, np.asarray([1.0]), low_res_logits


class Text4SegSamHProtocolTests(unittest.TestCase):
    def test_stable_sample_seed_is_name_specific(self):
        self.assertEqual(stable_sample_seed(3, "a"), stable_sample_seed(3, "a"))
        self.assertNotEqual(stable_sample_seed(3, "a"), stable_sample_seed(3, "b"))

    def test_sam_refine_runs_initial_prediction_and_cascades(self):
        mask = np.zeros((8, 8), dtype=bool)
        mask[2:6, 2:6] = True
        predictor = _Predictor(mask.shape)
        result = sam_refine(
            predictor,
            _IdentityResize,
            mask,
            cascade_steps=2,
            point_count=3,
            seed=9,
        )
        self.assertEqual(len(predictor.calls), 3)
        self.assertEqual(predictor.calls[0]["mask_input"].shape, (1, 256, 256))
        self.assertEqual(predictor.calls[0]["point_coords"].shape, (6, 2))
        self.assertTrue(result.all())

    def test_empty_mask_skips_sam(self):
        mask = np.zeros((8, 8), dtype=bool)
        predictor = _Predictor(mask.shape)
        result = sam_refine(predictor, _IdentityResize, mask, 2, 3, 9)
        self.assertFalse(result.any())
        self.assertEqual(predictor.calls, [])


if __name__ == "__main__":
    unittest.main()

