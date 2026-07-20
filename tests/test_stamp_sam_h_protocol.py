from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import torch

from training_free_refine.eval_stamp_sam_h import sam_refine, stable_sample_seed


class _FakePredictor:
    def __init__(self, output_shape: tuple[int, int]) -> None:
        self.output_shape = output_shape
        self.calls: list[dict[str, np.ndarray]] = []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        mask = np.ones((1, *self.output_shape), dtype=bool)
        score = np.asarray([0.9], dtype=np.float32)
        logits = np.full((1, 256, 256), len(self.calls), dtype=np.float32)
        return mask, score, logits


class StampSamHProtocolTests(unittest.TestCase):
    def test_full_runner_exposes_bounded_parallel_scheduler(self):
        root = Path(__file__).resolve().parents[1]
        script = (root / "run_frozen_samh_full_eval.sh").read_text(encoding="utf-8")
        self.assertIn('PARALLEL_JOBS="${SAMH_PARALLEL_JOBS:-1}"', script)
        self.assertIn("wait_batch", script)
        self.assertIn("combined summary was not generated", script)

    def test_stable_sample_seed_is_paired_and_name_specific(self):
        self.assertEqual(stable_sample_seed(7, "sample-a"), stable_sample_seed(7, "sample-a"))
        self.assertNotEqual(stable_sample_seed(7, "sample-a"), stable_sample_seed(7, "sample-b"))

    def test_sam_refine_uses_official_helpers_and_two_cascades(self):
        mask = np.zeros((8, 10), dtype=bool)
        mask[2:6, 3:8] = True
        helper_calls: dict[str, object] = {}

        def compute_logits(value: np.ndarray) -> np.ndarray:
            np.testing.assert_array_equal(value, mask.astype(np.float32))
            helper_calls["logits"] = True
            return np.zeros((1, 256, 256), dtype=np.float32)

        def sample_points(value: torch.Tensor):
            np.testing.assert_array_equal(value.numpy(), mask.astype(np.float32))
            helper_calls["numpy_draw"] = int(np.random.randint(0, 1_000_000))
            helper_calls["torch_draw"] = int(torch.randint(0, 1_000_000, (1,)).item())
            return np.asarray([[4.0, 3.0], [0.0, 0.0]]), np.asarray([1, 0])

        predictor = _FakePredictor(mask.shape)
        output = sam_refine(predictor, compute_logits, sample_points, mask, 2, 123)

        self.assertTrue(helper_calls["logits"])
        self.assertEqual(len(predictor.calls), 3)
        np.testing.assert_array_equal(output, np.ones_like(mask))
        np.testing.assert_array_equal(
            predictor.calls[1]["mask_input"], np.ones((1, 256, 256), dtype=np.float32)
        )
        np.testing.assert_array_equal(
            predictor.calls[2]["mask_input"], np.full((1, 256, 256), 2, dtype=np.float32)
        )

    def test_sam_refine_restores_numpy_and_torch_rng_state(self):
        mask = np.ones((4, 5), dtype=bool)

        def compute_logits(_: np.ndarray) -> np.ndarray:
            return np.zeros((1, 256, 256), dtype=np.float32)

        def sample_points(_: torch.Tensor):
            np.random.randint(0, 10)
            torch.randint(0, 10, (1,))
            return np.asarray([[1.0, 1.0]]), np.asarray([1])

        np.random.seed(41)
        torch.manual_seed(41)
        numpy_state = np.random.get_state()
        torch_state = torch.random.get_rng_state()

        sam_refine(_FakePredictor(mask.shape), compute_logits, sample_points, mask, 0, 9)

        actual_numpy = np.random.randint(0, 1_000_000)
        actual_torch = torch.randint(0, 1_000_000, (1,)).item()
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        self.assertEqual(actual_numpy, np.random.randint(0, 1_000_000))
        self.assertEqual(actual_torch, torch.randint(0, 1_000_000, (1,)).item())


if __name__ == "__main__":
    unittest.main()
