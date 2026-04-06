import os
import sys
import unittest

import numpy as np
import torch


REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
TRAIN_ROOT = os.path.join(REPO_ROOT, "train")
for path in (REPO_ROOT, TRAIN_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)


from train.methods.rad import RADMethod


class _FakeMesh:
    vertices = []


class _NonFiniteProblem:
    def pde_residual(self, model, x, y):
        return torch.full_like(x, float("nan"))


class _MixedResidualProblem:
    def pde_residual(self, model, x, y):
        values = torch.tensor(
            [0.0, 1.0, float("nan"), 4.0, float("inf")],
            dtype=x.dtype,
            device=x.device,
        )
        return values[: x.shape[0]]


class RADGuardTests(unittest.TestCase):
    def test_compute_residual_weights_falls_back_to_uniform_for_nonfinite_residuals(self):
        method = RADMethod(num_candidates=5, seed=123)
        method.set_problem(_NonFiniteProblem())
        candidate_points = np.array(
            [
                [0.1, 0.1],
                [0.2, 0.2],
                [0.3, 0.3],
                [0.4, 0.4],
                [0.5, 0.5],
            ],
            dtype=float,
        )

        weights, stats = method._compute_residual_weights(candidate_points, model=object())

        self.assertTrue(np.all(np.isfinite(weights)))
        self.assertTrue(np.allclose(weights, np.full(5, 0.2)))
        self.assertEqual(stats["pdf_status"], "uniform_fallback")
        self.assertEqual(stats["fallback_reason"], "no_finite_residuals")
        self.assertEqual(stats["residual_nonfinite"], 5)
        self.assertEqual(stats["weights_nonfinite"], 0)

    def test_get_collocation_points_survives_nan_pdf_path(self):
        method = RADMethod(num_candidates=5, seed=123)
        method.set_problem(_NonFiniteProblem())
        method._candidate_points = np.array(
            [
                [0.1, 0.1],
                [0.2, 0.2],
                [0.3, 0.3],
                [0.4, 0.4],
                [0.5, 0.5],
            ],
            dtype=float,
        )

        x, y = method.get_collocation_points(
            _FakeMesh(),
            model=object(),
            iteration=1,
            num_points=5,
        )

        self.assertEqual(len(x), 5)
        self.assertEqual(len(y), 5)
        self.assertEqual(method._last_iteration_stats["pdf_status"], "uniform_fallback")
        self.assertEqual(
            method._last_iteration_stats["fallback_reason"], "no_finite_residuals"
        )

    def test_compute_residual_weights_ignores_partial_nonfinite_entries(self):
        method = RADMethod(num_candidates=5, seed=123, c=0.1)
        method.set_problem(_MixedResidualProblem())
        candidate_points = np.array(
            [
                [0.1, 0.1],
                [0.2, 0.2],
                [0.3, 0.3],
                [0.4, 0.4],
                [0.5, 0.5],
            ],
            dtype=float,
        )

        weights, stats = method._compute_residual_weights(candidate_points, model=object())

        self.assertTrue(np.all(np.isfinite(weights)))
        self.assertAlmostEqual(float(np.sum(weights)), 1.0, places=12)
        self.assertEqual(stats["pdf_status"], "ok")
        self.assertEqual(stats["residual_finite"], 3)
        self.assertEqual(stats["residual_nonfinite"], 2)


if __name__ == "__main__":
    unittest.main()
