"""
Residual-based Adaptive Distribution (RAD) method.

Based on Wu et al. (2022) "A comprehensive study of non-adaptive and 
residual-based adaptive sampling for physics-informed neural networks"

RAD samples collocation points according to a probability density function
proportional to the PDE residual, allowing the network to focus on 
difficult regions while maintaining some coverage everywhere.
"""

import torch
import numpy as np
from typing import Tuple, Optional, Any
from .base import TrainingMethod
from .sampling import points_to_tensors, sample_points_in_domain


class RADMethod(TrainingMethod):
    """Residual-based Adaptive Distribution sampling method.

    At each resampling step:
    1. Evaluate PDE residual on a dense candidate set
    2. Compute probability weights: p(x) ∝ ε^k(x) / E[ε^k(x)] + c
    3. Sample all collocation points according to these weights

    Hyperparameters:
        k: Exponent controlling focus on high-residual regions (default: 2)
        c: Regularization ensuring some uniform coverage (default: 0)

    Special cases:
        k=0 or c→∞: Reduces to uniform Random-R
        k=1, c=0: Nabian et al.'s method
        k=2, c=0: Recommended default (Wu et al.)
    """

    name = "rad"
    description = "Residual-based Adaptive Distribution (Wu et al. 2022)"

    def __init__(
        self,
        domain_bounds: Tuple[float, float, float, float] = (0.0, 5.0, 0.0, 5.0),
        k: float = 2.0,
        c: float = 0.0,
        num_candidates: int = 2000,
        resample_period: int = 1,
        seed: Optional[int] = None,
    ):
        """Initialize RAD method.

        Args:
            domain_bounds: (min, max) for both x and y coordinates
            k: Exponent for residual weighting (higher = more focus on max residual)
            c: Regularization constant for uniform coverage
            num_candidates: Size of dense candidate set for residual evaluation
            resample_period: Resample every N iterations
            seed: Random seed for reproducibility
        """
        self.domain_bounds = domain_bounds
        self.k = k
        self.c = c
        self.num_candidates = num_candidates
        self.resample_period = resample_period
        self.seed = seed
        self._rng = np.random.RandomState(seed)

        self._candidate_points = None
        self._cached_points = None
        self._last_resample_iteration = -1
        self._last_iteration_stats: dict[str, int | float | bool | str] = {}

        # Reference to problem for computing residuals
        self._problem = None

    def set_problem(self, problem):
        """Set the PDE problem for residual computation.

        Args:
            problem: PDEProblem instance with pde_residual() method
        """
        self._problem = problem

    def _generate_candidate_points(
        self, mesh: Any, iteration: int | None = None
    ) -> np.ndarray:
        """Generate dense candidate set using rejection sampling.

        Args:
            mesh: NGSolve mesh for domain checking

        Returns:
            Array of shape (num_candidates, 2) with valid interior points
        """
        x_min, x_max, y_min, y_max = self.domain_bounds

        return sample_points_in_domain(
            mesh,
            self.num_candidates,
            lambda batch_size: np.column_stack(
                (
                    self._rng.uniform(x_min, x_max, size=batch_size),
                    self._rng.uniform(y_min, y_max, size=batch_size),
                )
            ),
            batch_size=max(512, self.num_candidates // 2),
            max_batches=40,
            warn_label="candidate points",
            method_name=self.name,
            iteration=iteration,
        )

    def _uniform_probability_weights(
        self,
        count: int,
        *,
        fallback_reason: str,
        residual_stats: dict[str, int | float | bool | str] | None = None,
    ) -> tuple[np.ndarray, dict[str, int | float | bool | str]]:
        """Build a uniform PDF with diagnostic metadata."""
        if count <= 0:
            raise ValueError("count must be positive")

        weights = np.full(count, 1.0 / count, dtype=np.float64)
        stats: dict[str, int | float | bool | str] = {
            "pdf_status": "uniform_fallback",
            "fallback_reason": fallback_reason,
            "weights_finite": int(count),
            "weights_nonfinite": 0,
            "weights_zero": 0,
            "weight_min": float(weights[0]),
            "weight_max": float(weights[0]),
            "weight_mean": float(weights[0]),
            "weight_sum": 1.0,
        }
        if residual_stats:
            stats.update(residual_stats)
        return weights, stats

    def _compute_residual_weights(
        self, candidate_points: np.ndarray, model: Any
    ) -> tuple[np.ndarray, dict[str, int | float | bool | str]]:
        """Compute probability weights based on PDE residual.

        Args:
            candidate_points: Array of shape (N, 2) with candidate coordinates
            model: Trained PINN model

        Returns:
            Tuple of normalized probability weights and diagnostic metadata
        """
        from config import DEVICE

        candidate_count = int(len(candidate_points))
        if candidate_count == 0:
            raise ValueError("candidate_points must be non-empty")

        # Convert to tensors
        x = torch.tensor(candidate_points[:, 0], dtype=torch.float32, device=DEVICE)
        y = torch.tensor(candidate_points[:, 1], dtype=torch.float32, device=DEVICE)
        x.requires_grad_(True)
        y.requires_grad_(True)

        # Compute PDE residual
        if self._problem is not None:
            residual = self._problem.pde_residual(model, x, y)
        else:
            # Fallback: use model's built-in residual (for backward compatibility)
            residual = model.PDE_residual(x, y)

        residual_abs = torch.abs(residual).detach().cpu().numpy().astype(np.float64)
        residual_abs = np.reshape(residual_abs, (-1,))

        residual_finite_mask = np.isfinite(residual_abs)
        finite_residuals = residual_abs[residual_finite_mask]
        residual_stats: dict[str, int | float | bool | str] = {
            "candidate_count": candidate_count,
            "residual_finite": int(np.count_nonzero(residual_finite_mask)),
            "residual_nonfinite": int(np.count_nonzero(~residual_finite_mask)),
            "residual_min": (
                float(np.min(finite_residuals))
                if finite_residuals.size
                else float("nan")
            ),
            "residual_max": (
                float(np.max(finite_residuals))
                if finite_residuals.size
                else float("nan")
            ),
            "residual_mean": (
                float(np.mean(finite_residuals))
                if finite_residuals.size
                else float("nan")
            ),
        }

        if not finite_residuals.size:
            return self._uniform_probability_weights(
                candidate_count,
                fallback_reason="no_finite_residuals",
                residual_stats=residual_stats,
            )

        safe_residuals = np.clip(finite_residuals, 1e-300, None)

        if self.k == 0:
            residual_component = np.ones_like(safe_residuals, dtype=np.float64)
        else:
            log_component = self.k * np.log(safe_residuals)
            max_log_component = np.max(log_component)
            shifted_component = log_component - max_log_component
            residual_component = np.exp(shifted_component)

        if not np.all(np.isfinite(residual_component)):
            return self._uniform_probability_weights(
                candidate_count,
                fallback_reason="nonfinite_residual_component",
                residual_stats=residual_stats,
            )

        mean_component = float(np.mean(residual_component))
        if not np.isfinite(mean_component) or mean_component <= 0.0:
            return self._uniform_probability_weights(
                candidate_count,
                fallback_reason="invalid_component_mean",
                residual_stats=residual_stats,
            )

        weights = np.zeros(candidate_count, dtype=np.float64)
        weights[residual_finite_mask] = residual_component / mean_component
        if self.c:
            weights += float(self.c)

        invalid_weight_mask = ~np.isfinite(weights) | (weights < 0.0)
        if np.any(invalid_weight_mask):
            weights[invalid_weight_mask] = 0.0

        weight_sum = float(np.sum(weights))
        if not np.isfinite(weight_sum) or weight_sum <= 0.0:
            return self._uniform_probability_weights(
                candidate_count,
                fallback_reason="invalid_weight_sum",
                residual_stats=residual_stats,
            )

        weights /= weight_sum

        finite_weight_mask = np.isfinite(weights)
        finite_weights = weights[finite_weight_mask]
        stats = {
            **residual_stats,
            "pdf_status": "ok",
            "fallback_reason": "",
            "weights_finite": int(np.count_nonzero(finite_weight_mask)),
            "weights_nonfinite": int(np.count_nonzero(~finite_weight_mask)),
            "weights_zero": int(np.count_nonzero(weights == 0.0)),
            "weight_min": (
                float(np.min(finite_weights)) if finite_weights.size else float("nan")
            ),
            "weight_max": (
                float(np.max(finite_weights)) if finite_weights.size else float("nan")
            ),
            "weight_mean": (
                float(np.mean(finite_weights)) if finite_weights.size else float("nan")
            ),
            "weight_sum": float(np.sum(finite_weights))
            if finite_weights.size
            else float("nan"),
        }
        return weights, stats

    def get_collocation_points(
        self,
        mesh: Any,
        model: Optional[Any] = None,
        iteration: int = 0,
        num_points: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample collocation points according to residual-based PDF.

        Args:
            mesh: Mesh object for domain checking
            model: Trained model for residual computation (required for adaptive)
            iteration: Current iteration (used for resample schedule)
            num_points: Number of points to sample. If None, matches mesh vertex count.

        Returns:
            Tuple of (x, y) tensors
        """
        if num_points is None:
            num_points = len(list(mesh.vertices))

        # Check if we should resample
        should_resample = (
            self._cached_points is None
            or (iteration - self._last_resample_iteration) >= self.resample_period
        )

        if should_resample:
            # Generate or regenerate candidate points
            if (
                self._candidate_points is None
                or len(self._candidate_points) < self.num_candidates
            ):
                self._candidate_points = self._generate_candidate_points(
                    mesh, iteration=iteration
                )

            if model is None:
                # First iteration: uniform random sampling
                indices = self._rng.choice(
                    len(self._candidate_points),
                    size=min(num_points, len(self._candidate_points)),
                    replace=False,
                )
                self._last_iteration_stats = {
                    "iteration": int(iteration),
                    "sampling_strategy": "initial_uniform_from_candidates",
                    "resampled": True,
                    "num_candidates": int(len(self._candidate_points)),
                    "sampled_points": int(len(indices)),
                    "replace": False,
                }
            else:
                # Compute residual-based weights
                weights, pdf_stats = self._compute_residual_weights(
                    self._candidate_points, model
                )
                if pdf_stats.get("pdf_status") != "ok":
                    print(
                        f"[RAD] Iteration {iteration}: falling back to uniform PDF "
                        f"({pdf_stats.get('fallback_reason', 'unknown')})"
                    )

                # Sample according to weights (with replacement if needed)
                replace = num_points > len(self._candidate_points)
                indices = self._rng.choice(
                    len(self._candidate_points),
                    size=num_points,
                    replace=replace,
                    p=weights,
                )
                self._last_iteration_stats = {
                    "iteration": int(iteration),
                    "sampling_strategy": "residual_pdf",
                    "resampled": True,
                    "num_candidates": int(len(self._candidate_points)),
                    "sampled_points": int(len(indices)),
                    "replace": bool(replace),
                    **pdf_stats,
                }

            self._cached_points = self._candidate_points[indices]
            self._last_resample_iteration = iteration
        else:
            self._last_iteration_stats = {
                "iteration": int(iteration),
                "sampling_strategy": "cached_residual_pdf",
                "resampled": False,
                "num_candidates": (
                    int(len(self._candidate_points))
                    if self._candidate_points is not None
                    else 0
                ),
                "sampled_points": int(len(self._cached_points))
                if self._cached_points is not None
                else 0,
                "replace": False,
            }

        points = self._cached_points
        return points_to_tensors(points)

    def get_element_errors(self, mesh: Any, model: Any) -> np.ndarray:
        """Compute element-wise residual errors for visualization.

        Args:
            mesh: NGSolve mesh
            model: Trained PINN model

        Returns:
            Array of residual values at current collocation points
        """
        if self._cached_points is None:
            return np.array([])

        from config import DEVICE

        x = torch.tensor(self._cached_points[:, 0], dtype=torch.float32, device=DEVICE)
        y = torch.tensor(self._cached_points[:, 1], dtype=torch.float32, device=DEVICE)
        x.requires_grad_(True)
        y.requires_grad_(True)

        if self._problem is not None:
            residual = self._problem.pde_residual(model, x, y)
        else:
            residual = model.PDE_residual(x, y)

        return torch.abs(residual).detach().cpu().numpy()

    def log_iteration(self, iteration: int, mesh: Any, model: Any) -> dict:
        base_log = super().log_iteration(iteration, mesh, model)
        if self._last_iteration_stats:
            base_log.update(self._last_iteration_stats)
        return base_log
