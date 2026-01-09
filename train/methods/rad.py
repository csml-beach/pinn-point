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
        domain_bounds: Tuple[float, float] = (0.0, 5.0),
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

        # Reference to problem for computing residuals
        self._problem = None

    def set_problem(self, problem):
        """Set the PDE problem for residual computation.

        Args:
            problem: PDEProblem instance with pde_residual() method
        """
        self._problem = problem

    def _generate_candidate_points(self, mesh: Any) -> np.ndarray:
        """Generate dense candidate set using rejection sampling.

        Args:
            mesh: NGSolve mesh for domain checking

        Returns:
            Array of shape (num_candidates, 2) with valid interior points
        """
        domain_min, domain_max = self.domain_bounds

        valid_points = []
        max_attempts = max(5000, self.num_candidates * 20)
        attempts = 0

        while len(valid_points) < self.num_candidates and attempts < max_attempts:
            x = self._rng.uniform(domain_min, domain_max)
            y = self._rng.uniform(domain_min, domain_max)

            try:
                if mesh(x, y).nr != -1:
                    valid_points.append((x, y))
            except Exception:
                pass
            attempts += 1

        if len(valid_points) < self.num_candidates:
            print(
                f"Warning: Generated {len(valid_points)}/{self.num_candidates} candidate points"
            )

        return np.array(valid_points)

    def _compute_residual_weights(
        self, candidate_points: np.ndarray, model: Any
    ) -> np.ndarray:
        """Compute probability weights based on PDE residual.

        Args:
            candidate_points: Array of shape (N, 2) with candidate coordinates
            model: Trained PINN model

        Returns:
            Array of shape (N,) with normalized probability weights
        """
        from config import DEVICE

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

        # Take absolute value and detach
        residual_abs = torch.abs(residual).detach().cpu().numpy()

        # Handle edge cases
        residual_abs = np.clip(residual_abs, 1e-10, None)  # Avoid zero

        # Compute weights: p(x) ∝ ε^k / E[ε^k] + c
        residual_k = np.power(residual_abs, self.k)
        mean_residual_k = np.mean(residual_k)

        if mean_residual_k < 1e-10:
            # Uniform weights if residual is near zero everywhere
            weights = np.ones(len(residual_k))
        else:
            weights = residual_k / mean_residual_k + self.c

        # Normalize to sum to 1
        weights = weights / np.sum(weights)

        return weights

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
                self._candidate_points = self._generate_candidate_points(mesh)

            if model is None:
                # First iteration: uniform random sampling
                indices = self._rng.choice(
                    len(self._candidate_points),
                    size=min(num_points, len(self._candidate_points)),
                    replace=False,
                )
            else:
                # Compute residual-based weights
                weights = self._compute_residual_weights(self._candidate_points, model)

                # Sample according to weights (with replacement if needed)
                replace = num_points > len(self._candidate_points)
                indices = self._rng.choice(
                    len(self._candidate_points),
                    size=num_points,
                    replace=replace,
                    p=weights,
                )

            self._cached_points = self._candidate_points[indices]
            self._last_resample_iteration = iteration

        points = self._cached_points
        x = torch.tensor(points[:, 0], dtype=torch.float32)
        y = torch.tensor(points[:, 1], dtype=torch.float32)

        return x, y

    def refine_mesh(
        self, mesh: Any, model: Any, iteration: int = 0
    ) -> Tuple[Any, bool]:
        """No mesh refinement for RAD - point selection is via resampling.

        Returns:
            (mesh, False) - mesh unchanged
        """
        return mesh, False

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
