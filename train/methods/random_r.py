"""
Random sampling with periodic resampling (Random-R).

Based on Wu et al. (2022) - resampling collocation points periodically
during training can improve convergence compared to fixed random points.
"""

import torch
import numpy as np
from typing import Tuple, Optional, Any
from .base import TrainingMethod


class RandomResamplingMethod(TrainingMethod):
    """Uniform random sampling with periodic resampling.

    Unlike fixed random sampling, this method regenerates collocation points
    every N iterations. This helps the network see different regions of the
    domain during training.

    From Wu et al. (2022): Random-R is a special case of RAD with k=0.
    """

    name = "random_r"
    description = "Uniform random sampling with periodic resampling"

    def __init__(
        self,
        domain_bounds: Tuple[float, float] = (0.0, 5.0),
        resample_period: int = 1,
        seed: Optional[int] = None,
    ):
        """Initialize Random-R method.

        Args:
            domain_bounds: (min, max) for both x and y coordinates
            resample_period: Resample every N iterations (1 = every iteration)
            seed: Random seed (if None, uses global numpy random state)
        """
        self.domain_bounds = domain_bounds
        self.resample_period = resample_period
        self.seed = seed
        self._rng = np.random.RandomState(seed)
        self._cached_points = None
        self._last_resample_iteration = -1

    def _generate_random_points(self, num_points: int, mesh: Any) -> np.ndarray:
        """Generate uniform random points using rejection sampling.

        Args:
            num_points: Target number of points inside domain
            mesh: NGSolve mesh for domain checking

        Returns:
            Array of shape (num_points, 2) with valid interior points
        """
        domain_min, domain_max = self.domain_bounds

        valid_points = []
        max_attempts = max(1000, num_points * 20)
        attempts = 0

        while len(valid_points) < num_points and attempts < max_attempts:
            x = self._rng.uniform(domain_min, domain_max)
            y = self._rng.uniform(domain_min, domain_max)

            try:
                if mesh(x, y).nr != -1:
                    valid_points.append((x, y))
            except Exception:
                pass
            attempts += 1

        if len(valid_points) < num_points:
            print(f"Warning: Generated {len(valid_points)}/{num_points} random points")

        if len(valid_points) == 0:
            raise ValueError("Could not generate any valid random points in the domain")

        return np.array(valid_points[:num_points])

    def get_collocation_points(
        self,
        mesh: Any,
        model: Optional[Any] = None,
        iteration: int = 0,
        num_points: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate random collocation points, resampling periodically.

        Args:
            mesh: Mesh object for domain checking
            model: Optional model (not used)
            iteration: Current iteration (used for resample schedule)
            num_points: Number of points. If None, matches mesh vertex count.

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
            points = self._generate_random_points(num_points, mesh)
            self._cached_points = points
            self._last_resample_iteration = iteration
        else:
            points = self._cached_points

        x = torch.tensor(points[:, 0], dtype=torch.float32)
        y = torch.tensor(points[:, 1], dtype=torch.float32)

        return x, y

    def refine_mesh(
        self, mesh: Any, model: Any, iteration: int = 0
    ) -> Tuple[Any, bool]:
        """No mesh refinement for Random-R.

        Returns:
            (mesh, False) - mesh unchanged
        """
        return mesh, False
