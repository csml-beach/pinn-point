"""
Quasi-random sampling methods (Halton, Sobol).

Low-discrepancy sequences provide better coverage than uniform random sampling.
These methods use scipy.stats.qmc for generating quasi-random points.
"""

import torch
import numpy as np
from typing import Tuple, Optional, Any
from .base import TrainingMethod
from .sampling import points_to_tensors, sample_points_in_domain

try:
    from scipy.stats import qmc

    HAS_QMC = True
except ImportError:
    HAS_QMC = False
    print("Warning: scipy.stats.qmc not available. Quasi-random methods disabled.")


class QuasiRandomMethod(TrainingMethod):
    """Base class for quasi-random sampling methods.

    Uses low-discrepancy sequences (Halton, Sobol) with bounding box + rejection
    sampling to generate well-distributed points within complex geometries.
    """

    name = "quasi_random"
    description = "Base quasi-random sampling"

    def __init__(
        self,
        domain_bounds: Tuple[float, float, float, float] = (0.0, 5.0, 0.0, 5.0),
        seed: int = 42,
    ):
        """Initialize quasi-random method.

        Args:
            domain_bounds: (min, max) for both x and y coordinates
            seed: Random seed for reproducibility
        """
        self.domain_bounds = domain_bounds
        self.seed = seed
        self._cached_points = None
        self._cached_num_points = None

    def _create_sampler(self, seed: int):
        """Create the quasi-random sampler. Override in subclasses."""
        raise NotImplementedError

    def _generate_quasi_random_points(self, num_points: int, mesh: Any) -> np.ndarray:
        """Generate quasi-random points using bounding box + rejection sampling.

        Args:
            num_points: Target number of points inside domain
            mesh: NGSolve mesh for domain checking

        Returns:
            Array of shape (num_points, 2) with valid interior points
        """
        if not HAS_QMC:
            raise ImportError("scipy.stats.qmc required for quasi-random sampling")

        x_min, x_max, y_min, y_max = self.domain_bounds
        x_range = x_max - x_min
        y_range = y_max - y_min

        batch_size = num_points * 4  # Oversample to account for rejection
        offset = 0

        def batch_generator(size: int) -> np.ndarray:
            nonlocal offset
            sampler = self._create_sampler(self.seed + offset)
            unit_points = sampler.random(size)
            scaled_points = np.empty_like(unit_points)
            scaled_points[:, 0] = unit_points[:, 0] * x_range + x_min
            scaled_points[:, 1] = unit_points[:, 1] * y_range + y_min
            offset += size
            return scaled_points

        return sample_points_in_domain(
            mesh,
            num_points,
            batch_generator,
            batch_size=batch_size,
            max_batches=50,
            warn_label="quasi-random points",
        )

    def get_collocation_points(
        self,
        mesh: Any,
        model: Optional[Any] = None,
        iteration: int = 0,
        num_points: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate quasi-random collocation points.

        Points are cached for efficiency - same points used across iterations
        unless num_points changes.

        Args:
            mesh: Mesh object for domain checking
            model: Optional model (not used)
            iteration: Current iteration
            num_points: Number of points. If None, matches mesh vertex count.

        Returns:
            Tuple of (x, y) tensors
        """
        if num_points is None:
            num_points = len(list(mesh.vertices))

        # Use cached points if available and size matches
        if self._cached_points is not None and self._cached_num_points == num_points:
            points = self._cached_points
        else:
            points = self._generate_quasi_random_points(num_points, mesh)
            self._cached_points = points
            self._cached_num_points = num_points

        return points_to_tensors(points)


class HaltonMethod(QuasiRandomMethod):
    """Halton sequence sampling method.

    The Halton sequence is a low-discrepancy sequence that provides
    better coverage than pseudo-random sampling.
    """

    name = "halton"
    description = "Halton low-discrepancy sequence sampling"

    def _create_sampler(self, seed: int):
        """Create Halton sampler."""
        return qmc.Halton(d=2, seed=seed)


class SobolMethod(QuasiRandomMethod):
    """Sobol sequence sampling method.

    The Sobol sequence is another low-discrepancy sequence with
    excellent uniformity properties.
    """

    name = "sobol"
    description = "Sobol low-discrepancy sequence sampling"

    def _create_sampler(self, seed: int):
        """Create Sobol sampler."""
        return qmc.Sobol(d=2, seed=seed)
