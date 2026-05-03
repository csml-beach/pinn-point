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

    def _create_sampler(self, seed: int, dim: int = 2):
        """Create the quasi-random sampler. Override in subclasses."""
        raise NotImplementedError

    def _generate_quasi_random_points(
        self, num_points: int, mesh: Any, iteration: int | None = None
    ) -> np.ndarray:
        """Generate quasi-random points using bounding box + rejection sampling.

        Each iteration uses a non-overlapping section of the global Halton/Sobol
        sequence, so points differ across iterations while preserving the
        low-discrepancy coverage guarantee within each iteration.

        Args:
            num_points: Target number of points inside domain
            mesh: NGSolve mesh for domain checking
            iteration: Current training iteration (used to advance the sequence)

        Returns:
            Array of shape (num_points, D) with valid interior points
        """
        if not HAS_QMC:
            raise ImportError("scipy.stats.qmc required for quasi-random sampling")

        is_3d = getattr(mesh, 'dim', 2) == 3

        if is_3d:
            points_array = np.array([v.point for v in mesh.vertices])
            mins = np.min(points_array, axis=0)
            maxs = np.max(points_array, axis=0)
            ranges = maxs - mins
        else:
            x_min, x_max, y_min, y_max = self.domain_bounds
            mins = np.array([x_min, y_min])
            ranges = np.array([x_max - x_min, y_max - y_min])

        dim = 3 if is_3d else 2
        batch_size = max(1024, num_points * 4)  # Oversample to account for rejection
        max_batches = 50

        # Reserve a full rejection-sampling window per iteration. Advancing by
        # only one batch can overlap if an earlier iteration needed extra
        # rejection batches to reach the requested accepted-point count.
        iter_advance = int(iteration or 0) * batch_size * max_batches

        # A single sampler shared across rejection batches so all batches
        # continue the same sequence (no duplicates, LDS property preserved).
        sampler = self._create_sampler(self.seed, dim=dim)
        if iter_advance > 0:
            sampler.fast_forward(iter_advance)

        def batch_generator(size: int) -> np.ndarray:
            unit_points = sampler.random(size)
            return unit_points * ranges + mins

        return sample_points_in_domain(
            mesh,
            num_points,
            batch_generator,
            batch_size=batch_size,
            max_batches=max_batches,
            warn_label="quasi-random points",
            method_name=self.name,
            iteration=iteration,
        )

    def get_collocation_points(
        self,
        mesh: Any,
        model: Optional[Any] = None,
        iteration: int = 0,
        num_points: Optional[int] = None,
    ) -> Tuple[torch.Tensor, ...]:
        """Generate quasi-random collocation points.

        A fresh, non-overlapping section of the low-discrepancy sequence is
        drawn each iteration so that Halton/Sobol resample on the same schedule
        as all other methods (including random).

        Args:
            mesh: Mesh object for domain checking
            model: Optional model (not used)
            iteration: Current iteration
            num_points: Number of points. If None, matches mesh vertex count.

        Returns:
            Tuple of tensors
        """
        if num_points is None:
            num_points = len(list(mesh.vertices))

        points = self._generate_quasi_random_points(
            num_points, mesh, iteration=iteration
        )
        return points_to_tensors(points)


class HaltonMethod(QuasiRandomMethod):
    """Halton sequence sampling method.

    The Halton sequence is a low-discrepancy sequence that provides
    better coverage than pseudo-random sampling.
    """

    name = "halton"
    description = "Halton low-discrepancy sequence sampling"

    def _create_sampler(self, seed: int, dim: int = 2):
        """Create Halton sampler."""
        return qmc.Halton(d=dim, seed=seed)


class SobolMethod(QuasiRandomMethod):
    """Sobol sequence sampling method.

    The Sobol sequence is another low-discrepancy sequence, often
    providing slightly better uniformity than Halton for smaller point counts.
    """

    name = "sobol"
    description = "Sobol low-discrepancy sequence sampling"

    def _create_sampler(self, seed: int, dim: int = 2):
        """Create Sobol sampler."""
        return qmc.Sobol(d=dim, seed=seed)
