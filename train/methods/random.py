"""
Random sampling method (baseline).

This method uses uniform random sampling for collocation point selection.
It serves as the baseline comparison for adaptive methods.
"""

import torch
import numpy as np
from typing import Tuple, Optional, Any
from .base import TrainingMethod
from .sampling import points_to_tensors, sample_points_in_domain


class RandomMethod(TrainingMethod):
    """Uniform random sampling method.

    Samples collocation points uniformly at random within the domain.
    The mesh is not refined; instead, new random points are generated
    each iteration to match the adaptive method's point count.
    """

    name = "random"
    description = "Uniform random point sampling (baseline)"

    def __init__(
        self, domain_bounds: Tuple[float, float, float, float] = (0.0, 5.0, 0.0, 5.0)
    ):
        """Initialize random method.

        Args:
            domain_bounds: (min, max) for both x and y coordinates
        """
        self.domain_bounds = domain_bounds

    def get_collocation_points(
        self,
        mesh: Any,
        model: Optional[Any] = None,
        iteration: int = 0,
        num_points: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate random collocation points within the domain.

        Args:
            mesh: Mesh object (used to check if points are inside domain)
            model: Optional model (not used)
            iteration: Current iteration
            num_points: Number of points to generate. If None, matches mesh vertex count.

        Returns:
            Tuple of (x, y) tensors with random coordinates inside domain
        """
        if num_points is None:
            num_points = len(mesh.vertices)

        x_min, x_max, y_min, y_max = self.domain_bounds

        points = sample_points_in_domain(
            mesh,
            num_points,
            lambda batch_size: np.column_stack(
                (
                    np.random.uniform(x_min, x_max, size=batch_size),
                    np.random.uniform(y_min, y_max, size=batch_size),
                )
            ),
            max_batches=20,
            warn_label="random points",
            method_name=self.name,
            iteration=iteration,
        )
        return points_to_tensors(points)

    def should_refine(self, iteration: int, max_iterations: int) -> bool:
        """Random method never refines."""
        return False

    def log_iteration(self, iteration: int, mesh: Any, model: Any) -> dict:
        """Log random method information."""
        base_log = super().log_iteration(iteration, mesh, model)
        base_log["sampling_strategy"] = "uniform_random"
        return base_log
