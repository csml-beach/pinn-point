"""
Random sampling method (baseline).

This method uses uniform random sampling for collocation point selection.
It serves as the baseline comparison for adaptive methods.
"""

import torch
import numpy as np
from typing import Tuple, Optional, Any
from .base import TrainingMethod


class RandomMethod(TrainingMethod):
    """Uniform random sampling method.

    Samples collocation points uniformly at random within the domain.
    The mesh is not refined; instead, new random points are generated
    each iteration to match the adaptive method's point count.
    """

    name = "random"
    description = "Uniform random point sampling (baseline)"

    def __init__(self, domain_bounds: Tuple[float, float] = (0.0, 5.0)):
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

        domain_min, domain_max = self.domain_bounds

        # Generate random points, rejecting those outside the domain
        random_points = []
        max_attempts = num_points * 20
        attempts = 0

        while len(random_points) < num_points and attempts < max_attempts:
            x = np.random.uniform(domain_min, domain_max)
            y = np.random.uniform(domain_min, domain_max)

            # Check if point is inside mesh domain
            try:
                if mesh(x, y).nr != -1:
                    random_points.append((x, y))
            except Exception:
                pass
            attempts += 1

        if len(random_points) < num_points:
            print(
                f"Warning: Generated {len(random_points)}/{num_points} points after {attempts} attempts"
            )

        if len(random_points) == 0:
            raise ValueError("Could not generate any valid random points in the domain")

        points = np.array(random_points)
        x = torch.tensor(points[:, 0], dtype=torch.float32)
        y = torch.tensor(points[:, 1], dtype=torch.float32)

        return x, y

    def refine_mesh(
        self, mesh: Any, model: Any, iteration: int = 0
    ) -> Tuple[Any, bool]:
        """Random method does not refine mesh.

        Returns the mesh unchanged.

        Args:
            mesh: Current mesh
            model: Model (unused)
            iteration: Current iteration

        Returns:
            Tuple of (same_mesh, False) indicating no refinement
        """
        return mesh, False

    def should_refine(self, iteration: int, max_iterations: int) -> bool:
        """Random method never refines."""
        return False

    def log_iteration(self, iteration: int, mesh: Any, model: Any) -> dict:
        """Log random method information."""
        base_log = super().log_iteration(iteration, mesh, model)
        base_log["sampling_strategy"] = "uniform_random"
        return base_log
