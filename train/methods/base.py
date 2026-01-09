"""
Base class for training methods.

This module defines the abstract interface for collocation point selection
and mesh refinement strategies.
"""

from abc import ABC, abstractmethod
import torch
from typing import Tuple, Optional, Any, List


class TrainingMethod(ABC):
    """Abstract base class for PINN training methods.

    A training method defines how collocation points are selected and
    optionally how the mesh is refined during iterative training.

    Subclasses must implement:
        - get_collocation_points: Select points for PDE residual evaluation
        - refine_mesh: Optionally refine mesh based on error indicators

    Attributes:
        name: Identifier for the method (used in filenames, logs)
        description: Human-readable description
    """

    name: str = "base"
    description: str = "Abstract base training method"

    @abstractmethod
    def get_collocation_points(
        self, mesh: Any, model: Optional[Any] = None, iteration: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get collocation points for PDE residual evaluation.

        Args:
            mesh: Current mesh object
            model: Optional trained model (for adaptive methods)
            iteration: Current iteration number

        Returns:
            Tuple of (x, y) tensors containing point coordinates
        """
        pass

    @abstractmethod
    def refine_mesh(
        self, mesh: Any, model: Any, iteration: int = 0
    ) -> Tuple[Any, bool]:
        """Optionally refine the mesh based on error indicators.

        Args:
            mesh: Current mesh object
            model: Trained model for computing error indicators
            iteration: Current iteration number

        Returns:
            Tuple of (new_mesh, was_refined) where was_refined indicates
            if any refinement was performed
        """
        pass

    def should_refine(self, iteration: int, max_iterations: int) -> bool:
        """Check if refinement should happen at this iteration.

        Default implementation refines at every iteration except the last.
        Override for custom refinement schedules.

        Args:
            iteration: Current iteration (0-indexed)
            max_iterations: Total number of iterations

        Returns:
            True if mesh should be refined
        """
        return iteration < max_iterations - 1

    def get_error_indicators(self, mesh: Any, model: Any) -> torch.Tensor:
        """Compute element-wise error indicators.

        Default returns None (no indicators). Override for adaptive methods.

        Args:
            mesh: Current mesh
            model: Trained model

        Returns:
            Tensor of error indicators per element, or None
        """
        return None

    def log_iteration(self, iteration: int, mesh: Any, model: Any) -> dict:
        """Log method-specific information for this iteration.

        Override to add custom logging for experiments.

        Args:
            iteration: Current iteration
            mesh: Current mesh
            model: Current model

        Returns:
            Dictionary of loggable values
        """
        return {
            "method": self.name,
            "iteration": iteration,
            "num_vertices": len(mesh.vertices) if hasattr(mesh, "vertices") else 0,
        }
