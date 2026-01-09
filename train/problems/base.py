"""
Base class for PDE problems.

This module defines the abstract interface that all PDE problems must implement.
"""

from abc import ABC, abstractmethod
import torch
from typing import Tuple, Optional, Any


class PDEProblem(ABC):
    """Abstract base class for PDE problems.

    Subclasses must implement:
        - pde_residual: Compute PDE residual for PINN loss
        - boundary_loss: Compute boundary condition loss
        - source_term: Return the source term function
        - solve_fem: Solve the problem using FEM
        - get_domain_bounds: Return domain bounds for the problem

    Attributes:
        name: Human-readable name of the problem
        description: Brief description of the PDE
    """

    name: str = "base"
    description: str = "Abstract base PDE problem"

    @abstractmethod
    def pde_residual(
        self, model: Any, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """Compute the PDE residual at given points.

        The residual should be zero when the PDE is satisfied exactly.
        For a PDE of the form L[u] = f, the residual is L[u] - f.

        Args:
            model: Neural network model with forward(x, y) method
            x: x-coordinates (requires_grad=True for autograd)
            y: y-coordinates (requires_grad=True for autograd)

        Returns:
            Tensor of residual values at each point
        """
        pass

    @abstractmethod
    def boundary_loss(self, model: Any, num_boundary_points: int = 100) -> torch.Tensor:
        """Compute the boundary condition loss.

        Args:
            model: Neural network model
            num_boundary_points: Number of points to sample on boundary

        Returns:
            Scalar loss value for boundary conditions
        """
        pass

    @abstractmethod
    def source_term(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Evaluate the source term f(x, y) at given points.

        Args:
            x: x-coordinates
            y: y-coordinates

        Returns:
            Source term values at each point
        """
        pass

    @abstractmethod
    def solve_fem(self, mesh) -> Tuple[Any, Any]:
        """Solve the problem using finite element method.

        Args:
            mesh: NGSolve mesh object

        Returns:
            Tuple of (gfu, fes) - GridFunction solution and finite element space
        """
        pass

    @abstractmethod
    def get_domain_bounds(self) -> Tuple[float, float, float, float]:
        """Get the bounding box of the problem domain.

        Returns:
            Tuple of (x_min, x_max, y_min, y_max)
        """
        pass

    def get_dirichlet_boundaries(self) -> list:
        """Get list of boundary names with Dirichlet conditions.

        Returns:
            List of boundary name strings (e.g., ['bottom'])
        """
        return ["bottom"]  # Default, can be overridden

    def compute_derivative(
        self, u: torch.Tensor, var: torch.Tensor, order: int = 1
    ) -> torch.Tensor:
        """Utility: compute n-th order derivative using autograd.

        Args:
            u: Function values
            var: Variable to differentiate with respect to
            order: Order of derivative

        Returns:
            Derivative tensor
        """
        if order == 0:
            return u

        from config import DEVICE

        du = torch.autograd.grad(
            u,
            var,
            torch.ones_like(u).to(DEVICE),
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )[0]

        if order == 1:
            return du
        return self.compute_derivative(du, var, order - 1)
