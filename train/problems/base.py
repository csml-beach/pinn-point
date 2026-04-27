"""
Base class for PDE problems.

This module defines the abstract interface that all PDE problems must implement.
"""

from abc import ABC, abstractmethod
from typing import Any, Tuple

import torch
from torch.utils.data import TensorDataset


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
    input_dim: int = 2
    output_dim: int = 1
    output_names: tuple[str, ...] = ("u",)
    has_time_input: bool = False

    @abstractmethod
    def pde_residual(
        self, model: Any, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Compute the PDE residual at given points.

        The residual should be zero when the PDE is satisfied exactly.
        For a PDE of the form L[u] = f, the residual is L[u] - f.

        Args:
            model: Neural network model with forward(...) method
            x: x-coordinates (requires_grad=True for autograd)
            y: y-coordinates (requires_grad=True for autograd)
            t: Optional time coordinates for transient problems

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
    def source_term(
        self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Evaluate the source term f(x, y) at given points.

        Args:
            x: x-coordinates
            y: y-coordinates
            t: Optional time coordinates for transient problems

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

    def create_mesh(self, maxh=None):
        """Create a mesh for this problem.

        The default implementation delegates to the existing project geometry.
        Override in a problem subclass when the PDE owns a different geometry.
        """
        from geometry import create_initial_mesh

        return create_initial_mesh(maxh=maxh)

    def get_time_bounds(self) -> Tuple[float, float] | None:
        """Return temporal bounds for transient problems, else None."""
        return None

    def augment_collocation_points(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        *,
        mesh: Any | None = None,
        iteration: int = 0,
        seed: int | None = None,
        purpose: str = "train",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Optionally attach extra coordinates (for example time) to collocation points."""
        return x, y, None

    def get_loss_weight_overrides(self) -> dict[str, float]:
        """Optional problem-specific overrides for model loss weights."""
        return {}

    def data_loss(
        self,
        model: Any,
        coordinates: torch.Tensor,
        targets: torch.Tensor,
        dataset: TensorDataset | None = None,
    ) -> torch.Tensor:
        """Compute supervised loss on problem-owned targets.

        Default behavior is full-output MSE against the provided targets.
        Problems can override this when only part of the state is supervised.
        """
        predictions = model.forward(coordinates)
        targets = torch.as_tensor(
            targets,
            dtype=predictions.dtype,
            device=predictions.device,
        )
        return torch.mean(torch.square(predictions - targets))

    def create_training_dataset(
        self, mesh, fem_solution: Any | None = None, seed: int | None = None
    ) -> TensorDataset | None:
        """Optionally build a problem-specific supervised dataset.

        Returning None falls back to the legacy static scalar dataset path.
        """
        return None

    def get_collocation_budget(
        self,
        initial_mesh,
        vertex_array: torch.Tensor,
        training_dataset: TensorDataset | None = None,
    ) -> int | None:
        """Optionally override the fixed accepted collocation budget.

        Returning None falls back to the legacy behavior of using the initial
        mesh vertex count as the collocation budget.
        """
        return None

    def create_smoke_collocation_points(
        self, mesh, seed: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None] | None:
        """Optionally build problem-specific PINN smoke collocation points.

        Returning None falls back to the legacy mesh-vertex collocation path.
        """
        return None

    def create_reference_solution(self, mesh_size_factor: float = 0.05):
        """Optionally build a problem-specific reference solution payload."""
        return None

    def build_geometry_smoke_metadata(self, mesh) -> dict[str, Any] | None:
        """Optionally provide problem-specific geometry-smoke artifacts/metadata."""
        return None

    def build_fem_smoke_metadata(
        self,
        mesh,
        *,
        dt: float | None = None,
        t_end: float | None = None,
    ) -> dict[str, Any] | None:
        """Optionally provide problem-specific FEM-smoke artifacts/metadata."""
        return None

    def evaluate_model_against_reference(
        self,
        model: Any,
        reference_mesh: Any,
        reference_solution: Any,
        *,
        export_images: bool = False,
        iteration: int | None = None,
    ) -> bool:
        """Optionally evaluate a model against a problem-specific reference payload.

        Return True when evaluation was handled by the problem.
        """
        return False

    def export_fem_solution(self, mesh, gfu) -> torch.Tensor:
        """Evaluate a FEM solution at mesh vertices.

        This avoids assuming that `gfu.vec` ordering always matches vertex order.
        """
        from geometry import export_vertex_coordinates

        coords = export_vertex_coordinates(mesh).detach().cpu().numpy()
        values = []
        for x_coord, y_coord in coords:
            try:
                values.append(float(gfu(mesh(float(x_coord), float(y_coord)))))
            except TypeError:
                values.append(float(gfu(float(x_coord), float(y_coord))))
        return torch.tensor(values, dtype=torch.float32)

    def get_sampling_bounds(self) -> Tuple[float, float, float, float]:
        """Get the axis-aligned bounds used by collocation samplers."""
        return self.get_domain_bounds()

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
