"""
Poisson equation problem.

Solves the Poisson equation:
    -∇²u = f(x,y)  in Ω
    u = 0          on Γ_D (Dirichlet boundary)

where f(x,y) is a localized multi-bump source term.

This is the default problem used in the PINN adaptive mesh refinement experiments.
"""

import torch
from typing import Tuple, Any
from ngsolve import *
from .base import PDEProblem


class PoissonProblem(PDEProblem):
    """Poisson equation with a localized multi-bump source term.

    Domain: Perforated square domain with repeated cutout patterns
    Boundary conditions: Dirichlet u=0 on bottom boundary
    """

    name = "poisson"
    description = (
        "Poisson equation: -∇²u = sum of localized Gaussian bumps with Dirichlet BC on bottom"
    )

    _BUMPS = (
        (1.67, 4.15, 18.0, 0.18),
        (3.33, 4.15, 18.0, 0.18),
        (2.50, 3.35, 10.0, 0.22),
    )

    def __init__(self, domain_size: float = 5.0):
        """Initialize Poisson problem.

        Args:
            domain_size: Size of the square domain (default 5.0)
        """
        self.domain_size = domain_size

    def _source_term_torch(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        source = torch.zeros_like(x)
        for x0, y0, amplitude, sigma in self._BUMPS:
            radius_sq = (x - x0) ** 2 + (y - y0) ** 2
            source = source + amplitude * torch.exp(
                -radius_sq / (2.0 * sigma * sigma)
            )
        return source

    def _source_term_ngsolve(self):
        source = 0
        for x0, y0, amplitude, sigma in self._BUMPS:
            radius_sq = (x - x0) * (x - x0) + (y - y0) * (y - y0)
            source = source + amplitude * exp(-radius_sq / (2.0 * sigma * sigma))
        return source

    def pde_residual(
        self, model: Any, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """Compute PDE residual: ∇²u + f = 0.

        Args:
            model: Neural network with forward(x, y) method
            x: x-coordinates (will be made differentiable)
            y: y-coordinates (will be made differentiable)

        Returns:
            Residual values (should be zero when PDE is satisfied)
        """
        from config import DEVICE

        # Ensure tensors are on device and require gradients
        x = torch.as_tensor(x, dtype=torch.float32, device=DEVICE)
        y = torch.as_tensor(y, dtype=torch.float32, device=DEVICE)
        x = x.clone().detach().requires_grad_(True)
        y = y.clone().detach().requires_grad_(True)

        # Forward pass
        u = model.forward(x, y)

        # Compute second derivatives
        d2u_dx2 = self.compute_derivative(u, x, 2)
        d2u_dy2 = self.compute_derivative(u, y, 2)

        # PDE residual: ∇²u + f = 0
        residual = d2u_dx2 + d2u_dy2 + self._source_term_torch(x, y)

        return residual

    def boundary_loss(self, model: Any, num_boundary_points: int = 100) -> torch.Tensor:
        """Compute Dirichlet boundary condition loss on bottom boundary.

        Enforces u = 0 on y = 0 (bottom boundary).

        Args:
            model: Neural network
            num_boundary_points: Number of points to sample

        Returns:
            Mean squared error of boundary predictions
        """
        from config import DEVICE

        # Bottom boundary: y = 0, x in [0, domain_size]
        x_bottom = torch.linspace(
            0, self.domain_size, num_boundary_points, device=DEVICE
        )
        y_bottom = torch.zeros(num_boundary_points, device=DEVICE)

        # Predict values on boundary
        u_pred = model.forward(x_bottom, y_bottom)

        # Dirichlet BC: u = 0
        loss_bc = torch.mean(torch.square(u_pred))

        return loss_bc

    def source_term(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Evaluate the localized source term f(x,y).

        Args:
            x: x-coordinates
            y: y-coordinates

        Returns:
            Source term values
        """
        x = torch.as_tensor(x, dtype=torch.float32)
        y = torch.as_tensor(y, dtype=torch.float32)
        return self._source_term_torch(x, y)

    def solve_fem(self, mesh) -> Tuple[Any, Any]:
        """Solve Poisson equation using NGSolve FEM.

        Args:
            mesh: NGSolve mesh object

        Returns:
            Tuple of (gfu, fes) - GridFunction solution and FE space
        """
        # H1-conforming finite element space
        fes = H1(mesh, order=1, dirichlet="bottom", autoupdate=True)

        # Trial and test functions
        u = fes.TrialFunction()
        v = fes.TestFunction()

        # Bilinear form: a(u,v) = ∫ ∇u·∇v dx
        a = BilinearForm(grad(u) * grad(v) * dx)

        # Linear form: f(v) = ∫ f*v dx for the localized source term
        f_source = self._source_term_ngsolve()
        f = LinearForm(f_source * v * dx)

        # Assemble and solve
        a.Assemble()
        f.Assemble()

        gfu = GridFunction(fes)
        gfu.vec.data = a.mat.Inverse(freedofs=fes.FreeDofs()) * f.vec

        return gfu, fes

    def get_domain_bounds(self) -> Tuple[float, float, float, float]:
        """Get domain bounding box.

        Returns:
            (x_min, x_max, y_min, y_max)
        """
        return (0.0, self.domain_size, 0.0, self.domain_size)

    def get_dirichlet_boundaries(self) -> list:
        """Get Dirichlet boundary names."""
        return ["bottom"]
