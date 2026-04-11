"""
Steady advection-diffusion-reaction problem.

Solves the PDE
    -eps * ∇²u + b·∇u + c*u = f(x,y)  in Ω

on the same perforated square geometry used by the Poisson benchmark. The
outer boundaries are Dirichlet:
    u = g_left(y) on x = 0
    u = 0         on top, bottom, and right outer boundaries

Internal hole boundaries inherit the natural boundary condition from the weak
form. The inflow profile and source bumps are chosen to create transport
through the narrow channels of the geometry while keeping the FEM reference
solve straightforward.
"""

from typing import Any, Tuple

import torch
from ngsolve import *

from .base import PDEProblem


class AdvectionDiffusionProblem(PDEProblem):
    """Steady 2D advection-diffusion-reaction problem on the perforated domain."""

    name = "advection_diffusion"
    description = (
        "Steady advection-diffusion-reaction equation with left inflow and "
        "localized sources on the perforated square domain"
    )

    _SOURCE_BUMPS = (
        (1.55, 1.10, 3.00, 0.18),
        (2.55, 3.80, -2.60, 0.16),
        (3.85, 2.05, 2.40, 0.18),
    )
    _INFLOW_BUMPS = (
        (1.10, 1.20, 0.18),
        (3.90, 0.95, 0.18),
    )

    def __init__(
        self,
        domain_size: float = 5.0,
        diffusion: float = 0.02,
        velocity_x: float = 1.8,
        velocity_y: float = 0.10,
        reaction: float = 0.05,
    ):
        self.domain_size = float(domain_size)
        self.diffusion = float(diffusion)
        self.velocity_x = float(velocity_x)
        self.velocity_y = float(velocity_y)
        self.reaction = float(reaction)

    def _source_term_torch(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        source = torch.zeros_like(x)
        for x0, y0, amplitude, sigma in self._SOURCE_BUMPS:
            radius_sq = (x - x0) ** 2 + (y - y0) ** 2
            source = source + amplitude * torch.exp(
                -radius_sq / (2.0 * sigma * sigma)
            )
        return source

    def _source_term_ngsolve(self):
        source = 0
        for x0, y0, amplitude, sigma in self._SOURCE_BUMPS:
            radius_sq = (x - x0) * (x - x0) + (y - y0) * (y - y0)
            source = source + amplitude * exp(-radius_sq / (2.0 * sigma * sigma))
        return source

    def _left_boundary_profile_torch(self, y: torch.Tensor) -> torch.Tensor:
        profile = torch.zeros_like(y)
        for y0, amplitude, sigma in self._INFLOW_BUMPS:
            profile = profile + amplitude * torch.exp(
                -((y - y0) ** 2) / (2.0 * sigma * sigma)
            )
        return profile

    def _left_boundary_profile_ngsolve(self):
        profile = 0
        for y0, amplitude, sigma in self._INFLOW_BUMPS:
            profile = profile + amplitude * exp(
                -((y - y0) * (y - y0)) / (2.0 * sigma * sigma)
            )
        return profile

    def pde_residual(
        self, model: Any, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """Compute residual of -eps*Δu + b·∇u + c*u - f = 0."""
        from config import DEVICE

        x = torch.as_tensor(x, dtype=torch.float32, device=DEVICE)
        y = torch.as_tensor(y, dtype=torch.float32, device=DEVICE)
        x = x.clone().detach().requires_grad_(True)
        y = y.clone().detach().requires_grad_(True)

        u = model.forward(x, y).reshape(-1)

        du_dx = self.compute_derivative(u, x, 1)
        du_dy = self.compute_derivative(u, y, 1)
        d2u_dx2 = self.compute_derivative(u, x, 2)
        d2u_dy2 = self.compute_derivative(u, y, 2)

        diffusion_term = -self.diffusion * (d2u_dx2 + d2u_dy2)
        advection_term = self.velocity_x * du_dx + self.velocity_y * du_dy
        reaction_term = self.reaction * u

        return diffusion_term + advection_term + reaction_term - self._source_term_torch(
            x, y
        )

    def boundary_loss(self, model: Any, num_boundary_points: int = 100) -> torch.Tensor:
        """Enforce Dirichlet data on the outer boundaries."""
        from config import DEVICE

        domain_size = self.domain_size
        side_points = max(1, num_boundary_points // 4)
        left_points = max(1, num_boundary_points - 3 * side_points)

        y_left = torch.linspace(0, domain_size, left_points, device=DEVICE)
        x_left = torch.zeros(left_points, device=DEVICE)
        u_left = model.forward(x_left, y_left).reshape(-1)
        target_left = self._left_boundary_profile_torch(y_left)

        x_right = torch.full((side_points,), domain_size, device=DEVICE)
        y_right = torch.linspace(0, domain_size, side_points, device=DEVICE)
        u_right = model.forward(x_right, y_right).reshape(-1)

        x_bottom = torch.linspace(0, domain_size, side_points, device=DEVICE)
        y_bottom = torch.zeros(side_points, device=DEVICE)
        u_bottom = model.forward(x_bottom, y_bottom).reshape(-1)

        x_top = torch.linspace(0, domain_size, side_points, device=DEVICE)
        y_top = torch.full((side_points,), domain_size, device=DEVICE)
        u_top = model.forward(x_top, y_top).reshape(-1)

        loss_left = torch.mean(torch.square(u_left - target_left))
        loss_right = torch.mean(torch.square(u_right))
        loss_bottom = torch.mean(torch.square(u_bottom))
        loss_top = torch.mean(torch.square(u_top))
        return 0.25 * (loss_left + loss_right + loss_bottom + loss_top)

    def source_term(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = torch.as_tensor(x, dtype=torch.float32)
        y = torch.as_tensor(y, dtype=torch.float32)
        return self._source_term_torch(x, y)

    def solve_fem(self, mesh) -> Tuple[Any, Any]:
        """Solve the steady advection-diffusion-reaction system with FEM."""
        dirichlet_boundaries = "left|right|top|bottom"
        fes = H1(mesh, order=1, dirichlet=dirichlet_boundaries, autoupdate=True)

        u = fes.TrialFunction()
        v = fes.TestFunction()
        beta = CoefficientFunction((self.velocity_x, self.velocity_y))
        source = self._source_term_ngsolve()

        a = BilinearForm(
            self.diffusion * InnerProduct(grad(u), grad(v)) * dx
            + InnerProduct(beta, grad(u)) * v * dx
            + self.reaction * u * v * dx
        )
        f = LinearForm(source * v * dx)

        a.Assemble()
        f.Assemble()

        gfu = GridFunction(fes)
        left_profile = self._left_boundary_profile_ngsolve()

        gfu.Set(0, definedon=mesh.Boundaries("right"))
        gfu.Set(0, definedon=mesh.Boundaries("top"))
        gfu.Set(0, definedon=mesh.Boundaries("bottom"))
        gfu.Set(left_profile, definedon=mesh.Boundaries("left"))

        residual = f.vec.CreateVector()
        residual.data = f.vec - a.mat * gfu.vec
        gfu.vec.data += a.mat.Inverse(freedofs=fes.FreeDofs()) * residual

        return gfu, fes

    def get_domain_bounds(self) -> Tuple[float, float, float, float]:
        return (0.0, self.domain_size, 0.0, self.domain_size)

    def get_dirichlet_boundaries(self) -> list:
        return ["left", "right", "top", "bottom"]
