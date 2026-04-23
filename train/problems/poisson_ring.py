"""
Poisson equation on an eccentric annulus.

Solves
    -Δu = f(x, y)  in Ω
    u = 0          on Γ_outer ∪ Γ_inner

The geometry is an annulus with an off-center inner hole. The source term is
asymmetric so the benchmark is not dominated by rotational symmetry.
"""

from __future__ import annotations

from typing import Any, Tuple

import torch
from netgen.occ import OCCGeometry, WorkPlane
from ngsolve import *

from .base import PDEProblem


class PoissonRingProblem(PDEProblem):
    """Poisson problem on an eccentric annulus."""

    name = "poisson_ring"
    description = (
        "Poisson equation on an eccentric annulus with asymmetric Gaussian sources "
        "and Dirichlet conditions on inner and outer boundaries"
    )

    _BUMPS = (
        (0.95, 0.65, 16.0, 0.18),
        (-0.45, 1.10, 12.0, 0.22),
        (-1.05, -0.35, 10.0, 0.20),
    )

    def __init__(
        self,
        outer_radius: float = 2.0,
        inner_radius: float = 0.65,
        inner_center_x: float = 0.38,
        inner_center_y: float = -0.22,
    ):
        self.outer_radius = float(outer_radius)
        self.inner_radius = float(inner_radius)
        self.inner_center_x = float(inner_center_x)
        self.inner_center_y = float(inner_center_y)

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

    def create_mesh(self, maxh=None):
        if maxh is None:
            maxh = 0.20

        outer = WorkPlane().Circle(self.outer_radius).Face()
        outer.edges.name = "outer"

        inner = (
            WorkPlane()
            .Circle(self.inner_radius)
            .Face()
            .Move((self.inner_center_x, self.inner_center_y, 0.0))
        )
        inner.edges.name = "inner"

        geo = outer - inner
        occ = OCCGeometry(geo, dim=2)
        return Mesh(occ.GenerateMesh(maxh=maxh)).Curve(3)

    def pde_residual(
        self, model: Any, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor | None = None
    ) -> torch.Tensor:
        from config import DEVICE

        x = torch.as_tensor(x, dtype=torch.float32, device=DEVICE)
        y = torch.as_tensor(y, dtype=torch.float32, device=DEVICE)
        x = x.clone().detach().requires_grad_(True)
        y = y.clone().detach().requires_grad_(True)

        u = model.forward(x, y)
        d2u_dx2 = self.compute_derivative(u, x, 2)
        d2u_dy2 = self.compute_derivative(u, y, 2)
        return d2u_dx2 + d2u_dy2 + self._source_term_torch(x, y)

    def boundary_loss(self, model: Any, num_boundary_points: int = 100) -> torch.Tensor:
        from config import DEVICE

        outer_count = max(1, num_boundary_points // 2)
        inner_count = max(1, num_boundary_points - outer_count)

        theta_outer = torch.linspace(
            0.0,
            2.0 * torch.pi,
            outer_count + 1,
            device=DEVICE,
        )[:-1]
        x_outer = self.outer_radius * torch.cos(theta_outer)
        y_outer = self.outer_radius * torch.sin(theta_outer)
        u_outer = model.forward(x_outer, y_outer).reshape(-1)

        theta_inner = torch.linspace(
            0.0,
            2.0 * torch.pi,
            inner_count + 1,
            device=DEVICE,
        )[:-1]
        x_inner = self.inner_center_x + self.inner_radius * torch.cos(theta_inner)
        y_inner = self.inner_center_y + self.inner_radius * torch.sin(theta_inner)
        u_inner = model.forward(x_inner, y_inner).reshape(-1)

        return 0.5 * (
            torch.mean(torch.square(u_outer)) + torch.mean(torch.square(u_inner))
        )

    def source_term(
        self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = torch.as_tensor(x, dtype=torch.float32)
        y = torch.as_tensor(y, dtype=torch.float32)
        return self._source_term_torch(x, y)

    def solve_fem(self, mesh) -> Tuple[Any, Any]:
        fes = H1(mesh, order=1, dirichlet="outer|inner", autoupdate=True)

        u = fes.TrialFunction()
        v = fes.TestFunction()
        a = BilinearForm(grad(u) * grad(v) * dx)
        f = LinearForm(self._source_term_ngsolve() * v * dx)

        a.Assemble()
        f.Assemble()

        gfu = GridFunction(fes)
        gfu.vec.data = a.mat.Inverse(freedofs=fes.FreeDofs()) * f.vec
        return gfu, fes

    def get_domain_bounds(self) -> Tuple[float, float, float, float]:
        r = self.outer_radius
        return (-r, r, -r, r)

    def get_dirichlet_boundaries(self) -> list:
        return ["outer", "inner"]
