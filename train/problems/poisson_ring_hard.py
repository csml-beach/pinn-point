"""
Poisson equation on a harder eccentric multi-hole annulus.

Solves
    -Δu = f(x, y)  in Ω
    u = 0          on Γ_outer ∪ Γ_holes

The domain is a circular outer boundary with three interior circular holes:
one large eccentric hole and two smaller satellite holes that create thin
corridors. This is intended to be a harder geometric control benchmark than the
simple eccentric annulus used in ``poisson_ring``.
"""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np
import torch
from netgen.occ import OCCGeometry, WorkPlane
from ngsolve import *
from torch.utils.data import TensorDataset

from .base import PDEProblem


class PoissonRingHardProblem(PDEProblem):
    """Poisson problem on an eccentric annulus with multiple narrow corridors."""

    name = "poisson_ring_hard"
    description = (
        "Poisson equation on an eccentric annulus with three interior holes, "
        "thin corridors, and asymmetric Gaussian sources"
    )

    _BUMPS = (
        (1.18, 0.18, 18.0, 0.16),
        (-0.12, 1.36, 15.0, 0.15),
        (-1.18, -0.18, 12.0, 0.18),
        (-0.42, -1.34, 14.0, 0.15),
    )

    def __init__(
        self,
        outer_radius: float = 2.0,
        main_hole_radius: float = 0.95,
        main_hole_center_x: float = 0.55,
        main_hole_center_y: float = -0.10,
        top_hole_radius: float = 0.33,
        top_hole_center_x: float = -0.55,
        top_hole_center_y: float = 1.05,
        bottom_hole_radius: float = 0.28,
        bottom_hole_center_x: float = -0.75,
        bottom_hole_center_y: float = -1.00,
        target_dataset_size: int = 51,
        target_collocation_budget: int = 51,
    ):
        self.outer_radius = float(outer_radius)
        self.main_hole_radius = float(main_hole_radius)
        self.main_hole_center_x = float(main_hole_center_x)
        self.main_hole_center_y = float(main_hole_center_y)
        self.top_hole_radius = float(top_hole_radius)
        self.top_hole_center_x = float(top_hole_center_x)
        self.top_hole_center_y = float(top_hole_center_y)
        self.bottom_hole_radius = float(bottom_hole_radius)
        self.bottom_hole_center_x = float(bottom_hole_center_x)
        self.bottom_hole_center_y = float(bottom_hole_center_y)
        self.target_dataset_size = max(int(target_dataset_size), 1)
        self.target_collocation_budget = max(int(target_collocation_budget), 1)

    def _circle_specs(self) -> tuple[tuple[str, float, float, float], ...]:
        return (
            (
                "main_hole",
                self.main_hole_radius,
                self.main_hole_center_x,
                self.main_hole_center_y,
            ),
            (
                "top_hole",
                self.top_hole_radius,
                self.top_hole_center_x,
                self.top_hole_center_y,
            ),
            (
                "bottom_hole",
                self.bottom_hole_radius,
                self.bottom_hole_center_x,
                self.bottom_hole_center_y,
            ),
        )

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
            maxh = 0.18

        outer = WorkPlane().Circle(self.outer_radius).Face()
        outer.edges.name = "outer"

        geo = outer
        for boundary_name, radius, center_x, center_y in self._circle_specs():
            hole = WorkPlane().Circle(radius).Face().Move((center_x, center_y, 0.0))
            hole.edges.name = boundary_name
            geo = geo - hole

        occ = OCCGeometry(geo, dim=2)
        return Mesh(occ.GenerateMesh(maxh=maxh)).Curve(3)

    def pde_residual(
        self,
        model: Any,
        x: torch.Tensor,
        y: torch.Tensor,
        t: torch.Tensor | None = None,
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

        # Allocate boundary samples approximately by perimeter.
        specs = [("outer", self.outer_radius, 0.0, 0.0), *self._circle_specs()]
        radii = [spec[1] for spec in specs]
        perimeter_weights = [max(radius, 1e-8) for radius in radii]
        total_weight = sum(perimeter_weights)

        remaining = max(int(num_boundary_points), len(specs))
        losses = []
        for idx, (boundary_name, radius, center_x, center_y) in enumerate(specs):
            if idx == len(specs) - 1:
                count = remaining
            else:
                count = max(1, int(round(num_boundary_points * radius / total_weight)))
                count = min(count, remaining - (len(specs) - idx - 1))
            remaining -= count

            theta = torch.linspace(
                0.0, 2.0 * torch.pi, count + 1, device=DEVICE
            )[:-1]
            x_boundary = center_x + radius * torch.cos(theta)
            y_boundary = center_y + radius * torch.sin(theta)
            u_boundary = model.forward(x_boundary, y_boundary).reshape(-1)
            losses.append(torch.mean(torch.square(u_boundary)))

        return torch.mean(torch.stack(losses))

    def source_term(
        self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = torch.as_tensor(x, dtype=torch.float32)
        y = torch.as_tensor(y, dtype=torch.float32)
        return self._source_term_torch(x, y)

    def solve_fem(self, mesh) -> Tuple[Any, Any]:
        dirichlet = "|".join(self.get_dirichlet_boundaries())
        fes = H1(mesh, order=1, dirichlet=dirichlet, autoupdate=True)

        u = fes.TrialFunction()
        v = fes.TestFunction()
        a = BilinearForm(grad(u) * grad(v) * dx)
        f = LinearForm(self._source_term_ngsolve() * v * dx)

        a.Assemble()
        f.Assemble()

        gfu = GridFunction(fes)
        gfu.vec.data = a.mat.Inverse(freedofs=fes.FreeDofs()) * f.vec
        return gfu, fes

    def _select_covering_vertex_indices(
        self, coords: np.ndarray, count: int, seed: int | None = None
    ) -> np.ndarray:
        coords = np.asarray(coords, dtype=np.float32)
        n = len(coords)
        if count >= n:
            return np.arange(n, dtype=int)

        center = np.mean(coords, axis=0)
        start_idx = int(np.argmax(np.sum((coords - center) ** 2, axis=1)))
        selected = [start_idx]
        remaining = np.ones(n, dtype=bool)
        remaining[start_idx] = False
        min_sq_dist = np.sum((coords - coords[start_idx]) ** 2, axis=1)

        while len(selected) < count:
            candidate_indices = np.where(remaining)[0]
            next_idx = int(candidate_indices[np.argmax(min_sq_dist[candidate_indices])])
            selected.append(next_idx)
            remaining[next_idx] = False
            next_sq_dist = np.sum((coords - coords[next_idx]) ** 2, axis=1)
            min_sq_dist = np.minimum(min_sq_dist, next_sq_dist)

        return np.asarray(selected, dtype=int)

    def create_training_dataset(
        self, mesh, fem_solution: Any | None = None, seed: int | None = None
    ) -> TensorDataset:
        gfu = fem_solution
        if gfu is None:
            gfu, _ = self.solve_fem(mesh)

        vertex_coords = np.asarray(
            [(float(v.point[0]), float(v.point[1])) for v in mesh.vertices],
            dtype=np.float32,
        )
        vertex_values = self.export_fem_solution(mesh, gfu).reshape(-1, 1)
        selected_idx = self._select_covering_vertex_indices(
            vertex_coords, self.target_dataset_size, seed=seed
        )
        dataset_coordinates = torch.tensor(
            vertex_coords[selected_idx], dtype=torch.float32
        )
        dataset_targets = vertex_values[selected_idx].to(dtype=torch.float32)
        return TensorDataset(dataset_coordinates, dataset_targets)

    def get_collocation_budget(
        self,
        initial_mesh,
        vertex_array: torch.Tensor,
        training_dataset: TensorDataset | None = None,
    ) -> int:
        return self.target_collocation_budget

    def get_domain_bounds(self) -> Tuple[float, float, float, float]:
        r = self.outer_radius
        return (-r, r, -r, r)

    def get_dirichlet_boundaries(self) -> list:
        return ["outer", "main_hole", "top_hole", "bottom_hole"]
