"""
Fixed channel-obstacle geometry and FEM-only transient Navier-Stokes prototype.

This module currently provides:
    - Netgen/OCC geometry and mesh generation
    - transient FEM reference solve with Taylor-Hood elements

PINN residual and training losses are intentionally left for a later step.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
from netgen.occ import OCCGeometry, WorkPlane, X, Y
from ngsolve import *

from .base import PDEProblem


class NavierStokesChannelObstacleProblem(PDEProblem):
    """Channel with a contraction and staggered circular obstacles."""

    name = "navier_stokes_channel_obstacle"
    description = (
        "Geometry-only prototype for a time-dependent Navier-Stokes benchmark on "
        "a contracted channel with circular obstacles"
    )

    def __init__(
        self,
        length: float = 8.0,
        height: float = 3.0,
        throat_center_x: float = 3.2,
        throat_length: float = 1.2,
        throat_half_height: float = 0.45,
        obstacle_radius: float = 0.22,
        viscosity: float = 0.02,
        t_end: float = 0.5,
        dt: float = 0.002,
        fe_order: int = 2,
        inlet_peak_velocity: float = 1.5,
    ):
        self.length = float(length)
        self.height = float(height)
        self.throat_center_x = float(throat_center_x)
        self.throat_length = float(throat_length)
        self.throat_half_height = float(throat_half_height)
        self.obstacle_radius = float(obstacle_radius)
        self.viscosity = float(viscosity)
        self.t_end = float(t_end)
        self.dt = float(dt)
        self.fe_order = int(fe_order)
        self.inlet_peak_velocity = float(inlet_peak_velocity)

    def _build_occ_face(self):
        length = self.length
        height = self.height
        throat_x0 = self.throat_center_x - 0.5 * self.throat_length
        throat_x1 = self.throat_center_x + 0.5 * self.throat_length
        mid_y = 0.5 * height
        top_cut_height = max(0.0, height - (mid_y + self.throat_half_height))
        bottom_cut_height = max(0.0, mid_y - self.throat_half_height)

        channel = WorkPlane().Rectangle(length, height).Face()

        # Contraction by subtracting top/bottom blocks in the throat region.
        top_cut = (
            WorkPlane()
            .MoveTo(throat_x0, mid_y + self.throat_half_height)
            .Rectangle(self.throat_length, top_cut_height)
            .Face()
        )
        bottom_cut = (
            WorkPlane()
            .MoveTo(throat_x0, 0.0)
            .Rectangle(self.throat_length, bottom_cut_height)
            .Face()
        )

        # Three staggered circular obstacles: upstream, throat-adjacent, downstream.
        circle1 = (
            WorkPlane()
            .Circle(self.obstacle_radius)
            .Face()
            .Move((2.0, 0.95, 0.0))
        )
        circle2 = (
            WorkPlane()
            .Circle(self.obstacle_radius)
            .Face()
            .Move((4.4, 2.05, 0.0))
        )
        circle3 = (
            WorkPlane()
            .Circle(self.obstacle_radius)
            .Face()
            .Move((5.7, 1.15, 0.0))
        )

        geo = channel - top_cut - bottom_cut - circle1 - circle2 - circle3

        # Start by marking all boundaries as obstacle-type, then override the
        # outer channel boundaries with the Navier-Stokes names we intend to use.
        geo.edges.name = "obstacle"
        geo.edges.Min(X).name = "inlet"
        geo.edges.Max(X).name = "outlet"
        geo.edges.Min(Y).name = "wall"
        geo.edges.Max(Y).name = "wall"
        return geo

    def create_mesh(self, maxh=None):
        if maxh is None:
            maxh = 0.20
        geo = self._build_occ_face()
        occ = OCCGeometry(geo, dim=2)
        return Mesh(occ.GenerateMesh(maxh=maxh)).Curve(3)

    def get_domain_bounds(self) -> Tuple[float, float, float, float]:
        return (0.0, self.length, 0.0, self.height)

    def get_dirichlet_boundaries(self) -> list:
        return ["inlet", "wall", "obstacle"]

    def _inflow_profile(self) -> Any:
        h = self.height
        ymax = max(h, 1e-8)
        return CoefficientFunction(
            (self.inlet_peak_velocity * 4.0 * y * (h - y) / (ymax * ymax), 0.0)
        )

    def solve_fem_time_series(
        self,
        mesh,
        dt: float | None = None,
        t_end: float | None = None,
        snapshot_times: List[float] | None = None,
    ) -> Dict[str, Any]:
        """Solve transient incompressible Navier-Stokes with IMEX Euler.

        Follows the documented NGSolve Taylor-Hood tutorial pattern:
        implicit Stokes / explicit convection.
        """
        dt = float(self.dt if dt is None else dt)
        t_end = float(self.t_end if t_end is None else t_end)
        snapshot_times = snapshot_times or [0.0, 0.25 * t_end, 0.5 * t_end, t_end]
        snapshot_targets = sorted({max(0.0, min(t_end, float(t))) for t in snapshot_times})

        k = self.fe_order
        V = VectorH1(mesh, order=k, dirichlet="wall|obstacle|inlet")
        Q = H1(mesh, order=max(1, k - 1))
        Xspace = V * Q

        gfu = GridFunction(Xspace)
        velocity, pressure = gfu.components
        velocity.Set(self._inflow_profile(), definedon=mesh.Boundaries("inlet"))

        (u, p), (v, q) = Xspace.TnT()

        a = BilinearForm(Xspace)
        stokes = (
            self.viscosity * InnerProduct(grad(u), grad(v))
            - div(u) * q
            - div(v) * p
        ) * dx
        a += stokes
        a.Assemble()

        f = LinearForm(Xspace)
        f.Assemble()
        res = f.vec.CreateVector()
        res.data = f.vec - a.mat * gfu.vec
        inv_stokes = a.mat.Inverse(Xspace.FreeDofs())
        gfu.vec.data += inv_stokes * res

        mstar = BilinearForm(Xspace)
        mstar += InnerProduct(u, v) * dx + dt * stokes
        mstar.Assemble()
        inv = mstar.mat.Inverse(Xspace.FreeDofs())

        conv = BilinearForm(Xspace, nonassemble=True)
        conv += (Grad(u) * u) * v * dx

        snapshots: List[Dict[str, Any]] = []
        vertex_coords = [(float(vtx.point[0]), float(vtx.point[1])) for vtx in mesh.vertices]

        def capture_snapshot(time_value: float):
            vel_vals = []
            pressure_vals = []
            for x_coord, y_coord in vertex_coords:
                mip = mesh(x_coord, y_coord)
                vel = velocity(mip)
                pres = pressure(mip)
                vel_mag = float((vel[0] ** 2 + vel[1] ** 2) ** 0.5)
                vel_vals.append(vel_mag)
                pressure_vals.append(float(pres))
            snapshots.append(
                {
                    "time": float(time_value),
                    "velocity_magnitude": vel_vals,
                    "pressure": pressure_vals,
                }
            )

        capture_snapshot(0.0)
        t = 0.0
        next_target_index = 0
        while next_target_index < len(snapshot_targets) and snapshot_targets[next_target_index] <= 0.0 + 1e-12:
            next_target_index += 1

        while t < t_end - 0.5 * dt:
            res.data = (a.mat + conv.mat) * gfu.vec
            gfu.vec.data -= dt * inv * res
            t += dt

            while (
                next_target_index < len(snapshot_targets)
                and t >= snapshot_targets[next_target_index] - 0.5 * dt
            ):
                capture_snapshot(snapshot_targets[next_target_index])
                next_target_index += 1

        if snapshots[-1]["time"] < t_end - 1e-12:
            capture_snapshot(t_end)

        return {
            "space": Xspace,
            "state": gfu,
            "snapshots": snapshots,
            "vertex_coords": vertex_coords,
        }

    def solve_fem(self, mesh):
        result = self.solve_fem_time_series(mesh)
        return result["state"], result["space"]

    # PINN methods are implemented in a later step once the FEM smoke is stable.
    def pde_residual(
        self, model: Any, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError(
            "PDE residual is not implemented yet for navier_stokes_channel_obstacle"
        )

    def boundary_loss(self, model: Any, num_boundary_points: int = 100) -> torch.Tensor:
        raise NotImplementedError(
            "Boundary loss is not implemented yet for navier_stokes_channel_obstacle"
        )

    def source_term(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "Source term is not implemented yet for navier_stokes_channel_obstacle"
        )
