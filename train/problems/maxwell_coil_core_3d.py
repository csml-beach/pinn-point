"""
3D magnetostatic vector-potential problem inspired by NGSolve coil examples.

Geometry:
    - outer air box
    - cylindrical ferromagnetic core
    - cylindrical shell coil region

PDE (evaluation/training form):
    curl(nu * curl(A)) + alpha * nu * A = J

with piecewise nu = 1/mu and impressed current density J in the coil region.
"""

from __future__ import annotations

from typing import Any, List, Tuple

import math
import numpy as np
import torch
from netgen.occ import Box, Cylinder, Glue, OCCGeometry, Pnt, Z
from ngsolve import *

from .base import PDEProblem


class MaxwellCoilCore3DProblem(PDEProblem):
    """3D magnetostatic coil-core benchmark (vector potential A)."""

    name = "maxwell_coil_core_3d"
    description = (
        "3D magnetostatic vector-potential problem with coil, ferromagnetic core, and air box"
    )

    input_dim = 3
    output_dim = 3
    output_names = ("Ax", "Ay", "Az")
    has_time_input = False
    has_spatial_3d = True

    def __init__(
        self,
        box_half_xy: float = 0.04,
        box_z_min: float = -0.03,
        box_z_max: float = 0.06,
        core_radius: float = 0.010,
        core_z_min: float = -0.02,
        core_z_max: float = 0.05,
        coil_inner_radius: float = 0.012,
        coil_outer_radius: float = 0.020,
        coil_z_min: float = -0.005,
        coil_z_max: float = 0.035,
        core_relative_permeability: float = 500.0,
        current_density: float = 1.0,
        gauge_alpha: float = 1e-6,
        fe_order: int = 2,
        bc_loss_weight: float = 10.0,
    ):
        self.box_half_xy = float(box_half_xy)
        self.box_z_min = float(box_z_min)
        self.box_z_max = float(box_z_max)
        self.core_radius = float(core_radius)
        self.core_z_min = float(core_z_min)
        self.core_z_max = float(core_z_max)
        self.coil_inner_radius = float(coil_inner_radius)
        self.coil_outer_radius = float(coil_outer_radius)
        self.coil_z_min = float(coil_z_min)
        self.coil_z_max = float(coil_z_max)
        self.core_relative_permeability = float(core_relative_permeability)
        self.current_density = float(current_density)
        self.gauge_alpha = float(gauge_alpha)
        self.fe_order = int(fe_order)
        self.bc_loss_weight = float(bc_loss_weight)

        if self.coil_inner_radius <= 0.0 or self.coil_outer_radius <= self.coil_inner_radius:
            raise ValueError("Expected 0 < coil_inner_radius < coil_outer_radius")
        if self.core_z_max <= self.core_z_min:
            raise ValueError("Expected core_z_max > core_z_min")
        if self.coil_z_max <= self.coil_z_min:
            raise ValueError("Expected coil_z_max > coil_z_min")

        self.mu0 = 4.0 * math.pi * 1.0e-7
        self.mu_air = self.mu0
        self.mu_core = self.mu0 * max(self.core_relative_permeability, 1.0)
        self.nu_air = 1.0 / self.mu_air
        self.nu_core = 1.0 / self.mu_core

    def _build_occ_solids(self):
        box = Box(
            Pnt(-self.box_half_xy, -self.box_half_xy, self.box_z_min),
            Pnt(self.box_half_xy, self.box_half_xy, self.box_z_max),
        )
        box.faces.name = "outer"

        core = Cylinder(
            Pnt(0.0, 0.0, self.core_z_min),
            Z,
            self.core_radius,
            self.core_z_max - self.core_z_min,
        )
        core.mat("core")

        coil_outer = Cylinder(
            Pnt(0.0, 0.0, self.coil_z_min),
            Z,
            self.coil_outer_radius,
            self.coil_z_max - self.coil_z_min,
        )
        coil_inner = Cylinder(
            Pnt(0.0, 0.0, self.coil_z_min),
            Z,
            self.coil_inner_radius,
            self.coil_z_max - self.coil_z_min,
        )
        coil = coil_outer - coil_inner
        coil.mat("coil")

        air = box - core - coil
        air.mat("air")
        air.faces.name = "outer"
        return air, core, coil

    def create_mesh(self, maxh=None):
        if maxh is None:
            maxh = 0.02
        air, core, coil = self._build_occ_solids()
        geo = OCCGeometry(Glue([air, core, coil]))
        ngmesh = geo.GenerateMesh(maxh=float(maxh))
        mesh = Mesh(ngmesh)
        mesh.Curve(3)
        return mesh

    def get_domain_bounds(self) -> Tuple[float, float, float, float]:
        return (-self.box_half_xy, self.box_half_xy, -self.box_half_xy, self.box_half_xy)

    def get_spatial_bounds_nd(self) -> List[Tuple[float, float]]:
        return [
            (-self.box_half_xy, self.box_half_xy),
            (-self.box_half_xy, self.box_half_xy),
            (self.box_z_min, self.box_z_max),
        ]

    def get_dirichlet_boundaries(self) -> list:
        return ["outer"]

    def _coil_mask_torch(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        r2 = x * x + y * y
        return (
            (r2 >= self.coil_inner_radius ** 2)
            & (r2 <= self.coil_outer_radius ** 2)
            & (z >= self.coil_z_min)
            & (z <= self.coil_z_max)
        )

    def _core_mask_torch(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        r2 = x * x + y * y
        return (
            (r2 <= self.core_radius ** 2)
            & (z >= self.core_z_min)
            & (z <= self.core_z_max)
        )

    def _nu_torch(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        core_mask = self._core_mask_torch(x, y, z)
        nu_core = torch.full_like(x, self.nu_core)
        nu_air = torch.full_like(x, self.nu_air)
        return torch.where(core_mask, nu_core, nu_air)

    def _current_density_torch(
        self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
        eps = torch.tensor(1.0e-12, dtype=x.dtype, device=x.device)
        r = torch.sqrt(torch.clamp(x * x + y * y, min=eps))
        jx = -y / r
        jy = x / r
        jz = torch.zeros_like(x)
        mask = self._coil_mask_torch(x, y, z).to(x.dtype)
        return self.current_density * torch.stack([jx * mask, jy * mask, jz], dim=1)

    def source_term(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if z is None:
            return torch.zeros((len(x), 3), dtype=torch.float32, device=x.device)
        x = torch.as_tensor(x, dtype=torch.float32, device=x.device).reshape(-1)
        y = torch.as_tensor(y, dtype=torch.float32, device=x.device).reshape(-1)
        z = torch.as_tensor(z, dtype=torch.float32, device=x.device).reshape(-1)
        return self._current_density_torch(x, y, z)

    def solve_fem(self, mesh) -> Tuple[Any, Any]:
        fes = HCurl(mesh, order=self.fe_order, dirichlet="outer", nograds=True)
        u, v = fes.TnT()

        materials = mesh.GetMaterials()
        nu_values = [
            self.nu_core if mat == "core" else self.nu_air for mat in materials
        ]
        nu_cf = CoefficientFunction(nu_values)

        r_cf = sqrt(x * x + y * y + 1.0e-12)
        j_cf = self.current_density * CoefficientFunction((-y / r_cf, x / r_cf, 0.0))

        a = BilinearForm(nu_cf * curl(u) * curl(v) * dx + self.gauge_alpha * nu_cf * u * v * dx)
        f = LinearForm(j_cf * v * dx("coil"))

        a.Assemble()
        f.Assemble()

        gfu = GridFunction(fes)
        gfu.vec.data = a.mat.Inverse(fes.FreeDofs()) * f.vec
        return gfu, fes

    def pde_residual(
        self,
        model: Any,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from config import DEVICE

        x = torch.as_tensor(x, dtype=torch.float32, device=DEVICE).reshape(-1)
        y = torch.as_tensor(y, dtype=torch.float32, device=DEVICE).reshape(-1)
        if z is None:
            z = torch.zeros_like(x)
        else:
            z = torch.as_tensor(z, dtype=torch.float32, device=DEVICE).reshape(-1)

        x = x.clone().requires_grad_(True)
        y = y.clone().requires_grad_(True)
        z = z.clone().requires_grad_(True)

        A = model.forward(x, y, z)
        Ax, Ay, Az = A[:, 0], A[:, 1], A[:, 2]
        ones = torch.ones_like(Ax)

        def _grad(field_comp, var):
            grad = torch.autograd.grad(
                field_comp,
                var,
                grad_outputs=ones,
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
            )[0]
            return grad if grad is not None else torch.zeros_like(var)

        dAx_dx = _grad(Ax, x)
        dAx_dy = _grad(Ax, y)
        dAx_dz = _grad(Ax, z)
        dAy_dx = _grad(Ay, x)
        dAy_dy = _grad(Ay, y)
        dAy_dz = _grad(Ay, z)
        dAz_dx = _grad(Az, x)
        dAz_dy = _grad(Az, y)
        dAz_dz = _grad(Az, z)

        # curl(A)
        Cx = dAz_dy - dAy_dz
        Cy = dAx_dz - dAz_dx
        Cz = dAy_dx - dAx_dy

        # curl(curl(A))
        dCx_dx = _grad(Cx, x)
        dCx_dy = _grad(Cx, y)
        dCx_dz = _grad(Cx, z)
        dCy_dx = _grad(Cy, x)
        dCy_dy = _grad(Cy, y)
        dCy_dz = _grad(Cy, z)
        dCz_dx = _grad(Cz, x)
        dCz_dy = _grad(Cz, y)
        dCz_dz = _grad(Cz, z)

        CCx = dCz_dy - dCy_dz
        CCy = dCx_dz - dCz_dx
        CCz = dCy_dx - dCx_dy

        nu = self._nu_torch(x, y, z)
        J = self._current_density_torch(x, y, z)

        rx = nu * CCx + self.gauge_alpha * nu * Ax - J[:, 0]
        ry = nu * CCy + self.gauge_alpha * nu * Ay - J[:, 1]
        rz = nu * CCz + self.gauge_alpha * nu * Az - J[:, 2]
        return torch.sqrt(rx * rx + ry * ry + rz * rz)

    def boundary_loss(self, model: Any, num_boundary_points: int = 240) -> torch.Tensor:
        from config import DEVICE

        n_face = max(int(num_boundary_points) // 6, 40)
        h = self.box_half_xy
        z0 = self.box_z_min
        z1 = self.box_z_max

        ys = torch.rand(n_face, device=DEVICE) * (2.0 * h) - h
        zs = torch.rand(n_face, device=DEVICE) * (z1 - z0) + z0
        xs = torch.rand(n_face, device=DEVICE) * (2.0 * h) - h

        pts = [
            (torch.full((n_face,), -h, device=DEVICE), ys, zs),
            (torch.full((n_face,), +h, device=DEVICE), ys, zs),
            (xs, torch.full((n_face,), -h, device=DEVICE), zs),
            (xs, torch.full((n_face,), +h, device=DEVICE), zs),
            (xs, ys, torch.full((n_face,), z0, device=DEVICE)),
            (xs, ys, torch.full((n_face,), z1, device=DEVICE)),
        ]

        losses = []
        for px, py, pz in pts:
            pred = model.forward(px, py, pz)
            losses.append(torch.mean(torch.square(pred)))
        return self.bc_loss_weight * sum(losses) / len(losses)

    def export_fem_solution(self, mesh, gfu) -> torch.Tensor:
        coords = np.array([v.point for v in mesh.vertices], dtype=float)
        values = []
        for pt in coords:
            try:
                val = gfu(mesh(float(pt[0]), float(pt[1]), float(pt[2])))
                values.append([float(val[0]), float(val[1]), float(val[2])])
            except Exception:
                values.append([0.0, 0.0, 0.0])
        return torch.tensor(values, dtype=torch.float32)

    def create_reference_solution(self, mesh_size_factor: float = 0.012):
        print(f"Creating Maxwell 3D reference FEM solution (maxh={mesh_size_factor})...")
        ref_mesh = self.create_mesh(maxh=mesh_size_factor)
        gfu, _fes = self.solve_fem(ref_mesh)
        n_pts = len(list(ref_mesh.vertices))
        print(f"Maxwell reference solution: {n_pts:,} vertices")
        return ref_mesh, gfu

    def evaluate_model_against_reference(
        self,
        model,
        reference_mesh,
        reference_solution,
        *,
        export_images: bool = False,
        iteration=None,
    ) -> bool:
        import math
        from config import DEVICE
        from geometry import export_vertex_coordinates
        from mesh_refinement import _append_history_value

        ref_coords = export_vertex_coordinates(reference_mesh).detach().cpu().numpy()
        n = len(ref_coords)

        ref_x = torch.tensor(ref_coords[:, 0], dtype=torch.float32, device=DEVICE)
        ref_y = torch.tensor(ref_coords[:, 1], dtype=torch.float32, device=DEVICE)
        ref_z = torch.tensor(ref_coords[:, 2], dtype=torch.float32, device=DEVICE)

        with torch.no_grad():
            pred = model.forward(ref_x, ref_y, ref_z).detach().cpu().numpy()

        ref_arr = np.zeros((n, 3), dtype=float)
        for i, pt in enumerate(ref_coords):
            try:
                val = reference_solution(
                    reference_mesh(float(pt[0]), float(pt[1]), float(pt[2]))
                )
                ref_arr[i] = [float(val[0]), float(val[1]), float(val[2])]
            except Exception:
                pass

        diff = pred - ref_arr
        total_error = float(np.mean(np.sum(diff * diff, axis=1)))
        ref_norm_sq = float(np.mean(np.sum(ref_arr * ref_arr, axis=1)))
        relative_l2 = (
            float(math.sqrt(total_error / ref_norm_sq)) if ref_norm_sq > 0 else float("nan")
        )

        h = self.box_half_xy
        z0, z1 = self.box_z_min, self.box_z_max
        bdry_mask = (
            (np.abs(ref_coords[:, 0] + h) < 1e-5)
            | (np.abs(ref_coords[:, 0] - h) < 1e-5)
            | (np.abs(ref_coords[:, 1] + h) < 1e-5)
            | (np.abs(ref_coords[:, 1] - h) < 1e-5)
            | (np.abs(ref_coords[:, 2] - z0) < 1e-5)
            | (np.abs(ref_coords[:, 2] - z1) < 1e-5)
        )
        boundary_error = (
            float(np.mean(np.sum(diff[bdry_mask] ** 2, axis=1)))
            if np.any(bdry_mask)
            else float("nan")
        )

        total_error_rms = float(math.sqrt(total_error))
        ref_rms = float(math.sqrt(ref_norm_sq))
        relative_error_rms = total_error_rms / ref_rms if ref_rms > 0 else float("nan")

        model.total_error_history.append(total_error)
        model.boundary_error_history.append(boundary_error)
        _append_history_value(model, "relative_l2_error_history", relative_l2)
        _append_history_value(model, "total_error_rms_history", total_error_rms)
        _append_history_value(model, "relative_error_rms_history", relative_error_rms)

        rx = ref_x.detach().clone().requires_grad_(True)
        ry = ref_y.detach().clone().requires_grad_(True)
        rz = ref_z.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            residual = self.pde_residual(model, rx, ry, rz)
        residual_sq = torch.square(residual).detach().cpu().numpy().reshape(-1)
        residual_sq = residual_sq[np.isfinite(residual_sq)]
        residual_rms = (
            float(math.sqrt(float(np.mean(residual_sq)))) if residual_sq.size else float("nan")
        )

        _append_history_value(model, "fixed_rms_residual_history", residual_rms)
        _append_history_value(model, "relative_fixed_rms_residual_history", float("nan"))
        _append_history_value(model, "fixed_total_residual_history", float(np.mean(residual_sq)) if residual_sq.size else float("nan"))
        _append_history_value(model, "relative_fixed_l2_residual_history", float("nan"))
        _append_history_value(model, "fixed_boundary_residual_history", float("nan"))

        print(
            f"Error (vs reference, vertex-L2): {total_error:.6e}, "
            f"Relative L2: {relative_l2:.6e}, "
            f"Boundary Error: {boundary_error:.6e}, "
            f"Residual RMS: {residual_rms:.6e}"
        )
        return True

    def build_geometry_smoke_metadata(self, mesh) -> dict:
        return {
            "num_vertices": len(list(mesh.vertices)),
            "mesh_dimension": 3,
            "materials": list(mesh.GetMaterials()),
            "boundaries": list(mesh.GetBoundaries()),
            "problem": self.name,
        }

    def build_fem_smoke_metadata(self, mesh, dt=None, t_end=None) -> dict:
        from ngsolve import VOL

        print("Running Maxwell FEM solve for smoke test...")
        gfu, fes = self.solve_fem(mesh)
        n_dofs = fes.ndof
        n_verts = len(list(mesh.vertices))
        n_elems = sum(1 for _ in mesh.Elements(VOL))

        sample_pts = [(0.0, 0.0, 0.0), (0.015, 0.0, 0.015)]
        sample_norms = []
        for pt in sample_pts:
            try:
                val = gfu(mesh(pt[0], pt[1], pt[2]))
                sample_norms.append(float(sum(v * v for v in val) ** 0.5))
            except Exception:
                sample_norms.append(float("nan"))

        print(f"FEM: {n_dofs} DOFs, {n_verts} vertices, {n_elems} elements")
        return {
            "num_vertices": n_verts,
            "num_elements": n_elems,
            "num_dofs": n_dofs,
            "sample_vector_norms": sample_norms,
            "materials": list(mesh.GetMaterials()),
            "boundaries": list(mesh.GetBoundaries()),
            "problem": self.name,
        }
