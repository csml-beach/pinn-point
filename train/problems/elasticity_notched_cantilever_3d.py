"""
3D linear elasticity problem on a notched cantilever beam.

Geometry: a rectangular box (3 × 0.6 × 1) with three cylindrical holes.
Boundary conditions:
    - fix face (x=0): clamped, u = 0
    - force face (x=3): traction g = (traction_magnitude, 0, 0) in x-direction
    - all other faces: traction-free (Neumann, g = 0)

PDE: div(σ(ε(u))) = 0  in Ω
     σ = 2μ·ε + λ·tr(ε)·I
     ε = ½(∇u + (∇u)ᵀ)

Parameters (Lamé):
    E = 210, ν = 0.2
    μ = E / (2(1+ν)),  λ = E·ν / ((1+ν)(1-2ν))
"""

from __future__ import annotations

from typing import Any, List, Sequence, Tuple

import numpy as np
import torch
from netgen.occ import Box, Cylinder, Pnt, Vec, X, Y, Z, OCCGeometry
from ngsolve import *

from .base import PDEProblem


class ElasticityNotchedCantilever3DProblem(PDEProblem):
    """3D linear elasticity on a notched cantilever beam with clamped + traction BCs."""

    name = "elasticity_notched_cantilever_3d"
    description = (
        "3D linear elasticity on a notched cantilever beam "
        "(box with cylindrical cutouts, clamped at x=0, traction at x=3)"
    )

    input_dim = 3
    output_dim = 3
    output_names = ("ux", "uy", "uz")
    has_time_input = False
    has_spatial_3d = True

    def __init__(
        self,
        length: float = 3.0,
        width: float = 0.6,
        height: float = 1.0,
        cylinder_radius: float = 0.25,
        youngs_modulus: float = 210.0,
        poisson_ratio: float = 0.2,
        traction_magnitude: float = 1e-3,
        fe_order: int = 2,
        bc_loss_weight: float = 10.0,
        traction_loss_weight: float = 1.0,
        hole_centers_x: Sequence[float] | None = None,
        localized_traction: bool = False,
        traction_patch_center_y: float | None = None,
        traction_patch_center_z: float | None = None,
        traction_patch_sigma: float = 0.15,
    ):
        self.length = float(length)
        self.width = float(width)
        self.height = float(height)
        self.cylinder_radius = float(cylinder_radius)
        self.E = float(youngs_modulus)
        self.nu = float(poisson_ratio)
        self.mu = self.E / (2.0 * (1.0 + self.nu))
        self.lam = self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))
        self.traction_magnitude = float(traction_magnitude)
        self.fe_order = int(fe_order)
        self.bc_loss_weight = float(bc_loss_weight)
        self.traction_loss_weight = float(traction_loss_weight)
        self.hole_centers_x = [float(cx) for cx in (hole_centers_x or [0.5, 1.5, 2.5])]
        self.localized_traction = bool(localized_traction)
        self.traction_patch_center_y = (
            self.width / 2.0
            if traction_patch_center_y is None
            else float(traction_patch_center_y)
        )
        self.traction_patch_center_z = (
            self.height / 2.0
            if traction_patch_center_z is None
            else float(traction_patch_center_z)
        )
        self.traction_patch_sigma = float(traction_patch_sigma)

    # ------------------------------------------------------------------
    # Geometry and mesh
    # ------------------------------------------------------------------

    def _build_occ_solid(self):
        box = Box(Pnt(0, 0, 0), Pnt(self.length, self.width, self.height))
        box.faces.Min(X).name = "fix"
        box.faces.Max(X).name = "force"
        box.faces.name = "outer"  # name all faces first, then override specific ones
        box.faces.Min(X).name = "fix"
        box.faces.Max(X).name = "force"

        # Three cylindrical cutouts along Y axis at different X positions
        cyls = []
        for cx in self.hole_centers_x:
            cyl = Cylinder(
                Pnt(cx, 0, self.height / 2.0),
                Y,
                self.cylinder_radius,
                self.width,
            )
            cyls.append(cyl)

        geo = box
        for cyl in cyls:
            geo = geo - cyl

        # Re-name boundary faces after boolean ops
        geo.faces.Min(X).name = "fix"
        geo.faces.Max(X).name = "force"

        return geo

    def create_mesh(self, maxh=None):
        if maxh is None:
            maxh = 0.35

        geo = self._build_occ_solid()
        occ = OCCGeometry(geo)
        ngmesh = occ.GenerateMesh(maxh=float(maxh))
        mesh = Mesh(ngmesh)
        mesh.Curve(3)
        return mesh

    def get_domain_bounds(self) -> Tuple[float, float, float, float]:
        return (0.0, self.length, 0.0, self.width)

    def get_spatial_bounds_nd(self) -> List[Tuple[float, float]]:
        return [
            (0.0, self.length),
            (0.0, self.width),
            (0.0, self.height),
        ]

    def get_dirichlet_boundaries(self) -> list:
        return ["fix"]

    def _traction_profile_torch(
        self, y_coord: torch.Tensor, z_coord: torch.Tensor
    ) -> torch.Tensor:
        if not self.localized_traction:
            return torch.ones_like(y_coord)

        sigma = max(self.traction_patch_sigma, 1.0e-6)
        dy = (y_coord - self.traction_patch_center_y) / sigma
        dz = (z_coord - self.traction_patch_center_z) / sigma
        return torch.exp(-0.5 * (dy * dy + dz * dz))

    def _force_face_training_points(self, count: int):
        from config import DEVICE

        count = int(count)
        if not self.localized_traction:
            y_force = torch.rand(count, device=DEVICE) * self.width
            z_force = torch.rand(count, device=DEVICE) * self.height
        else:
            focused = count // 2
            uniform = count - focused
            sigma = max(self.traction_patch_sigma, 1.0e-6)
            y_patch = torch.normal(
                mean=self.traction_patch_center_y,
                std=sigma,
                size=(focused,),
                device=DEVICE,
            ).clamp(0.0, self.width)
            z_patch = torch.normal(
                mean=self.traction_patch_center_z,
                std=sigma,
                size=(focused,),
                device=DEVICE,
            ).clamp(0.0, self.height)
            y_uniform = torch.rand(uniform, device=DEVICE) * self.width
            z_uniform = torch.rand(uniform, device=DEVICE) * self.height
            y_force = torch.cat([y_patch, y_uniform], dim=0)
            z_force = torch.cat([z_patch, z_uniform], dim=0)
        x_force = torch.full((count,), self.length, device=DEVICE)
        return x_force, y_force, z_force

    # ------------------------------------------------------------------
    # FEM solve
    # ------------------------------------------------------------------

    def solve_fem(self, mesh) -> Tuple[Any, Any]:
        fes = VectorH1(mesh, order=self.fe_order, dirichlet="fix")
        u, v = fes.TnT()
        gfu = GridFunction(fes)

        def stress(strain):
            return 2 * self.mu * strain + self.lam * Trace(strain) * Id(3)

        a = BilinearForm(
            InnerProduct(stress(Sym(Grad(u))), Sym(Grad(v))) * dx
        )
        a.Assemble()

        # Traction force on "force" face.
        if self.localized_traction:
            sigma = max(self.traction_patch_sigma, 1.0e-6)
            profile = exp(
                -0.5
                * (
                    ((y - self.traction_patch_center_y) / sigma) ** 2
                    + ((z - self.traction_patch_center_z) / sigma) ** 2
                )
            )
            force = CoefficientFunction((self.traction_magnitude * profile, 0.0, 0.0))
        else:
            force = CoefficientFunction((self.traction_magnitude, 0.0, 0.0))
        f = LinearForm(force * v * ds("force"))
        f.Assemble()

        gfu.vec.data = a.mat.Inverse(fes.FreeDofs()) * f.vec
        return gfu, fes

    # ------------------------------------------------------------------
    # Source term (body forces = 0)
    # ------------------------------------------------------------------

    def source_term(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return torch.zeros(len(x), 3, dtype=torch.float32, device=x.device)

    # ------------------------------------------------------------------
    # PDE residual (autograd-based)
    # ------------------------------------------------------------------

    def pde_residual(
        self,
        model: Any,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute div(σ(ε(u))) residual at interior points.

        Returns a tensor of shape (N,) — the sum of squared momentum residuals.
        """
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

        # Forward: u is (N, 3)
        uvw = model.forward(x, y, z)
        ux = uvw[:, 0]
        uy = uvw[:, 1]
        uz = uvw[:, 2]

        ones = torch.ones_like(ux)

        def _grad(u_comp, var):
            if not u_comp.requires_grad:
                return torch.zeros_like(var)
            grad = torch.autograd.grad(
                u_comp, var,
                grad_outputs=ones,
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
            )[0]
            return grad if grad is not None else torch.zeros_like(var)

        # First-order partials (strain components)
        dux_dx = _grad(ux, x)
        dux_dy = _grad(ux, y)
        dux_dz = _grad(ux, z)
        duy_dx = _grad(uy, x)
        duy_dy = _grad(uy, y)
        duy_dz = _grad(uy, z)
        duz_dx = _grad(uz, x)
        duz_dy = _grad(uz, y)
        duz_dz = _grad(uz, z)

        # Symmetric strain tensor components
        exx = dux_dx
        eyy = duy_dy
        ezz = duz_dz
        exy = 0.5 * (dux_dy + duy_dx)
        exz = 0.5 * (dux_dz + duz_dx)
        eyz = 0.5 * (duy_dz + duz_dy)

        tr_e = exx + eyy + ezz

        # Stress components: σ = 2μ·ε + λ·tr(ε)·I
        mu = self.mu
        lam = self.lam
        sxx = 2 * mu * exx + lam * tr_e
        syy = 2 * mu * eyy + lam * tr_e
        szz = 2 * mu * ezz + lam * tr_e
        sxy = 2 * mu * exy
        sxz = 2 * mu * exz
        syz = 2 * mu * eyz

        # Divergence of stress = body forces (= 0)
        def _grad2(s_comp, var):
            if not s_comp.requires_grad:
                return torch.zeros_like(var)
            grad = torch.autograd.grad(
                s_comp, var,
                grad_outputs=torch.ones_like(s_comp),
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
            )[0]
            return grad if grad is not None else torch.zeros_like(var)

        rx = _grad2(sxx, x) + _grad2(sxy, y) + _grad2(sxz, z)
        ry = _grad2(sxy, x) + _grad2(syy, y) + _grad2(syz, z)
        rz = _grad2(sxz, x) + _grad2(syz, y) + _grad2(szz, z)

        residual = rx ** 2 + ry ** 2 + rz ** 2
        return residual.sqrt()  # shape (N,) — sqrt so MSE loss is on squared residual

    def _stress_components(
        self,
        model: Any,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (sxx, syy, szz, sxy, sxz, syz) for boundary traction checks."""
        uvw = model.forward(x, y, z)
        ux = uvw[:, 0]
        uy = uvw[:, 1]
        uz = uvw[:, 2]

        def _grad(u_comp, var):
            if not u_comp.requires_grad:
                return torch.zeros_like(var)
            grad = torch.autograd.grad(
                u_comp,
                var,
                grad_outputs=torch.ones_like(u_comp),
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
            )[0]
            return grad if grad is not None else torch.zeros_like(var)

        dux_dx = _grad(ux, x)
        dux_dy = _grad(ux, y)
        dux_dz = _grad(ux, z)
        duy_dx = _grad(uy, x)
        duy_dy = _grad(uy, y)
        duy_dz = _grad(uy, z)
        duz_dx = _grad(uz, x)
        duz_dy = _grad(uz, y)
        duz_dz = _grad(uz, z)

        exx = dux_dx
        eyy = duy_dy
        ezz = duz_dz
        exy = 0.5 * (dux_dy + duy_dx)
        exz = 0.5 * (dux_dz + duz_dx)
        eyz = 0.5 * (duy_dz + duz_dy)
        tr_e = exx + eyy + ezz

        sxx = 2 * self.mu * exx + self.lam * tr_e
        syy = 2 * self.mu * eyy + self.lam * tr_e
        szz = 2 * self.mu * ezz + self.lam * tr_e
        sxy = 2 * self.mu * exy
        sxz = 2 * self.mu * exz
        syz = 2 * self.mu * eyz
        return sxx, syy, szz, sxy, sxz, syz

    # ------------------------------------------------------------------
    # Boundary loss
    # ------------------------------------------------------------------

    def boundary_loss(
        self, model: Any, num_boundary_points: int = 200
    ) -> torch.Tensor:
        """Enforce clamped BC (u=0) on fix face and traction BC on force face."""
        from config import DEVICE

        # Dirichlet: fix face (x=0), sample uniform over y∈[0,W], z∈[0,H]
        n_fix = max(num_boundary_points, 100)
        y_fix = torch.rand(n_fix, device=DEVICE) * self.width
        z_fix = torch.rand(n_fix, device=DEVICE) * self.height
        x_fix = torch.zeros(n_fix, device=DEVICE)
        u_fix = model.forward(x_fix, y_fix, z_fix)  # (N, 3)
        loss_fix = torch.mean(torch.square(u_fix))

        # Neumann: force face (x=L), traction σn = (traction_magnitude, 0, 0).
        n_force = max(num_boundary_points, 100)
        x_force, y_force, z_force = self._force_face_training_points(n_force)
        x_force.requires_grad_(True)
        y_force.requires_grad_(True)
        z_force.requires_grad_(True)

        sxx, _, _, sxy, sxz, _ = self._stress_components(
            model, x_force, y_force, z_force
        )
        target_tx = self.traction_magnitude * self._traction_profile_torch(
            y_force, z_force
        )
        loss_force = torch.mean(
            torch.square(sxx - target_tx)
            + torch.square(sxy)
            + torch.square(sxz)
        )

        return (
            loss_fix * self.bc_loss_weight
            + loss_force * self.traction_loss_weight
        )

    # ------------------------------------------------------------------
    # Training dataset (from FEM vertices)
    # ------------------------------------------------------------------

    def export_fem_solution(self, mesh, gfu) -> torch.Tensor:
        """Evaluate VectorH1 FEM solution at mesh vertices → (N, 3) tensor."""
        coords = np.array([v.point for v in mesh.vertices], dtype=float)
        values = []
        for pt in coords:
            try:
                val = gfu(mesh(float(pt[0]), float(pt[1]), float(pt[2])))
                values.append([float(val[0]), float(val[1]), float(val[2])])
            except Exception:
                values.append([0.0, 0.0, 0.0])
        return torch.tensor(values, dtype=torch.float32)  # (N, 3)

    # ------------------------------------------------------------------
    # Reference solution (override for 3D fine mesh)
    # ------------------------------------------------------------------

    def create_reference_solution(self, mesh_size_factor: float = 0.1):
        print(f"Creating 3D reference FEM solution (maxh={mesh_size_factor})...")
        ref_mesh = self.create_mesh(maxh=mesh_size_factor)
        gfu, fes = self.solve_fem(ref_mesh)
        n_pts = len(list(ref_mesh.vertices))
        print(f"3D reference solution: {n_pts:,} vertices")
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
        """Compute L2 error and residual for 3D VectorH1 problem."""
        import math
        from geometry import export_vertex_coordinates
        from config import DEVICE
        from ngsolve import Integrate, VOL

        ref_coords = export_vertex_coordinates(reference_mesh).detach().cpu().numpy()
        n = len(ref_coords)
        ref_x = torch.tensor(ref_coords[:, 0], dtype=torch.float32, device=DEVICE)
        ref_y = torch.tensor(ref_coords[:, 1], dtype=torch.float32, device=DEVICE)
        ref_z = torch.tensor(ref_coords[:, 2], dtype=torch.float32, device=DEVICE)

        with torch.no_grad():
            u_pred = model.forward(ref_x, ref_y, ref_z).detach().cpu().numpy()  # (N, 3)

        # Reference FEM solution at vertices: interleaved [ux0,uy0,uz0, ux1,...]
        try:
            ref_vec = reference_solution.vec.FV().NumPy()
            n_dof = len(ref_vec)
            # VectorH1 with order k stores n_spatial * ndof_scalar DOFs
            # At the coarse mesh resolution, DOFs match vertex count * 3
            ref_arr = ref_vec.reshape(-1, 3) if n_dof == n * 3 else None
        except Exception:
            ref_arr = None

        if ref_arr is None:
            # Fall back: evaluate gfu at each vertex
            ref_arr = np.zeros((n, 3), dtype=float)
            for i, pt in enumerate(ref_coords):
                try:
                    val = reference_solution(reference_mesh(float(pt[0]), float(pt[1]), float(pt[2])))
                    ref_arr[i] = [float(val[0]), float(val[1]), float(val[2])]
                except Exception:
                    pass

        # L2 error at vertices
        diff = u_pred - ref_arr
        total_error = float(np.mean(np.sum(diff**2, axis=1)))
        ref_norm_sq = float(np.mean(np.sum(ref_arr**2, axis=1)))
        relative_l2 = float(math.sqrt(total_error / ref_norm_sq)) if ref_norm_sq > 0 else float("nan")

        # Boundary error (vertices near x=0 or x=L)
        bdry_mask = (ref_coords[:, 0] < 0.01) | (ref_coords[:, 0] > self.length - 0.01)
        if np.any(bdry_mask):
            boundary_error = float(np.mean(np.sum(diff[bdry_mask]**2, axis=1)))
        else:
            boundary_error = float("nan")

        # RMS metrics
        total_error_rms = float(math.sqrt(total_error))
        ref_rms = float(math.sqrt(ref_norm_sq))
        relative_error_rms = total_error_rms / ref_rms if ref_rms > 0 else float("nan")

        # Store on model
        from mesh_refinement import _append_history_value
        model.total_error_history.append(total_error)
        model.boundary_error_history.append(boundary_error)
        _append_history_value(model, "relative_l2_error_history", relative_l2)
        _append_history_value(model, "total_error_rms_history", total_error_rms)
        _append_history_value(model, "relative_error_rms_history", relative_error_rms)

        # Fixed residual metrics at the same reference vertices. The volume-scaled
        # value is an evaluation diagnostic, not a quadrature-exact FEM integral.
        ref_x_req = ref_x.detach().clone().requires_grad_(True)
        ref_y_req = ref_y.detach().clone().requires_grad_(True)
        ref_z_req = ref_z.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            residual = self.pde_residual(model, ref_x_req, ref_y_req, ref_z_req)
        residual_sq = torch.square(residual).detach().cpu().numpy().reshape(-1)
        residual_sq = residual_sq[np.isfinite(residual_sq)]
        if residual_sq.size:
            residual_rms = float(math.sqrt(float(np.mean(residual_sq))))
            try:
                volume = float(Integrate(1.0, reference_mesh, VOL))
            except Exception:
                volume = float("nan")
            residual_total = (
                float(np.mean(residual_sq) * volume)
                if math.isfinite(volume) and volume > 0.0
                else float("nan")
            )
        else:
            residual_rms = float("nan")
            residual_total = float("nan")
        _append_history_value(model, "fixed_rms_residual_history", residual_rms)
        _append_history_value(model, "relative_fixed_rms_residual_history", float("nan"))
        _append_history_value(model, "fixed_total_residual_history", residual_total)
        _append_history_value(model, "relative_fixed_l2_residual_history", float("nan"))
        _append_history_value(model, "fixed_boundary_residual_history", float("nan"))

        print(
            f"Error (vs reference, vertex-L2): {total_error:.6e}, "
            f"Relative L2: {relative_l2:.6e}, "
            f"Boundary Error: {boundary_error:.6e}, "
            f"Residual RMS: {residual_rms:.6e}"
        )
        return True

    # ------------------------------------------------------------------
    # Geometry smoke metadata
    # ------------------------------------------------------------------

    def build_geometry_smoke_metadata(self, mesh) -> dict:
        verts = list(mesh.vertices)
        return {
            "num_vertices": len(verts),
            "mesh_dimension": 3,
            "problem": self.name,
            "cylinder_radius": self.cylinder_radius,
            "localized_traction": self.localized_traction,
            "traction_patch_sigma": self.traction_patch_sigma,
        }

    def build_fem_smoke_metadata(self, mesh, dt=None, t_end=None) -> dict:
        """Run FEM and return metadata dict (skips 2D mesh-plot in main.py)."""
        from ngsolve import VOL
        print("Running 3D FEM solve for smoke test...")
        gfu, fes = self.solve_fem(mesh)
        n_dofs = fes.ndof
        n_verts = len(list(mesh.vertices))
        n_elems = sum(1 for _ in mesh.Elements(VOL))

        # Sample displacement norm at a few interior points
        sample_pts = [(1.0, 0.3, 0.5), (2.0, 0.3, 0.5)]
        disp_norms = []
        for pt in sample_pts:
            try:
                val = gfu(mesh(pt[0], pt[1], pt[2]))
                disp_norms.append(float(sum(v**2 for v in val)**0.5))
            except Exception:
                disp_norms.append(float("nan"))

        print(f"FEM: {n_dofs} DOFs, {n_verts} vertices, {n_elems} elements")
        print(f"Displacement norms at sample points: {disp_norms}")
        return {
            "num_vertices": n_verts,
            "num_elements": n_elems,
            "num_dofs": n_dofs,
            "displacement_norms": disp_norms,
            "problem": self.name,
            "cylinder_radius": self.cylinder_radius,
            "localized_traction": self.localized_traction,
            "traction_patch_sigma": self.traction_patch_sigma,
        }


class ElasticityLocalizedCantilever3DProblem(ElasticityNotchedCantilever3DProblem):
    """Harder 3D elasticity case with narrow ligaments and localized end loading."""

    name = "elasticity_localized_cantilever_3d"
    description = (
        "Hard 3D elasticity cantilever with larger cylindrical cutouts and "
        "a localized Gaussian traction patch on the loaded end face"
    )

    def __init__(
        self,
        cylinder_radius: float = 0.285,
        traction_magnitude: float = 5e-3,
        traction_patch_sigma: float = 0.12,
        traction_patch_center_y: float | None = None,
        traction_patch_center_z: float | None = None,
        **kwargs,
    ):
        super().__init__(
            cylinder_radius=cylinder_radius,
            traction_magnitude=traction_magnitude,
            localized_traction=True,
            traction_patch_sigma=traction_patch_sigma,
            traction_patch_center_y=traction_patch_center_y,
            traction_patch_center_z=traction_patch_center_z,
            **kwargs,
        )
