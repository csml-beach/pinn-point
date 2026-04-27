"""
2D Allen-Cahn equation on a rectangular domain with circular obstacles.

This benchmark is designed to create thin interior interfaces that interact with
holes in the domain. That is a better match for interior residual-guided
adaptive sampling than boundary-dominated benchmarks such as plate-with-hole
elasticity.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import torch
from netgen.occ import OCCGeometry, WorkPlane, X, Y
from ngsolve import *
from scipy.stats import qmc
from torch.utils.data import TensorDataset

from paths import comparison_images_dir
from .base import PDEProblem


class AllenCahnObstacles2DProblem(PDEProblem):
    """Transient Allen-Cahn benchmark with interior interface/obstacle interaction."""

    name = "allen_cahn_obstacles_2d"
    description = (
        "Transient Allen-Cahn equation on a rectangular domain with circular "
        "obstacles and no-flux boundaries"
    )
    input_dim = 3
    output_dim = 1
    output_names = ("u",)
    has_time_input = True

    def __init__(
        self,
        length: float = 4.0,
        height: float = 2.0,
        obstacle_radius: float = 0.22,
        epsilon: float = 0.045,
        t_end: float = 0.30,
        dt: float = 0.002,
        fe_order: int = 2,
        supervised_time_slices: int = 6,
        reference_time_slices: int = 11,
        smoke_collocation_count: int = 1024,
    ):
        self.length = float(length)
        self.height = float(height)
        self.obstacle_radius = float(obstacle_radius)
        self.epsilon = float(epsilon)
        self.t_end = float(t_end)
        self.dt = float(dt)
        self.fe_order = int(fe_order)
        self.supervised_time_slices = max(int(supervised_time_slices), 2)
        self.reference_time_slices = max(int(reference_time_slices), 2)
        self.smoke_collocation_count = max(int(smoke_collocation_count), 128)
        self.obstacle_centers = (
            (1.55, 0.70),
            (2.55, 1.30),
        )
        self._sampling_mesh = None

    def get_domain_bounds(self) -> Tuple[float, float, float, float]:
        return (0.0, self.length, 0.0, self.height)

    def get_time_bounds(self) -> Tuple[float, float] | None:
        return (0.0, self.t_end)

    def get_dirichlet_boundaries(self) -> list:
        return []

    def create_mesh(self, maxh=None):
        if maxh is None:
            maxh = 0.12

        outer = WorkPlane().Rectangle(self.length, self.height).Face()
        outer.edges.Min(X).name = "left"
        outer.edges.Max(X).name = "right"
        outer.edges.Min(Y).name = "bottom"
        outer.edges.Max(Y).name = "top"

        hole_faces = []
        for idx, (cx, cy) in enumerate(self.obstacle_centers, start=1):
            hole = (
                WorkPlane()
                .Circle(self.obstacle_radius)
                .Face()
                .Move((cx, cy, 0.0))
            )
            hole.edges.name = f"obstacle{idx}"
            hole_faces.append(hole)

        geo = outer
        for hole in hole_faces:
            geo = geo - hole

        return Mesh(OCCGeometry(geo, dim=2).GenerateMesh(maxh=maxh)).Curve(3)

    def _initial_condition_torch(self, x_values: torch.Tensor, y_values: torch.Tensor) -> torch.Tensor:
        interface = 1.15 + 0.22 * torch.sin(np.pi * y_values / self.height)
        signed_distance = x_values - interface
        scale = np.sqrt(2.0) * self.epsilon
        return torch.tanh(-signed_distance / scale)

    def _initial_condition_ngsolve(self):
        interface = 1.15 + 0.22 * sin(np.pi * y / self.height)
        scale = np.sqrt(2.0) * self.epsilon
        argument = -(x - interface) / scale
        return 2.0 / (1.0 + exp(-2.0 * argument)) - 1.0

    def source_term(
        self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = torch.as_tensor(x, dtype=torch.float32)
        return torch.zeros_like(x)

    def pde_residual(
        self,
        model: Any,
        x: torch.Tensor,
        y: torch.Tensor,
        t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from config import DEVICE

        if t is None:
            raise ValueError("Allen-Cahn residual requires time coordinates")

        x = torch.as_tensor(x, dtype=torch.float32, device=DEVICE).reshape(-1)
        y = torch.as_tensor(y, dtype=torch.float32, device=DEVICE).reshape(-1)
        t = torch.as_tensor(t, dtype=torch.float32, device=DEVICE).reshape(-1)

        x = x.clone().detach().requires_grad_(True)
        y = y.clone().detach().requires_grad_(True)
        t = t.clone().detach().requires_grad_(True)

        u = model.forward(x, y, t).reshape(-1)
        u_t = model.compute_derivative(u, t, 1)
        u_xx = model.compute_derivative(u, x, 2)
        u_yy = model.compute_derivative(u, y, 2)
        reaction = u * u * u - u
        return u_t - (self.epsilon ** 2) * (u_xx + u_yy) + reaction

    def _allocate_boundary_counts(self, total: int) -> dict[str, int]:
        total = max(int(total), 6)
        perimeters = {
            "left": self.height,
            "right": self.height,
            "top": self.length,
            "bottom": self.length,
            "obstacles": 2.0 * np.pi * self.obstacle_radius * len(self.obstacle_centers),
        }
        total_weight = sum(perimeters.values())
        remaining = total
        counts = {}
        names = list(perimeters.keys())
        for idx, name in enumerate(names):
            if idx == len(names) - 1:
                counts[name] = remaining
            else:
                count = max(1, int(round(total * perimeters[name] / total_weight)))
                count = min(count, remaining - (len(names) - idx - 1))
                counts[name] = count
                remaining -= count
        return counts

    def _sample_times(self, count: int, device: torch.device) -> torch.Tensor:
        if count <= 0:
            return torch.empty(0, dtype=torch.float32, device=device)
        return torch.rand(count, dtype=torch.float32, device=device) * self.t_end

    def _normal_derivative(
        self,
        model: Any,
        x_values: torch.Tensor,
        y_values: torch.Tensor,
        t_values: torch.Tensor,
        nx: torch.Tensor,
        ny: torch.Tensor,
    ) -> torch.Tensor:
        x_values = x_values.clone().detach().requires_grad_(True)
        y_values = y_values.clone().detach().requires_grad_(True)
        t_values = t_values.clone().detach().requires_grad_(True)
        u = model.forward(x_values, y_values, t_values).reshape(-1)
        u_x = model.compute_derivative(u, x_values, 1)
        u_y = model.compute_derivative(u, y_values, 1)
        return u_x * nx + u_y * ny

    def boundary_loss(self, model: Any, num_boundary_points: int = 100) -> torch.Tensor:
        from config import DEVICE

        counts = self._allocate_boundary_counts(num_boundary_points)
        losses = []

        left_y = torch.linspace(0.0, self.height, counts["left"] + 2, device=DEVICE)[1:-1]
        left_x = torch.zeros_like(left_y)
        left_t = self._sample_times(len(left_y), DEVICE)
        left_dn = self._normal_derivative(
            model,
            left_x,
            left_y,
            left_t,
            nx=-torch.ones_like(left_x),
            ny=torch.zeros_like(left_x),
        )
        losses.append(torch.mean(torch.square(left_dn)))

        right_y = torch.linspace(0.0, self.height, counts["right"] + 2, device=DEVICE)[1:-1]
        right_x = torch.full_like(right_y, self.length)
        right_t = self._sample_times(len(right_y), DEVICE)
        right_dn = self._normal_derivative(
            model,
            right_x,
            right_y,
            right_t,
            nx=torch.ones_like(right_x),
            ny=torch.zeros_like(right_x),
        )
        losses.append(torch.mean(torch.square(right_dn)))

        top_x = torch.linspace(0.0, self.length, counts["top"] + 2, device=DEVICE)[1:-1]
        top_y = torch.full_like(top_x, self.height)
        top_t = self._sample_times(len(top_x), DEVICE)
        top_dn = self._normal_derivative(
            model,
            top_x,
            top_y,
            top_t,
            nx=torch.zeros_like(top_x),
            ny=torch.ones_like(top_x),
        )
        losses.append(torch.mean(torch.square(top_dn)))

        bottom_x = torch.linspace(0.0, self.length, counts["bottom"] + 2, device=DEVICE)[1:-1]
        bottom_y = torch.zeros_like(bottom_x)
        bottom_t = self._sample_times(len(bottom_x), DEVICE)
        bottom_dn = self._normal_derivative(
            model,
            bottom_x,
            bottom_y,
            bottom_t,
            nx=torch.zeros_like(bottom_x),
            ny=-torch.ones_like(bottom_x),
        )
        losses.append(torch.mean(torch.square(bottom_dn)))

        obstacle_count = max(counts["obstacles"], len(self.obstacle_centers))
        per_circle = np.random.multinomial(
            obstacle_count,
            np.full(len(self.obstacle_centers), 1.0 / len(self.obstacle_centers)),
        )
        obstacle_terms = []
        for circle_count, (cx, cy) in zip(per_circle, self.obstacle_centers):
            if circle_count <= 0:
                continue
            theta = 2.0 * np.pi * torch.rand(circle_count, dtype=torch.float32, device=DEVICE)
            x_values = cx + self.obstacle_radius * torch.cos(theta)
            y_values = cy + self.obstacle_radius * torch.sin(theta)
            t_values = self._sample_times(circle_count, DEVICE)
            dn = self._normal_derivative(
                model,
                x_values,
                y_values,
                t_values,
                nx=torch.cos(theta),
                ny=torch.sin(theta),
            )
            obstacle_terms.append(torch.mean(torch.square(dn)))
        if obstacle_terms:
            losses.append(torch.mean(torch.stack(obstacle_terms)))

        return torch.mean(torch.stack(losses))

    def _build_space(self, mesh):
        return H1(mesh, order=self.fe_order, autoupdate=True)

    def solve_fem_time_series(
        self,
        mesh,
        dt: float | None = None,
        t_end: float | None = None,
        snapshot_times: List[float] | None = None,
    ) -> Dict[str, Any]:
        dt = float(self.dt if dt is None else dt)
        t_end = float(self.t_end if t_end is None else t_end)
        snapshot_times = snapshot_times or [0.0, 0.5 * t_end, t_end]
        snapshot_targets = sorted({max(0.0, min(t_end, float(t))) for t in snapshot_times})

        fes = self._build_space(mesh)
        u = fes.TrialFunction()
        v = fes.TestFunction()
        system = BilinearForm(u * v * dx + dt * (self.epsilon ** 2) * grad(u) * grad(v) * dx)
        system.Assemble()
        inv = system.mat.Inverse()

        gfu = GridFunction(fes)
        gfu.Set(self._initial_condition_ngsolve())

        vertex_coords = [(float(vtx.point[0]), float(vtx.point[1])) for vtx in mesh.vertices]
        snapshots: List[Dict[str, Any]] = []

        def capture_snapshot(time_value: float):
            values = []
            for x_coord, y_coord in vertex_coords:
                values.append(float(gfu(mesh(x_coord, y_coord))))
            snapshots.append(
                {
                    "time": float(time_value),
                    "u": values,
                }
            )

        capture_snapshot(0.0)
        current_time = 0.0
        next_target_index = 0
        while (
            next_target_index < len(snapshot_targets)
            and snapshot_targets[next_target_index] <= 0.0 + 1e-12
        ):
            next_target_index += 1

        while current_time < t_end - 0.5 * dt:
            rhs = LinearForm(fes)
            rhs += (gfu - dt * (gfu * gfu * gfu - gfu)) * v * dx
            rhs.Assemble()
            gfu.vec.data = inv * rhs.vec
            current_time += dt

            while (
                next_target_index < len(snapshot_targets)
                and current_time >= snapshot_targets[next_target_index] - 0.5 * dt
            ):
                capture_snapshot(snapshot_targets[next_target_index])
                next_target_index += 1

        if snapshots[-1]["time"] < t_end - 1e-12:
            capture_snapshot(t_end)

        return {
            "space": fes,
            "state": gfu,
            "snapshots": snapshots,
            "vertex_coords": vertex_coords,
        }

    def solve_fem(self, mesh) -> Tuple[Any, Any]:
        result = self.solve_fem_time_series(mesh)
        return result["state"], result["space"]

    def create_training_dataset(
        self, mesh, fem_solution: Any | None = None, seed: int | None = None
    ) -> TensorDataset:
        snapshot_times = np.linspace(0.0, self.t_end, self.supervised_time_slices).tolist()
        result = self.solve_fem_time_series(mesh, snapshot_times=snapshot_times)
        vertex_coords = np.asarray(result["vertex_coords"], dtype=np.float32)

        coordinate_batches = []
        target_batches = []
        for snapshot in result["snapshots"]:
            time_column = np.full((len(vertex_coords), 1), float(snapshot["time"]), dtype=np.float32)
            coordinates = np.hstack((vertex_coords, time_column))
            values = np.asarray(snapshot["u"], dtype=np.float32).reshape(-1, 1)
            coordinate_batches.append(coordinates)
            target_batches.append(values)

        return TensorDataset(
            torch.tensor(np.vstack(coordinate_batches), dtype=torch.float32),
            torch.tensor(np.vstack(target_batches), dtype=torch.float32),
        )

    def _point_is_in_domain(self, mesh, x_coord: float, y_coord: float) -> bool:
        try:
            return mesh(float(x_coord), float(y_coord)).nr != -1
        except Exception:
            return False

    def create_smoke_collocation_points(
        self, mesh, seed: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        seed_value = 1729 if seed is None else int(seed)
        sampler = qmc.Halton(d=3, scramble=True, seed=seed_value)
        x_min, x_max, y_min, y_max = self.get_domain_bounds()
        t_min, t_max = self.get_time_bounds()

        accepted = []
        batch_size = max(4 * self.smoke_collocation_count, 512)
        for _ in range(50):
            unit = sampler.random(batch_size)
            x_values = x_min + (x_max - x_min) * unit[:, 0]
            y_values = y_min + (y_max - y_min) * unit[:, 1]
            t_values = t_min + (t_max - t_min) * unit[:, 2]
            for x_value, y_value, t_value in zip(x_values, y_values, t_values):
                if self._point_is_in_domain(mesh, float(x_value), float(y_value)):
                    accepted.append((float(x_value), float(y_value), float(t_value)))
                    if len(accepted) >= self.smoke_collocation_count:
                        break
            if len(accepted) >= self.smoke_collocation_count:
                break

        if len(accepted) < self.smoke_collocation_count:
            raise ValueError(
                f"Only generated {len(accepted)}/{self.smoke_collocation_count} Allen-Cahn collocation points"
            )

        points = np.asarray(accepted, dtype=np.float32)
        return (
            torch.tensor(points[:, 0], dtype=torch.float32),
            torch.tensor(points[:, 1], dtype=torch.float32),
            torch.tensor(points[:, 2], dtype=torch.float32),
        )

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
        purpose_offsets = {
            "initial": 101,
            "train": 211,
            "validation": 307,
            "adaptive_score": 401,
        }
        seed_value = (
            purpose_offsets.get(str(purpose), 503) + 1009 * int(iteration)
            if seed is None
            else int(seed) + purpose_offsets.get(str(purpose), 503) + 1009 * int(iteration)
        )
        sampler = qmc.Halton(d=1, scramble=True, seed=seed_value)
        unit = sampler.random(len(x)).reshape(-1)
        t_values = self.t_end * unit
        return x, y, torch.tensor(t_values, dtype=torch.float32, device=x.device)

    def _extract_triangulation(self, mesh):
        verts = [(float(v.point[0]), float(v.point[1])) for v in mesh.vertices]
        tris = []
        for el in mesh.Elements():
            if getattr(el, "vertices", None) and len(el.vertices) == 3:
                tris.append([v.nr for v in el.vertices])
        if not verts or not tris:
            raise RuntimeError("Could not extract triangulation for allen_cahn_obstacles_2d")
        xs, ys = zip(*verts)
        return np.asarray(verts, dtype=float), mtri.Triangulation(xs, ys, tris)

    def _save_scalar_plot(self, triang, values, *, title: str, filename: str) -> str:
        out_path = os.path.join(comparison_images_dir(), filename)
        fig, ax = plt.subplots(figsize=(8, 4))
        tpc = ax.tripcolor(triang, np.asarray(values, dtype=float), shading="gouraud", cmap="coolwarm")
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(tpc, ax=ax)
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return out_path

    def _save_mesh_plot(self, triang, filename: str = "geometry_mesh.png") -> str:
        out_path = os.path.join(comparison_images_dir(), filename)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.triplot(triang, color="black", linewidth=0.45)
        ax.set_aspect("equal")
        ax.set_title("Geometry mesh")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return out_path

    def build_geometry_smoke_metadata(self, mesh) -> dict[str, Any] | None:
        _, triang = self._extract_triangulation(mesh)
        return {
            "mesh_plot": self._save_mesh_plot(triang),
            "num_vertices": len(mesh.vertices),
            "num_elements": len(list(mesh.Elements())),
            "mesh_dimension": int(getattr(mesh, "dim", 0)),
        }

    def create_reference_solution(self, mesh_size_factor: float = 0.05):
        reference_mesh = self.create_mesh(maxh=mesh_size_factor)
        snapshot_times = np.linspace(0.0, self.t_end, self.reference_time_slices).tolist()
        reference_solution = self.solve_fem_time_series(reference_mesh, snapshot_times=snapshot_times)
        coords, triang = self._extract_triangulation(reference_mesh)
        reference_solution["vertex_coords"] = coords.astype(np.float32)
        reference_solution["triangulation"] = triang
        return reference_mesh, reference_solution

    def evaluate_model_against_reference(
        self,
        model: Any,
        reference_mesh: Any,
        reference_solution: Any,
        *,
        export_images: bool = False,
        iteration: int | None = None,
    ) -> bool:
        if not isinstance(reference_solution, dict) or "snapshots" not in reference_solution:
            return False

        from paths import comparison_images_dir, method_images_dir

        device = next(model.parameters()).device
        vertex_coords = np.asarray(reference_solution["vertex_coords"], dtype=np.float32)
        coordinate_batches = []
        reference_batches = []
        final_reference = None
        final_prediction = None

        for snapshot in reference_solution["snapshots"]:
            time_value = float(snapshot["time"])
            time_column = np.full((len(vertex_coords), 1), time_value, dtype=np.float32)
            coordinate_batches.append(np.hstack((vertex_coords, time_column)))
            values = np.asarray(snapshot["u"], dtype=np.float32).reshape(-1)
            reference_batches.append(values)

        all_coords = torch.tensor(np.vstack(coordinate_batches), dtype=torch.float32, device=device)
        with torch.no_grad():
            prediction = model.forward(all_coords).reshape(-1)
        pred = prediction.detach().cpu().numpy().reshape(-1)
        reference_values = np.concatenate(reference_batches)

        error_sq = (pred - reference_values) ** 2
        reference_sq = reference_values ** 2
        total_error = float(np.mean(error_sq))
        boundary_error = float(model.loss_boundary_condition().detach().cpu())
        error_rms = float(np.sqrt(np.mean(error_sq)))
        reference_rms = float(np.sqrt(np.mean(reference_sq))) if np.mean(reference_sq) > 0 else float("nan")
        relative_l2 = (
            float(np.sqrt(total_error / float(np.mean(reference_sq))))
            if float(np.mean(reference_sq)) > 0
            else float("nan")
        )
        relative_rms = float(error_rms / reference_rms) if reference_rms > 0 else float("nan")

        model.total_error_history.append(total_error)
        model.boundary_error_history.append(boundary_error)
        model.relative_l2_error_history.append(relative_l2)
        model.total_error_rms_history.append(error_rms)
        model.relative_error_rms_history.append(relative_rms)

        tx = all_coords[:, 0].clone().detach().requires_grad_(True)
        ty = all_coords[:, 1].clone().detach().requires_grad_(True)
        tt = all_coords[:, 2].clone().detach().requires_grad_(True)
        with torch.enable_grad():
            residual = model.PDE_residual(tx, ty, tt)
        residual_sq = np.square(residual.detach().cpu().numpy().reshape(-1))
        total_residual = float(np.mean(residual_sq))
        residual_rms = float(np.sqrt(np.mean(residual_sq)))
        relative_residual = float(residual_rms / reference_rms) if reference_rms > 0 else float("nan")

        model.fixed_total_residual_history.append(total_residual)
        model.relative_fixed_l2_residual_history.append(relative_residual)
        model.fixed_boundary_residual_history.append(boundary_error)
        model.fixed_rms_residual_history.append(residual_rms)
        model.relative_fixed_rms_residual_history.append(relative_residual)

        if export_images and iteration is not None:
            final_snapshot = reference_solution["snapshots"][-1]
            final_time = float(final_snapshot["time"])
            final_coords = torch.tensor(
                np.hstack((vertex_coords, np.full((len(vertex_coords), 1), final_time, dtype=np.float32))),
                dtype=torch.float32,
                device=device,
            )
            with torch.no_grad():
                final_prediction = model.forward(final_coords).reshape(-1).detach().cpu().numpy()
            final_reference = np.asarray(final_snapshot["u"], dtype=np.float32)
            triang = reference_solution["triangulation"]

            reference_path = os.path.join(comparison_images_dir(), "reference_solution.png")
            if not os.path.exists(reference_path):
                self._save_scalar_plot(
                    triang,
                    final_reference,
                    title=f"Reference phase field at t={final_time:.2f}",
                    filename="reference_solution.png",
                )

            method_dir = method_images_dir(getattr(model, "method_name", "unknown"))
            self._save_scalar_plot(
                triang,
                final_prediction,
                title=f"Predicted phase field at t={final_time:.2f}",
                filename=os.path.join(method_dir, f"solutions_{iteration}.png"),
            )
            self._save_scalar_plot(
                triang,
                np.square(final_prediction - final_reference),
                title=f"Squared error at t={final_time:.2f}",
                filename=os.path.join(method_dir, f"errors_{iteration}.png"),
            )

        print(
            "Allen-Cahn error: "
            f"MSE={total_error:.6e}, Relative L2 error={relative_l2:.6e}, "
            f"Boundary loss={boundary_error:.6e}"
        )
        print(
            "Allen-Cahn residual: "
            f"MSE={total_residual:.6e}, Relative RMS residual={relative_residual:.6e}"
        )
        return True

    def build_fem_smoke_metadata(
        self,
        mesh,
        *,
        dt: float | None = None,
        t_end: float | None = None,
    ) -> dict[str, Any] | None:
        result = self.solve_fem_time_series(
            mesh,
            dt=dt,
            t_end=t_end,
            snapshot_times=[0.0, 0.5 * float(self.t_end if t_end is None else t_end), float(self.t_end if t_end is None else t_end)],
        )
        coords, triang = self._extract_triangulation(mesh)
        slice_plots = []
        for snapshot in result["snapshots"]:
            time_tag = f"{snapshot['time']:.2f}".replace(".", "p")
            plot_path = self._save_scalar_plot(
                triang,
                snapshot["u"],
                title=f"Phase field at t={snapshot['time']:.2f}",
                filename=f"phase_field_t{time_tag}.png",
            )
            slice_plots.append(
                {
                    "time": float(snapshot["time"]),
                    "solution_plot": plot_path,
                }
            )

        final_values = np.asarray(result["snapshots"][-1]["u"], dtype=np.float32)
        return {
            "mesh_plot": self._save_mesh_plot(triang),
            "num_vertices": len(mesh.vertices),
            "num_elements": len(list(mesh.Elements())),
            "fem_ndof": int(result["space"].ndof),
            "slice_plots": slice_plots,
            "probe_values": {
                "phase_mean_t_end": float(np.mean(final_values)),
                "phase_min_t_end": float(np.min(final_values)),
                "phase_max_t_end": float(np.max(final_values)),
            },
        }
