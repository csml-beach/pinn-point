"""
Fixed channel-obstacle transient Navier-Stokes problem.

This module currently provides:
    - Netgen/OCC geometry and mesh generation
    - transient FEM reference solve with Taylor-Hood elements
    - PINN residual and boundary/initial losses for (u, v, p)(x, y, t)
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import TensorDataset
from netgen.occ import OCCGeometry, WorkPlane, X, Y
from ngsolve import *

from .base import PDEProblem


class NavierStokesChannelObstacleProblem(PDEProblem):
    """Channel with a contraction and staggered circular obstacles."""

    name = "navier_stokes_channel_obstacle"
    input_dim = 3
    output_dim = 3
    output_names = ("u", "v", "p")
    has_time_input = True
    description = (
        "Time-dependent Navier-Stokes benchmark on "
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
        supervised_time_slices: int = 6,
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
        self.supervised_time_slices = max(int(supervised_time_slices), 2)
        self.obstacle_centers = (
            (2.0, 0.95),
            (4.4, 2.05),
            (5.7, 1.15),
        )
        self._sampling_mesh = None
        self._initial_state_cache = None

    def _throat_bounds(self) -> Tuple[float, float, float, float]:
        throat_x0 = self.throat_center_x - 0.5 * self.throat_length
        throat_x1 = self.throat_center_x + 0.5 * self.throat_length
        mid_y = 0.5 * self.height
        throat_y0 = mid_y - self.throat_half_height
        throat_y1 = mid_y + self.throat_half_height
        return throat_x0, throat_x1, throat_y0, throat_y1

    def _build_occ_face(self):
        length = self.length
        height = self.height
        throat_x0, throat_x1, throat_y0, throat_y1 = self._throat_bounds()
        top_cut_height = max(0.0, height - throat_y1)
        bottom_cut_height = max(0.0, throat_y0)

        channel = WorkPlane().Rectangle(length, height).Face()

        # Contraction by subtracting top/bottom blocks in the throat region.
        top_cut = (
            WorkPlane()
            .MoveTo(throat_x0, throat_y1)
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
            .Move((*self.obstacle_centers[0], 0.0))
        )
        circle2 = (
            WorkPlane()
            .Circle(self.obstacle_radius)
            .Face()
            .Move((*self.obstacle_centers[1], 0.0))
        )
        circle3 = (
            WorkPlane()
            .Circle(self.obstacle_radius)
            .Face()
            .Move((*self.obstacle_centers[2], 0.0))
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

    def _inflow_profile_torch(self, y_values: torch.Tensor) -> torch.Tensor:
        h = max(self.height, 1e-8)
        return self.inlet_peak_velocity * 4.0 * y_values * (self.height - y_values) / (
            h * h
        )

    def _build_taylor_hood_space(self, mesh):
        k = self.fe_order
        velocity_space = VectorH1(mesh, order=k, dirichlet="wall|obstacle|inlet")
        pressure_space = H1(mesh, order=max(1, k - 1))
        return velocity_space * pressure_space

    def _assemble_stokes_system(self, space):
        (u, p), (v, q) = space.TnT()
        stokes_form = (
            self.viscosity * InnerProduct(grad(u), grad(v))
            - div(u) * q
            - div(v) * p
        ) * dx

        stiffness = BilinearForm(space)
        stiffness += stokes_form
        stiffness.Assemble()
        return stiffness, stokes_form

    def _solve_initial_stokes_state(self, mesh):
        space = self._build_taylor_hood_space(mesh)
        gfu = GridFunction(space)
        velocity, _ = gfu.components
        velocity.Set(self._inflow_profile(), definedon=mesh.Boundaries("inlet"))

        stiffness, stokes_form = self._assemble_stokes_system(space)
        rhs = LinearForm(space)
        rhs.Assemble()
        residual = rhs.vec.CreateVector()
        residual.data = rhs.vec - stiffness.mat * gfu.vec
        gfu.vec.data += stiffness.mat.Inverse(space.FreeDofs()) * residual
        return {
            "space": space,
            "state": gfu,
            "stokes_form": stokes_form,
            "stiffness": stiffness,
        }

    def _get_sampling_mesh(self):
        if self._sampling_mesh is None:
            self._sampling_mesh = self.create_mesh(maxh=0.20)
        return self._sampling_mesh

    def _get_initial_state_cache(self):
        if self._initial_state_cache is None:
            sampling_mesh = self._get_sampling_mesh()
            self._initial_state_cache = self._solve_initial_stokes_state(sampling_mesh)
        return self._initial_state_cache

    def _sample_interior_points(
        self, num_points: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from methods.sampling import points_to_tensors, sample_points_in_domain

        bounds = self.get_domain_bounds()
        rng = np.random.RandomState()
        mesh = self._get_sampling_mesh()
        points = sample_points_in_domain(
            mesh,
            num_points,
            batch_generator=lambda n: np.column_stack(
                (
                    rng.uniform(bounds[0], bounds[1], size=n),
                    rng.uniform(bounds[2], bounds[3], size=n),
                )
            ),
            warn_label="navier_stokes_interior_points",
        )
        x_points, y_points = points_to_tensors(points)
        return x_points.to(device), y_points.to(device)

    def _sample_time_points(
        self, count: int, device: torch.device
    ) -> torch.Tensor:
        return self.dt + (self.t_end - self.dt) * torch.rand(
            count, dtype=torch.float32, device=device
        )

    def _sample_segment_points(
        self,
        segments: list[tuple[tuple[float, float], tuple[float, float]]],
        count: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if count <= 0:
            return (
                torch.empty(0, dtype=torch.float32, device=device),
                torch.empty(0, dtype=torch.float32, device=device),
            )

        lengths = np.asarray(
            [
                np.hypot(p1[0] - p0[0], p1[1] - p0[1])
                for p0, p1 in segments
            ],
            dtype=float,
        )
        probs = lengths / max(np.sum(lengths), 1e-12)
        counts = np.random.multinomial(count, probs)

        x_batches = []
        y_batches = []
        for segment_count, (start, end) in zip(counts, segments):
            if segment_count <= 0:
                continue
            s = torch.rand(segment_count, dtype=torch.float32, device=device)
            x_batches.append(start[0] + s * (end[0] - start[0]))
            y_batches.append(start[1] + s * (end[1] - start[1]))

        if not x_batches:
            return (
                torch.empty(0, dtype=torch.float32, device=device),
                torch.empty(0, dtype=torch.float32, device=device),
            )
        return torch.cat(x_batches), torch.cat(y_batches)

    def _sample_wall_points(
        self, count: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        throat_x0, throat_x1, throat_y0, throat_y1 = self._throat_bounds()
        outer_wall_segments = [
            ((0.0, 0.0), (throat_x0, 0.0)),
            ((throat_x1, 0.0), (self.length, 0.0)),
            ((0.0, self.height), (throat_x0, self.height)),
            ((throat_x1, self.height), (self.length, self.height)),
        ]
        throat_segments = [
            ((throat_x0, throat_y1), (throat_x1, throat_y1)),
            ((throat_x0, throat_y0), (throat_x1, throat_y0)),
            ((throat_x0, 0.0), (throat_x0, throat_y0)),
            ((throat_x0, throat_y1), (throat_x0, self.height)),
            ((throat_x1, 0.0), (throat_x1, throat_y0)),
            ((throat_x1, throat_y1), (throat_x1, self.height)),
        ]

        outer_count = max(1, count // 2)
        throat_count = max(1, count - outer_count)
        outer_x, outer_y = self._sample_segment_points(
            outer_wall_segments, outer_count, device
        )
        throat_x, throat_y = self._sample_segment_points(
            throat_segments, throat_count, device
        )
        return torch.cat((outer_x, throat_x)), torch.cat((outer_y, throat_y))

    def _sample_circle_points(
        self, count: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if count <= 0:
            return (
                torch.empty(0, dtype=torch.float32, device=device),
                torch.empty(0, dtype=torch.float32, device=device),
            )
        per_circle = np.random.multinomial(
            count, np.full(len(self.obstacle_centers), 1.0 / len(self.obstacle_centers))
        )
        x_batches = []
        y_batches = []
        for circle_count, (cx, cy) in zip(per_circle, self.obstacle_centers):
            if circle_count <= 0:
                continue
            theta = 2.0 * np.pi * torch.rand(circle_count, dtype=torch.float32, device=device)
            x_batches.append(cx + self.obstacle_radius * torch.cos(theta))
            y_batches.append(cy + self.obstacle_radius * torch.sin(theta))
        return torch.cat(x_batches), torch.cat(y_batches)

    def _evaluate_initial_velocity(
        self, x_points: torch.Tensor, y_points: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cached = self._get_initial_state_cache()
        mesh = self._get_sampling_mesh()
        velocity_field, _ = cached["state"].components
        u_values = []
        v_values = []
        for x_coord, y_coord in zip(
            x_points.detach().cpu().numpy(), y_points.detach().cpu().numpy()
        ):
            mip = mesh(float(x_coord), float(y_coord))
            vel = velocity_field(mip)
            u_values.append(float(vel[0]))
            v_values.append(float(vel[1]))
        device = x_points.device
        return (
            torch.tensor(u_values, dtype=torch.float32, device=device),
            torch.tensor(v_values, dtype=torch.float32, device=device),
        )

    def _split_prediction(
        self, prediction: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if prediction.ndim != 2 or prediction.shape[1] != 3:
            raise ValueError("Navier-Stokes model must output a tensor of shape (N, 3)")
        return prediction[:, 0], prediction[:, 1], prediction[:, 2]

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

        initial_state = self._solve_initial_stokes_state(mesh)
        Xspace = initial_state["space"]
        gfu = initial_state["state"]
        velocity, pressure = gfu.components

        (u, p), (v, q) = Xspace.TnT()
        stokes_form = initial_state["stokes_form"]
        stiffness = initial_state["stiffness"]

        mstar = BilinearForm(Xspace)
        mstar += InnerProduct(u, v) * dx + dt * stokes_form
        mstar.Assemble()
        inv = mstar.mat.Inverse(Xspace.FreeDofs())

        conv = BilinearForm(Xspace, nonassemble=True)
        conv += (Grad(u) * u) * v * dx

        snapshots: List[Dict[str, Any]] = []
        vertex_coords = [(float(vtx.point[0]), float(vtx.point[1])) for vtx in mesh.vertices]

        def capture_snapshot(time_value: float):
            vel_vals = []
            u_vals = []
            v_vals = []
            pressure_vals = []
            for x_coord, y_coord in vertex_coords:
                mip = mesh(x_coord, y_coord)
                vel = velocity(mip)
                pres = pressure(mip)
                u_comp = float(vel[0])
                v_comp = float(vel[1])
                vel_mag = float((u_comp ** 2 + v_comp ** 2) ** 0.5)
                vel_vals.append(vel_mag)
                u_vals.append(u_comp)
                v_vals.append(v_comp)
                pressure_vals.append(float(pres))
            snapshots.append(
                {
                    "time": float(time_value),
                    "u_velocity": u_vals,
                    "v_velocity": v_vals,
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
            res = initial_state["state"].vec.CreateVector()
            res.data = (stiffness.mat + conv.mat) * gfu.vec
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

    def data_loss(
        self, model: Any, coordinates: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        predictions = model.forward(coordinates)
        velocity_predictions = predictions[:, :2]
        targets = torch.as_tensor(
            targets,
            dtype=velocity_predictions.dtype,
            device=velocity_predictions.device,
        )
        if targets.ndim != 2 or targets.shape[1] != 2:
            raise ValueError(
                "Navier-Stokes coarse supervision expects targets shaped (N, 2) for (u, v)"
            )
        return torch.mean(torch.square(velocity_predictions - targets))

    def create_training_dataset(
        self, mesh, fem_solution: Any | None = None, seed: int | None = None
    ) -> TensorDataset:
        snapshot_times = np.linspace(0.0, self.t_end, self.supervised_time_slices).tolist()
        result = self.solve_fem_time_series(mesh, snapshot_times=snapshot_times)
        vertex_coords = np.asarray(result["vertex_coords"], dtype=np.float32)

        coordinate_batches = []
        target_batches = []
        for snapshot in result["snapshots"]:
            time_column = np.full(
                (len(vertex_coords), 1), float(snapshot["time"]), dtype=np.float32
            )
            coordinates = np.hstack((vertex_coords, time_column))
            velocity_targets = np.column_stack(
                (
                    np.asarray(snapshot["u_velocity"], dtype=np.float32),
                    np.asarray(snapshot["v_velocity"], dtype=np.float32),
                )
            )
            coordinate_batches.append(coordinates)
            target_batches.append(velocity_targets)

        dataset_coordinates = torch.tensor(
            np.vstack(coordinate_batches), dtype=torch.float32
        )
        dataset_targets = torch.tensor(np.vstack(target_batches), dtype=torch.float32)
        return TensorDataset(dataset_coordinates, dataset_targets)

    def pde_residual(
        self,
        model: Any,
        x: torch.Tensor,
        y: torch.Tensor,
        t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from config import DEVICE

        x = torch.as_tensor(x, dtype=torch.float32, device=DEVICE).clone().detach().requires_grad_(True)
        y = torch.as_tensor(y, dtype=torch.float32, device=DEVICE).clone().detach().requires_grad_(True)
        if t is None:
            t = (
                self.dt + (self.t_end - self.dt) * torch.rand_like(x)
            ).clone().detach().requires_grad_(True)
        else:
            t = torch.as_tensor(t, dtype=torch.float32, device=DEVICE).clone().detach().requires_grad_(True)

        prediction = model.forward(x, y, t)
        u, v, p = self._split_prediction(prediction)

        u_t = self.compute_derivative(u, t, 1)
        v_t = self.compute_derivative(v, t, 1)

        u_x = self.compute_derivative(u, x, 1)
        u_y = self.compute_derivative(u, y, 1)
        v_x = self.compute_derivative(v, x, 1)
        v_y = self.compute_derivative(v, y, 1)

        u_xx = self.compute_derivative(u, x, 2)
        u_yy = self.compute_derivative(u, y, 2)
        v_xx = self.compute_derivative(v, x, 2)
        v_yy = self.compute_derivative(v, y, 2)

        p_x = self.compute_derivative(p, x, 1)
        p_y = self.compute_derivative(p, y, 1)

        momentum_x = u_t + u * u_x + v * u_y + p_x - self.viscosity * (u_xx + u_yy)
        momentum_y = v_t + u * v_x + v * v_y + p_y - self.viscosity * (v_xx + v_yy)
        continuity = u_x + v_y

        return torch.sqrt(momentum_x.square() + momentum_y.square() + continuity.square() + 1e-12)

    def boundary_loss(self, model: Any, num_boundary_points: int = 100) -> torch.Tensor:
        device = next(model.parameters()).device

        num_boundary_points = max(int(num_boundary_points), 16)
        inlet_count = max(1, num_boundary_points // 5)
        outlet_count = max(1, num_boundary_points // 5)
        wall_count = max(1, num_boundary_points // 4)
        circle_count = max(1, num_boundary_points - inlet_count - outlet_count - wall_count)
        initial_count = max(1, num_boundary_points)

        inlet_y = torch.rand(inlet_count, dtype=torch.float32, device=device) * self.height
        inlet_x = torch.zeros(inlet_count, dtype=torch.float32, device=device)
        inlet_t = self._sample_time_points(inlet_count, device)
        inlet_prediction = model.forward(inlet_x, inlet_y, inlet_t)
        inlet_u, inlet_v, _ = self._split_prediction(inlet_prediction)
        target_inlet_u = self._inflow_profile_torch(inlet_y)
        inlet_loss = torch.mean((inlet_u - target_inlet_u).square() + inlet_v.square())

        outlet_y = torch.rand(outlet_count, dtype=torch.float32, device=device) * self.height
        outlet_x = torch.full((outlet_count,), self.length, dtype=torch.float32, device=device)
        outlet_t = self._sample_time_points(outlet_count, device)
        outlet_prediction = model.forward(outlet_x, outlet_y, outlet_t)
        _, _, outlet_p = self._split_prediction(outlet_prediction)
        outlet_loss = torch.mean(outlet_p.square())

        wall_x, wall_y = self._sample_wall_points(wall_count, device)
        wall_t = self._sample_time_points(len(wall_x), device)
        wall_prediction = model.forward(wall_x, wall_y, wall_t)
        wall_u, wall_v, _ = self._split_prediction(wall_prediction)
        wall_loss = torch.mean(wall_u.square() + wall_v.square())

        circle_x, circle_y = self._sample_circle_points(circle_count, device)
        circle_t = self._sample_time_points(len(circle_x), device)
        circle_prediction = model.forward(circle_x, circle_y, circle_t)
        circle_u, circle_v, _ = self._split_prediction(circle_prediction)
        circle_loss = torch.mean(circle_u.square() + circle_v.square())

        init_x, init_y = self._sample_interior_points(initial_count, device)
        init_t = torch.zeros(initial_count, dtype=torch.float32, device=device)
        init_prediction = model.forward(init_x, init_y, init_t)
        init_u, init_v, _ = self._split_prediction(init_prediction)
        target_u0, target_v0 = self._evaluate_initial_velocity(init_x, init_y)
        initial_loss = torch.mean((init_u - target_u0).square() + (init_v - target_v0).square())

        return 0.2 * (
            inlet_loss + outlet_loss + wall_loss + circle_loss + initial_loss
        )

    def source_term(
        self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = torch.as_tensor(x, dtype=torch.float32)
        return torch.zeros_like(x)

    def get_time_bounds(self) -> Tuple[float, float]:
        return (0.0, self.t_end)

    def evaluate_smoke_diagnostics(
        self,
        model: Any,
        dataset: TensorDataset,
        mesh,
        *,
        inlet_probe_count: int = 128,
    ) -> Dict[str, Any]:
        """Evaluate simple anti-collapse diagnostics for PINN smoke runs."""
        device = next(model.parameters()).device
        coords, targets = dataset.tensors
        coords = torch.as_tensor(coords, dtype=torch.float32, device=device)
        targets = torch.as_tensor(targets, dtype=torch.float32, device=device)

        with torch.no_grad():
            predictions = model.forward(coords)
            pred_u = predictions[:, 0]
            pred_v = predictions[:, 1]
            target_u = targets[:, 0]
            target_v = targets[:, 1]

            pred_speed = torch.sqrt(pred_u.square() + pred_v.square())
            target_speed = torch.sqrt(target_u.square() + target_v.square())

            unique_times = torch.unique(coords[:, 2].detach().cpu(), sorted=True)
            time_slice_metrics = []
            for time_value in unique_times.tolist():
                mask = torch.isclose(
                    coords[:, 2],
                    torch.tensor(time_value, dtype=coords.dtype, device=device),
                    atol=1e-6,
                    rtol=0.0,
                )
                if not bool(mask.any()):
                    continue
                time_slice_metrics.append(
                    {
                        "time": float(time_value),
                        "mean_pred_speed": float(pred_speed[mask].mean().cpu()),
                        "mean_target_speed": float(target_speed[mask].mean().cpu()),
                        "velocity_rmse": float(
                            torch.sqrt(
                                torch.mean(
                                    (pred_u[mask] - target_u[mask]).square()
                                    + (pred_v[mask] - target_v[mask]).square()
                                )
                            ).cpu()
                        ),
                    }
                )

            inlet_y = torch.linspace(0.0, self.height, inlet_probe_count, device=device)
            inlet_x = torch.zeros(inlet_probe_count, dtype=torch.float32, device=device)
            inlet_t = torch.full(
                (inlet_probe_count,),
                float(self.t_end),
                dtype=torch.float32,
                device=device,
            )
            inlet_prediction = model.forward(inlet_x, inlet_y, inlet_t)
            inlet_u, inlet_v, _ = self._split_prediction(inlet_prediction)
            inlet_target_u = self._inflow_profile_torch(inlet_y)

            inlet_u_rmse = float(
                torch.sqrt(torch.mean((inlet_u - inlet_target_u).square())).cpu()
            )
            inlet_v_rmse = float(torch.sqrt(torch.mean(inlet_v.square())).cpu())

        mesh_vertex_coords = torch.tensor(
            [(float(v.point[0]), float(v.point[1])) for v in mesh.vertices],
            dtype=torch.float32,
            device=device,
        )
        unique_times_list = [item["time"] for item in time_slice_metrics]
        snapshot_data = []
        with torch.no_grad():
            for time_value in unique_times_list:
                time_column = torch.full(
                    (len(mesh_vertex_coords), 1),
                    float(time_value),
                    dtype=torch.float32,
                    device=device,
                )
                prediction = model.forward(torch.cat((mesh_vertex_coords, time_column), dim=1))
                u_pred = prediction[:, 0]
                v_pred = prediction[:, 1]
                vel_mag = torch.sqrt(u_pred.square() + v_pred.square()).cpu().numpy()
                snapshot_data.append(
                    {
                        "time": float(time_value),
                        "velocity_magnitude": vel_mag.tolist(),
                    }
                )

        return {
            "overall_mean_pred_speed": float(pred_speed.mean().cpu()),
            "overall_mean_target_speed": float(target_speed.mean().cpu()),
            "overall_velocity_rmse": float(
                torch.sqrt(
                    torch.mean((pred_u - target_u).square() + (pred_v - target_v).square())
                ).cpu()
            ),
            "inlet_u_rmse_t_end": inlet_u_rmse,
            "inlet_v_rmse_t_end": inlet_v_rmse,
            "time_slice_metrics": time_slice_metrics,
            "predicted_velocity_snapshots": snapshot_data,
        }
