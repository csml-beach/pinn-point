#!/usr/bin/env python3
"""
Reference-resolution study for the Navier-Stokes channel-obstacle problem.

Compares FEM solutions across spatial and temporal resolutions on a common
space-time probe set, using the finest case as the internal reference.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass

import numpy as np
from scipy.stats import qmc

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from train.problems import get_problem


@dataclass
class CaseResult:
    label: str
    mesh_size: float
    dt: float
    probes_u: np.ndarray
    probes_v: np.ndarray
    section_fluxes: np.ndarray


def build_probe_points(problem, mesh, count: int, seed: int) -> np.ndarray:
    sampler = qmc.Halton(d=2, scramble=True, seed=seed)
    x_min, x_max, y_min, y_max = problem.get_domain_bounds()
    accepted = []
    batch_size = max(4 * count, 512)
    while len(accepted) < count:
        pts = sampler.random(batch_size)
        xs = x_min + (x_max - x_min) * pts[:, 0]
        ys = y_min + (y_max - y_min) * pts[:, 1]
        for x_val, y_val in zip(xs, ys):
            if problem._point_is_in_fluid(mesh, float(x_val), float(y_val)):
                accepted.append((float(x_val), float(y_val)))
                if len(accepted) >= count:
                    break
    return np.asarray(accepted, dtype=np.float64)


def run_case(problem, mesh_size: float, dt: float, snapshot_times: list[float], probe_xy: np.ndarray) -> CaseResult:
    mesh = problem.create_mesh(maxh=mesh_size)
    initial_state = problem._solve_initial_stokes_state(mesh)
    Xspace = initial_state["space"]
    gfu = initial_state["state"]
    velocity, pressure = gfu.components

    (u, p), (v, q) = Xspace.TnT()
    stokes_form = initial_state["stokes_form"]
    stiffness = initial_state["stiffness"]

    from ngsolve import BilinearForm, InnerProduct, Grad, dx

    # Rebuild mass+stokes operator exactly as in problem.solve_fem_time_series.
    mstar = BilinearForm(Xspace)
    mstar += InnerProduct(u, v) * dx + dt * stokes_form
    mstar.Assemble()
    inv = mstar.mat.Inverse(Xspace.FreeDofs())

    conv = BilinearForm(Xspace, nonassemble=True)
    conv += (Grad(u) * u) * v * dx

    sections = problem._build_flux_probe_sections(mesh)
    snapshot_targets = sorted({max(0.0, min(problem.t_end, float(t))) for t in snapshot_times})
    probe_u = []
    probe_v = []
    fluxes = []

    def capture():
        current_u = []
        current_v = []
        for x_val, y_val in probe_xy:
            vel = velocity(mesh(float(x_val), float(y_val)))
            current_u.append(float(vel[0]))
            current_v.append(float(vel[1]))
        probe_u.append(current_u)
        probe_v.append(current_v)
        flux_vec = [float(item["flux"]) for item in problem._evaluate_section_fluxes(mesh, velocity, sections)]
        fluxes.append(flux_vec)

    capture()
    t = 0.0
    next_target_index = 0
    while next_target_index < len(snapshot_targets) and snapshot_targets[next_target_index] <= 1e-12:
        next_target_index += 1

    while t < problem.t_end - 0.5 * dt:
        res = initial_state["state"].vec.CreateVector()
        res.data = (stiffness.mat + conv.mat) * gfu.vec
        gfu.vec.data -= dt * inv * res
        t += dt

        while next_target_index < len(snapshot_targets) and t >= snapshot_targets[next_target_index] - 0.5 * dt:
            capture()
            next_target_index += 1

    if len(probe_u) < len(snapshot_targets):
        capture()

    return CaseResult(
        label=f"mesh{mesh_size:g}_dt{dt:g}",
        mesh_size=float(mesh_size),
        dt=float(dt),
        probes_u=np.asarray(probe_u, dtype=np.float64),
        probes_v=np.asarray(probe_v, dtype=np.float64),
        section_fluxes=np.asarray(fluxes, dtype=np.float64),
    )


def relative_velocity_error(case: CaseResult, ref: CaseResult) -> float:
    du = case.probes_u - ref.probes_u
    dv = case.probes_v - ref.probes_v
    num = np.mean(du * du + dv * dv)
    den = np.mean(ref.probes_u * ref.probes_u + ref.probes_v * ref.probes_v)
    return math.sqrt(num / max(den, 1e-12))


def relative_flux_error(case: CaseResult, ref: CaseResult) -> float:
    diff = case.section_fluxes - ref.section_fluxes
    num = np.mean(diff * diff)
    den = np.mean(ref.section_fluxes * ref.section_fluxes)
    return math.sqrt(num / max(den, 1e-12))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", default="navier_stokes_channel_obstacle")
    parser.add_argument("--probe-count", type=int, default=1024)
    parser.add_argument("--probe-seed", type=int, default=42)
    parser.add_argument("--mesh-sizes", default="0.08,0.05,0.035,0.025")
    parser.add_argument("--dts", default="0.008,0.004,0.002,0.001")
    parser.add_argument("--snapshot-times", default="")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    problem = get_problem(args.problem)

    if args.snapshot_times.strip():
        snapshot_times = [float(x) for x in args.snapshot_times.split(",") if x.strip()]
    else:
        snapshot_times = np.linspace(0.0, problem.t_end, problem.reference_time_slices).tolist()

    mesh_sizes = [float(x) for x in args.mesh_sizes.split(",") if x.strip()]
    dts = [float(x) for x in args.dts.split(",") if x.strip()]
    fine_mesh = min(mesh_sizes)
    fine_dt = min(dts)

    probe_mesh = problem.create_mesh(maxh=fine_mesh)
    probe_xy = build_probe_points(problem, probe_mesh, args.probe_count, args.probe_seed)

    spatial_cases = []
    for mesh_size in mesh_sizes:
        spatial_cases.append(run_case(problem, mesh_size=mesh_size, dt=fine_dt, snapshot_times=snapshot_times, probe_xy=probe_xy))

    temporal_cases = []
    for dt in dts:
        temporal_cases.append(run_case(problem, mesh_size=fine_mesh, dt=dt, snapshot_times=snapshot_times, probe_xy=probe_xy))

    spatial_ref = next(case for case in spatial_cases if case.mesh_size == fine_mesh)
    temporal_ref = next(case for case in temporal_cases if case.dt == fine_dt)

    spatial_rows = []
    for case in spatial_cases:
        spatial_rows.append({
            "mesh_size": case.mesh_size,
            "dt": case.dt,
            "relative_velocity_error": relative_velocity_error(case, spatial_ref),
            "relative_flux_error": relative_flux_error(case, spatial_ref),
        })

    temporal_rows = []
    for case in temporal_cases:
        temporal_rows.append({
            "mesh_size": case.mesh_size,
            "dt": case.dt,
            "relative_velocity_error": relative_velocity_error(case, temporal_ref),
            "relative_flux_error": relative_flux_error(case, temporal_ref),
        })

    with open(os.path.join(args.output_dir, "spatial_convergence.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(spatial_rows[0].keys()))
        writer.writeheader()
        writer.writerows(spatial_rows)

    with open(os.path.join(args.output_dir, "temporal_convergence.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(temporal_rows[0].keys()))
        writer.writeheader()
        writer.writerows(temporal_rows)

    summary = {
        "problem": args.problem,
        "probe_count": args.probe_count,
        "probe_seed": args.probe_seed,
        "snapshot_times": snapshot_times,
        "spatial_reference": {"mesh_size": fine_mesh, "dt": fine_dt},
        "temporal_reference": {"mesh_size": fine_mesh, "dt": fine_dt},
        "spatial_convergence": spatial_rows,
        "temporal_convergence": temporal_rows,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
