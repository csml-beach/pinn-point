#!/usr/bin/env python3
"""Generate Figure A (3D problem view) and Figure B (budget convergence)."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

matplotlib.use("Agg")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--run-root",
        default=(
            "/Users/arash/Documents/GitHub/pinn-point/outputs/"
            "m3-cpu-xl-maxwell3d-budget-sweep-2026-05-05"
        ),
    )
    p.add_argument(
        "--out-dir",
        default="/Users/arash/Documents/GitHub/pinn-point/artifacts/figures/maxwell3d",
    )
    return p.parse_args()


def _load_histories(run_root: Path, field: str, methods: list[str]) -> dict[str, dict[str, list[np.ndarray]]]:
    """Return budget -> method -> [series per seed]."""
    out: dict[str, dict[str, list[np.ndarray]]] = {}
    for budget_dir in sorted([p for p in run_root.iterdir() if p.is_dir() and p.name.startswith("e")]):
        budget_name = budget_dir.name
        out.setdefault(budget_name, {m: [] for m in methods})
        for csv_path in budget_dir.glob("*/reports/all_methods_histories.csv"):
            per_method: dict[str, list[float]] = {m: [] for m in methods}
            with csv_path.open() as f:
                reader = csv.DictReader(f)
                for row in reader:
                    method = (row.get("method") or "").strip()
                    if method not in per_method:
                        continue
                    raw = row.get(field, "")
                    try:
                        val = float(raw)
                    except Exception:
                        val = float("nan")
                    per_method[method].append(val)
            for m in methods:
                if per_method[m]:
                    out[budget_name][m].append(np.asarray(per_method[m], dtype=float))
    return out


def _mean_std_with_padding(series_list: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    if not series_list:
        return np.array([]), np.array([])
    max_len = max(len(s) for s in series_list)
    arr = np.full((len(series_list), max_len), np.nan, dtype=float)
    for i, s in enumerate(series_list):
        arr[i, : len(s)] = s
    return np.nanmean(arr, axis=0), np.nanstd(arr, axis=0)


def _plot_budget_curves(run_root: Path, out_dir: Path) -> Path:
    methods = ["adaptive_power_tempered", "halton"]
    colors = {
        "e100_i12": "#1f77b4",
        "e150_i8": "#ff7f0e",
        "e200_i6": "#2ca02c",
        "e300_i4": "#d62728",
    }
    linestyles = {"adaptive_power_tempered": "-", "halton": "--"}

    error_data = _load_histories(run_root, "total_error", methods)
    resid_data = _load_histories(run_root, "fixed_rms_residual", methods)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, data, ylabel, title in [
        (axes[0], error_data, "Total Error", "Error vs Iteration"),
        (axes[1], resid_data, "Fixed RMS Residual", "Residual vs Iteration"),
    ]:
        for budget_name in sorted(data):
            for method in methods:
                mean, std = _mean_std_with_padding(data[budget_name][method])
                if mean.size == 0:
                    continue
                x = np.arange(1, mean.size + 1, dtype=float)
                label = f"{budget_name} | {'APT' if method == 'adaptive_power_tempered' else 'Halton'}"
                ax.plot(
                    x,
                    mean,
                    color=colors.get(budget_name, "black"),
                    linestyle=linestyles[method],
                    linewidth=2.0,
                    label=label,
                )
                ax.fill_between(x, mean - std, mean + std, color=colors.get(budget_name, "black"), alpha=0.10)
        ax.set_xlabel("Iteration")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, ncol=1)
        ax.set_xlim(left=1)
    axes[1].set_yscale("log")

    plt.tight_layout()
    out_path = out_dir / "figure_b_budget_convergence.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_geometry_and_slices(out_dir: Path) -> Path:
    # Geometry parameters (must match defaults in maxwell_coil_core_3d problem)
    box_half_xy = 0.04
    z_min, z_max = -0.03, 0.06
    core_r = 0.010
    coil_r_in, coil_r_out = 0.012, 0.020
    coil_z_min, coil_z_max = -0.005, 0.035

    # Build real FEM reference once and sample |A| on slices.
    repo_root = Path(__file__).resolve().parents[1]
    train_dir = repo_root / "train"
    if str(train_dir) not in sys.path:
        sys.path.insert(0, str(train_dir))
    from problems import get_problem  # type: ignore

    problem = get_problem("maxwell_coil_core_3d")
    ref_mesh, ref_solution = problem.create_reference_solution(mesh_size_factor=0.02)

    n = 200
    xx = np.linspace(-box_half_xy, box_half_xy, n)
    yy = np.linspace(-box_half_xy, box_half_xy, n)
    X, Y = np.meshgrid(xx, yy)

    def sample_slice(z_level: float) -> np.ndarray:
        field = np.full_like(X, np.nan, dtype=float)
        for i in range(n):
            for j in range(n):
                x = float(X[i, j])
                y = float(Y[i, j])
                try:
                    mip = ref_mesh(x, y, float(z_level))
                    v = ref_solution(mip)
                    field[i, j] = float(math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]))
                except Exception:
                    continue
        return field

    fields_xy = [sample_slice(zv) for zv in [0.0, 0.015, 0.03]]

    # XZ slice at y=0
    zz = np.linspace(z_min, z_max, n)
    XZ_X, XZ_Z = np.meshgrid(xx, zz)
    field_xz = np.full_like(XZ_X, np.nan, dtype=float)
    for i in range(n):
        for j in range(n):
            x = float(XZ_X[i, j])
            z = float(XZ_Z[i, j])
            try:
                mip = ref_mesh(x, 0.0, z)
                v = ref_solution(mip)
                field_xz[i, j] = float(math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]))
            except Exception:
                continue

    fig = plt.figure(figsize=(14.5, 5.2))

    # Panel 1: true 3D geometry wireframe/solids
    ax1 = fig.add_subplot(1, 3, 1, projection="3d")

    # box edges
    corners = np.array(
        [
            [-box_half_xy, -box_half_xy, z_min],
            [ box_half_xy, -box_half_xy, z_min],
            [ box_half_xy,  box_half_xy, z_min],
            [-box_half_xy,  box_half_xy, z_min],
            [-box_half_xy, -box_half_xy, z_max],
            [ box_half_xy, -box_half_xy, z_max],
            [ box_half_xy,  box_half_xy, z_max],
            [-box_half_xy,  box_half_xy, z_max],
        ]
    )
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    for a,b in edges:
        ax1.plot(
            [corners[a,0], corners[b,0]],
            [corners[a,1], corners[b,1]],
            [corners[a,2], corners[b,2]],
            color="black",
            linewidth=1.2,
            alpha=0.75,
        )

    # core + coil cylinder surfaces
    theta = np.linspace(0.0, 2.0 * math.pi, 80)
    z_core = np.linspace(-0.02, 0.05, 2)
    T_core, Z_core = np.meshgrid(theta, z_core)
    X_core = core_r * np.cos(T_core)
    Y_core = core_r * np.sin(T_core)
    ax1.plot_surface(X_core, Y_core, Z_core, alpha=0.35, color="#2ca02c", linewidth=0)

    z_coil = np.linspace(coil_z_min, coil_z_max, 2)
    T_coil, Z_coil = np.meshgrid(theta, z_coil)
    for rr, alpha in [(coil_r_in, 0.18), (coil_r_out, 0.22)]:
        X_coil = rr * np.cos(T_coil)
        Y_coil = rr * np.sin(T_coil)
        ax1.plot_surface(X_coil, Y_coil, Z_coil, alpha=alpha, color="#d62728", linewidth=0)

    ax1.set_title("Figure A1: 3D Geometry (Air Box + Core + Coil)")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("z")
    ax1.view_init(elev=20, azim=-58)
    ax1.set_box_aspect((1, 1, 1.2))
    ax1.set_xlim(-box_half_xy, box_half_xy)
    ax1.set_ylim(-box_half_xy, box_half_xy)
    ax1.set_zlim(z_min, z_max)

    # Panel 2: real |A| on XZ slice (y=0)
    ax2 = fig.add_subplot(1, 3, 2)
    im = ax2.imshow(
        field_xz,
        extent=[-box_half_xy, box_half_xy, z_min, z_max],
        origin="lower",
        cmap="viridis",
        aspect="auto",
    )
    ax2.set_title("Figure A2: |A| on XZ Slice (y=0)")
    ax2.set_xlabel("x")
    ax2.set_ylabel("z")
    cbar = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label("|A|")

    # Panel 3: real |A| on XY slice (z=0.015)
    ax3 = fig.add_subplot(1, 3, 3)
    im2 = ax3.imshow(
        fields_xy[1],
        extent=[-box_half_xy, box_half_xy, -box_half_xy, box_half_xy],
        origin="lower",
        cmap="viridis",
    )
    ax3.set_aspect("equal")
    ax3.set_title("Figure A3: |A| on XY Slice (z=0.015)")
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    cbar2 = fig.colorbar(im2, ax=ax3, fraction=0.046, pad=0.04)
    cbar2.set_label("|A|")

    plt.tight_layout()
    out_path = out_dir / "figure_a_geometry_and_field_proxy.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    args = _parse_args()
    run_root = Path(args.run_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig_a = _plot_geometry_and_slices(out_dir)
    fig_b = _plot_budget_curves(run_root, out_dir)

    print(f"figure_a={fig_a}")
    print(f"figure_b={fig_b}")


if __name__ == "__main__":
    main()
