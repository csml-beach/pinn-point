#!/usr/bin/env python3
"""Generate an illustrative Netgen disk residual-scaffold diagram.

The figure is intended for the Method section.  Netgen generates the disk mesh,
while a synthetic scalar residual energy density r(x, y)^2 supplies the local
values.  The scoring, smoothing, tempering, quota allocation, and point
sampling then follow the method formulas:

    raw element scores s_i^(k) -> smoothed scores \tilde{s}_i^(k)
    -> power-tempered element probabilities p_i^(k).

Run with the Netgen pyenv:

    /Users/arash/.pyenv/versions/netgen/bin/python scripts/plot_toy_residual_scaffold.py
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize


@dataclass(frozen=True)
class DiskMesh:
    points: np.ndarray
    triangles: np.ndarray
    centroids: np.ndarray
    areas: np.ndarray
    neighbors: list[list[int]]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        default="/Users/arash/Documents/GitHub/pinn-point/paper/figures",
        help="Directory where PDF, SVG, and PNG outputs are written.",
    )
    parser.add_argument("--basename", default="toy_circular_residual_scaffold")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-points", type=int, default=220)
    parser.add_argument("--maxh", type=float, default=0.24)
    parser.add_argument("--score-exponent", type=float, default=0.5)
    parser.add_argument("--coverage-exponent", type=float, default=0.5)
    return parser.parse_args()


def _build_disk_mesh(maxh: float = 0.24) -> DiskMesh:
    from netgen.geom2d import SplineGeometry
    from ngsolve import Mesh, VOL

    geometry = SplineGeometry()
    geometry.AddCircle((0.0, 0.0), 1.0, bc="outer")
    mesh = Mesh(geometry.GenerateMesh(maxh=float(maxh)))

    point_to_idx: dict[tuple[float, float], int] = {}
    points: list[tuple[float, float]] = []
    triangles: list[tuple[int, int, int]] = []
    element_ids: list[int] = []

    def point_id(point: tuple[float, float]) -> int:
        key = (round(float(point[0]), 14), round(float(point[1]), 14))
        if key not in point_to_idx:
            point_to_idx[key] = len(points)
            points.append((float(point[0]), float(point[1])))
        return point_to_idx[key]

    for element in mesh.Elements(VOL):
        vertices = []
        for vertex in element.vertices:
            p = mesh[vertex].point
            vertices.append(point_id((float(p[0]), float(p[1]))))
        if len(vertices) == 3:
            triangles.append(tuple(vertices))
            element_ids.append(int(element.nr))

    pts = np.asarray(points, dtype=float)
    tris = np.asarray(triangles, dtype=int)
    polys = pts[tris]
    centroids = polys.mean(axis=1)
    areas = 0.5 * np.abs(
        (polys[:, 1, 0] - polys[:, 0, 0]) * (polys[:, 2, 1] - polys[:, 0, 1])
        - (polys[:, 1, 1] - polys[:, 0, 1]) * (polys[:, 2, 0] - polys[:, 0, 0])
    )

    id_to_local = {element_id: idx for idx, element_id in enumerate(element_ids)}
    neighbor_sets = [set() for _ in range(len(tris))]
    for edge in mesh.edges:
        adjacent = []
        for element in getattr(edge, "elements", ()):
            try:
                if element.VB() != VOL:
                    continue
                local_idx = id_to_local.get(int(element.nr))
                if local_idx is not None:
                    adjacent.append(local_idx)
            except Exception:
                continue
        for i in adjacent:
            for j in adjacent:
                if i != j:
                    neighbor_sets[i].add(j)
    neighbors = [sorted(nbrs) for nbrs in neighbor_sets]

    return DiskMesh(pts, tris, centroids, areas, neighbors)


def _residual_energy_density(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Synthetic scalar residual energy r(x,y)^2 on the disk."""

    primary = 1.20 * np.exp(-(((x - 0.34) / 0.24) ** 2 + ((y - 0.13) / 0.18) ** 2))
    secondary = 0.72 * np.exp(-(((x + 0.40) / 0.28) ** 2 + ((y + 0.30) / 0.22) ** 2))
    ridge = 0.20 * np.exp(-((y + 0.07 - 0.35 * x) / 0.16) ** 2)
    ripple_window = np.exp(-(((x + 0.03) / 0.82) ** 2 + ((y - 0.03) / 0.78) ** 2))
    ripple = 0.08 * ripple_window * (1.0 + np.sin(16.0 * x - 10.0 * y)) ** 2
    return np.clip(0.02 + primary + secondary + ridge + ripple, 0.0, None)


def _raw_scores(mesh: DiskMesh, chi: float = 0.5) -> np.ndarray:
    bary = np.asarray(
        [
            [2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0],
            [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
            [1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0],
        ],
        dtype=float,
    )
    weights = np.full(3, 1.0 / 3.0, dtype=float)
    vertices = mesh.points[mesh.triangles]
    quad_points = np.einsum("qb,ebc->eqc", bary, vertices)
    values = _residual_energy_density(quad_points[:, :, 0], quad_points[:, :, 1])
    return np.power(mesh.areas, float(chi)) * np.sum(weights[None, :] * values, axis=1)


def _smooth_scores(mesh: DiskMesh, scores: np.ndarray, lam: float = 0.42) -> np.ndarray:
    smoothed = scores.copy()
    for idx, nbrs in enumerate(mesh.neighbors):
        if not nbrs:
            continue
        weights = mesh.areas[nbrs]
        denom = float(np.sum(weights))
        if denom <= 0.0:
            continue
        neighbor_mean = float(np.dot(weights, scores[nbrs]) / denom)
        smoothed[idx] = (1.0 - lam) * scores[idx] + lam * neighbor_mean
    return smoothed


def _rank_normalize(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(values, dtype=float)
    if len(values) == 1:
        ranks[0] = 1.0
        return ranks
    ranks[order] = np.arange(len(values), dtype=float) / float(len(values) - 1)
    return ranks


def _power_tempered_probs(
    mesh: DiskMesh,
    scores: np.ndarray,
    *,
    beta_min: float = 1.5,
    beta_max: float = 7.0,
    gamma: float = 0.5,
) -> tuple[np.ndarray, float]:
    positive = np.clip(scores, 0.0, None)
    score_sum = float(np.sum(positive))
    if score_sum <= 0.0:
        concentration = 0.0
    else:
        rho = positive / score_sum
        entropy = -float(np.sum(rho * np.log(np.clip(rho, 1.0e-12, None))))
        entropy /= math.log(len(rho))
        concentration = float(np.clip(1.0 - entropy, 0.0, 1.0))

    beta = beta_min + (beta_max - beta_min) * concentration
    ranks = _rank_normalize(scores)
    weights = (mesh.areas**float(gamma)) * np.exp(beta * (ranks - np.max(ranks)))
    probs = weights / np.sum(weights)
    return probs, beta


def _largest_remainder_counts(probs: np.ndarray, num_points: int) -> np.ndarray:
    expected = probs * int(num_points)
    counts = np.floor(expected).astype(int)
    remaining = int(num_points) - int(np.sum(counts))
    if remaining > 0:
        order = np.argsort(-(expected - counts), kind="mergesort")
        counts[order[:remaining]] += 1
    return counts


def _sample_points(mesh: DiskMesh, probs: np.ndarray, num_points: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    counts = _largest_remainder_counts(probs, num_points)
    samples = []
    for tri, count in zip(mesh.triangles, counts):
        if count <= 0:
            continue
        verts = mesh.points[tri]
        r1 = np.sqrt(rng.random(count))
        r2 = rng.random(count)
        pts = (
            (1.0 - r1)[:, None] * verts[0]
            + (r1 * (1.0 - r2))[:, None] * verts[1]
            + (r1 * r2)[:, None] * verts[2]
        )
        samples.append(pts)
    if not samples:
        return np.empty((0, 2), dtype=float)
    out = np.vstack(samples)
    rng.shuffle(out)
    return out


def _cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "residual_scaffold",
        [
            "#f7f8fb",
            "#d4e7f7",
            "#68a9cf",
            "#f6c064",
            "#b92f2b",
        ],
    )


def _draw_panel(
    ax: plt.Axes,
    mesh: DiskMesh,
    values: np.ndarray,
    *,
    title: str,
    subtitle: str,
    cmap: LinearSegmentedColormap,
    selected: int | None = None,
    samples: np.ndarray | None = None,
) -> PolyCollection:
    polygons = mesh.points[mesh.triangles]
    norm = Normalize(vmin=float(np.min(values)), vmax=float(np.max(values)))
    collection = PolyCollection(
        polygons,
        array=values,
        cmap=cmap,
        norm=norm,
        edgecolors="#222222",
        linewidths=0.42,
        joinstyle="round",
    )
    ax.add_collection(collection)

    theta = np.linspace(0.0, 2.0 * np.pi, 400)
    ax.plot(np.cos(theta), np.sin(theta), color="#111111", linewidth=1.5)

    if selected is not None:
        selected_poly = mesh.points[mesh.triangles[selected]]
        ax.add_patch(
            plt.Polygon(
                selected_poly,
                closed=True,
                fill=False,
                edgecolor="#2634b8",
                linewidth=2.3,
                joinstyle="round",
            )
        )
        cx, cy = mesh.centroids[selected]
        ax.text(
            cx + 0.025,
            cy + 0.015,
            r"$E_i^{(k)}$",
            color="#2634b8",
            fontsize=14,
            fontweight="bold",
            ha="center",
            va="center",
        )

    if samples is not None and len(samples):
        ax.scatter(samples[:, 0], samples[:, 1], s=5, c="#111111", alpha=0.78, linewidths=0)

    ax.set_aspect("equal")
    ax.set_xlim(-1.06, 1.06)
    ax.set_ylim(-1.06, 1.06)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(
        0.5,
        1.095,
        title,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=12.5,
        fontweight="bold",
        color="#111111",
    )
    ax.text(
        0.5,
        1.045,
        subtitle,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=9.5,
        color="#333333",
    )
    for spine in ax.spines.values():
        spine.set_visible(False)
    return collection


def _save_figure(out_dir: Path, basename: str, mesh: DiskMesh, raw: np.ndarray, smoothed: np.ndarray, probs: np.ndarray, beta: float, samples: np.ndarray) -> list[Path]:
    cmap = _cmap()
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 5.25), constrained_layout=True)
    fig.patch.set_facecolor("white")

    _draw_panel(
        axes[0],
        mesh,
        raw,
        title=r"Raw element score",
        subtitle=r"local quadrature estimate $s_i^{(k)}$",
        cmap=cmap,
    )
    _draw_panel(
        axes[1],
        mesh,
        smoothed,
        title=r"Adjacency-smoothed score",
        subtitle=r"measure-weighted graph average $\tilde{s}_i^{(k)}$",
        cmap=cmap,
    )
    prob_collection = _draw_panel(
        axes[2],
        mesh,
        probs / np.max(probs),
        title=r"Power-tempered allocation",
        subtitle=rf"$p_i^{{(k)}}$ and sampled $\mathcal{{X}}_r^{{(k+1)}}$; $\beta={beta:.2f}$",
        cmap=cmap,
        samples=samples,
    )

    for left_ax, right_ax in ((axes[0], axes[1]), (axes[1], axes[2])):
        left_ax.annotate(
            "",
            xy=(1.08, 0.50),
            xytext=(1.00, 0.50),
            xycoords="axes fraction",
            textcoords="axes fraction",
            arrowprops=dict(arrowstyle="->", color="#2634b8", linewidth=1.8),
            annotation_clip=False,
        )
        right_ax.annotate(
            "",
            xy=(-0.08, 0.50),
            xytext=(-0.00, 0.50),
            xycoords="axes fraction",
            textcoords="axes fraction",
            arrowprops=dict(arrowstyle="<-", color="#2634b8", linewidth=0.0),
            annotation_clip=False,
        )

    cbar = fig.colorbar(
        prob_collection,
        ax=axes,
        location="bottom",
        fraction=0.055,
        pad=0.035,
        aspect=55,
    )
    cbar.set_label("relative magnitude", fontsize=9.5)
    cbar.ax.tick_params(labelsize=8)

    out_paths = []
    for ext, kwargs in {
        "pdf": {},
        "svg": {},
        "png": {"dpi": 260},
    }.items():
        path = out_dir / f"{basename}.{ext}"
        fig.savefig(path, bbox_inches="tight", facecolor="white", **kwargs)
        out_paths.append(path)
    plt.close(fig)
    return out_paths


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mesh = _build_disk_mesh(maxh=args.maxh)
    raw = _raw_scores(mesh, chi=args.score_exponent)
    smoothed = _smooth_scores(mesh, raw)
    probs, beta = _power_tempered_probs(
        mesh,
        smoothed,
        gamma=args.coverage_exponent,
    )
    samples = _sample_points(mesh, probs, num_points=args.num_points, seed=args.seed + 100)
    out_paths = _save_figure(out_dir, args.basename, mesh, raw, smoothed, probs, beta, samples)
    for path in out_paths:
        print(path)


if __name__ == "__main__":
    main()
