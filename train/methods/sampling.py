"""
Shared sampling helpers for collocation-point methods.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch
from config import RANDOM_CONFIG


def _log_sampling_stats(
    *,
    method_name: str | None,
    iteration: int | None,
    warn_label: str,
    requested: int,
    accepted: int,
    generated: int,
):
    if not RANDOM_CONFIG.get("log_sampling_stats", False):
        return
    if not method_name:
        return

    try:
        from paths import method_reports_dir

        out_dir = method_reports_dir(method_name)
        acceptance_rate = accepted / max(generated, 1)
        iteration_value = iteration if iteration is not None else "n/a"
        with open(f"{out_dir}/sampling_stats.txt", "a") as f:
            f.write(
                f"iteration={iteration_value}, label={warn_label}, "
                f"requested={requested}, accepted={accepted}, generated={generated}, "
                f"acceptance_rate={acceptance_rate:.4f}\n"
            )
    except Exception:
        pass


def filter_points_in_domain(
    mesh: Any, candidate_points: np.ndarray, limit: int | None = None
) -> np.ndarray:
    """Keep only candidate points that lie inside the mesh domain.

    Supports both 2D (N,2) and 3D (N,3) candidate arrays.
    """
    candidates = np.asarray(candidate_points, dtype=float)
    if candidates.ndim != 2 or candidates.shape[1] not in (2, 3):
        raise ValueError("candidate_points must have shape (N, 2) or (N, 3)")
    ndim = candidates.shape[1]
    accepted: list[tuple] = []

    for row in candidates:
        try:
            if ndim == 3:
                pt = mesh(float(row[0]), float(row[1]), float(row[2]))
            else:
                pt = mesh(float(row[0]), float(row[1]))
            if pt.nr != -1:
                accepted.append(tuple(row))
                if limit is not None and len(accepted) >= limit:
                    break
        except Exception:
            continue

    if not accepted:
        return np.empty((0, ndim), dtype=float)
    return np.asarray(accepted, dtype=float)


def sample_points_in_domain(
    mesh: Any,
    num_points: int,
    batch_generator: Callable[[int], np.ndarray],
    *,
    batch_size: int | None = None,
    max_batches: int = 50,
    warn_label: str = "points",
    method_name: str | None = None,
    iteration: int | None = None,
) -> np.ndarray:
    """Use batched rejection sampling to generate interior points."""
    if num_points <= 0:
        raise ValueError("num_points must be positive")

    batch_size = batch_size or max(256, num_points * 4)
    accepted_batches: list[np.ndarray] = []
    accepted_count = 0
    generated_count = 0

    for _ in range(max_batches):
        candidates = np.asarray(batch_generator(batch_size), dtype=float)
        if candidates.ndim != 2 or candidates.shape[1] not in (2, 3):
            raise ValueError("batch_generator must return an array of shape (N, 2) or (N, 3)")
        generated_count += len(candidates)

        accepted = filter_points_in_domain(
            mesh, candidates, limit=num_points - accepted_count
        )
        if accepted.size:
            accepted_batches.append(accepted)
            accepted_count += len(accepted)
            if accepted_count >= num_points:
                break

    if not accepted_batches:
        raise ValueError(f"Could not generate any valid {warn_label} in the domain")

    points = np.vstack(accepted_batches)[:num_points]
    if len(points) < num_points:
        print(f"Warning: Generated {len(points)}/{num_points} {warn_label}")

    _log_sampling_stats(
        method_name=method_name,
        iteration=iteration,
        warn_label=warn_label,
        requested=num_points,
        accepted=len(points),
        generated=generated_count,
    )

    return points


def points_to_tensors(
    points: np.ndarray,
) -> tuple[torch.Tensor, ...]:
    """Convert an (N, 2) or (N, 3) point array to float tensors."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] not in (2, 3) or len(points) == 0:
        raise ValueError("points must be a non-empty array of shape (N, 2) or (N, 3)")
    tensors = tuple(
        torch.tensor(points[:, i], dtype=torch.float32)
        for i in range(points.shape[1])
    )
    return tensors


def sample_uniform_points_in_triangle(
    triangle: np.ndarray, count: int, rng: np.random.RandomState
) -> np.ndarray:
    """Sample points uniformly inside a triangle.

    Args:
        triangle: Array of shape (3, 2) with triangle vertices
        count: Number of interior points to sample
        rng: Random number generator

    Returns:
        Array of shape (count, 2)
    """
    triangle = np.asarray(triangle, dtype=float)
    if triangle.shape != (3, 2):
        raise ValueError("triangle must have shape (3, 2)")
    if count <= 0:
        return np.empty((0, 2), dtype=float)

    uv = rng.uniform(size=(count, 2))
    reflect_mask = np.sum(uv, axis=1) > 1.0
    uv[reflect_mask] = 1.0 - uv[reflect_mask]

    vertex_a = triangle[0]
    edge_ab = triangle[1] - vertex_a
    edge_ac = triangle[2] - vertex_a
    return vertex_a + uv[:, :1] * edge_ab + uv[:, 1:] * edge_ac


def sample_uniform_points_in_tetrahedron(
    tet: np.ndarray, count: int, rng: np.random.RandomState
) -> np.ndarray:
    """Sample points uniformly inside a tetrahedron using barycentric coordinates.

    Args:
        tet: Array of shape (4, 3) with tetrahedron vertices
        count: Number of interior points to sample
        rng: Random number generator

    Returns:
        Array of shape (count, 3)
    """
    tet = np.asarray(tet, dtype=float)
    if tet.shape != (4, 3):
        raise ValueError("tet must have shape (4, 3)")
    if count <= 0:
        return np.empty((0, 3), dtype=float)

    # Generate uniform barycentric coordinates via exponential distribution trick
    # (equivalent to uniform on tetrahedron)
    raw = -np.log(rng.uniform(1e-10, 1.0, size=(count, 4)))
    bary = raw / raw.sum(axis=1, keepdims=True)  # (count, 4), sums to 1
    return bary @ tet  # (count, 3)
