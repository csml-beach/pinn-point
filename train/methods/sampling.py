"""
Shared sampling helpers for collocation-point methods.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch


def filter_points_in_domain(
    mesh: Any, candidate_points: np.ndarray, limit: int | None = None
) -> np.ndarray:
    """Keep only candidate points that lie inside the mesh domain."""
    accepted: list[tuple[float, float]] = []

    for x_coord, y_coord in np.asarray(candidate_points, dtype=float):
        try:
            if mesh(float(x_coord), float(y_coord)).nr != -1:
                accepted.append((float(x_coord), float(y_coord)))
                if limit is not None and len(accepted) >= limit:
                    break
        except Exception:
            continue

    if not accepted:
        return np.empty((0, 2), dtype=float)
    return np.asarray(accepted, dtype=float)


def sample_points_in_domain(
    mesh: Any,
    num_points: int,
    batch_generator: Callable[[int], np.ndarray],
    *,
    batch_size: int | None = None,
    max_batches: int = 50,
    warn_label: str = "points",
) -> np.ndarray:
    """Use batched rejection sampling to generate interior points."""
    if num_points <= 0:
        raise ValueError("num_points must be positive")

    batch_size = batch_size or max(256, num_points * 4)
    accepted_batches: list[np.ndarray] = []
    accepted_count = 0

    for _ in range(max_batches):
        candidates = np.asarray(batch_generator(batch_size), dtype=float)
        if candidates.ndim != 2 or candidates.shape[1] != 2:
            raise ValueError("batch_generator must return an array of shape (N, 2)")

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

    return points


def points_to_tensors(points: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert an (N, 2) point array to float tensors."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) == 0:
        raise ValueError("points must be a non-empty array of shape (N, 2)")
    return (
        torch.tensor(points[:, 0], dtype=torch.float32),
        torch.tensor(points[:, 1], dtype=torch.float32),
    )
