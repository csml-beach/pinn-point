"""
Adaptive residual-guided collocation sampling.

This method uses a refined mesh only as an indicator scaffold. The mesh is scored
with residual estimates, optionally refined, and then a fixed collocation budget
is sampled from element interiors rather than taken directly from mesh vertices.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np
import torch
from ngsolve import VOL

from .base import TrainingMethod
from .sampling import points_to_tensors, sample_uniform_points_in_triangle


class AdaptiveMethod(TrainingMethod):
    """Residual-guided adaptive interior sampling on a mesh scaffold."""

    name = "adaptive"
    description = "Residual-guided interior sampling on a refined mesh scaffold"

    _GLOBAL_COVERAGE_FRACTION = 0.2
    _ADAPTIVE_MASS_TARGET = 0.8
    _MIN_SELECTED_TRIANGLES = 32
    _QUADRATURE_BARYCENTRIC = np.array(
        [
            [2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0],
            [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
            [1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0],
        ],
        dtype=float,
    )

    def __init__(
        self,
        refinement_threshold: float = 0.5,
        seed: int | None = None,
        global_coverage_fraction: float = _GLOBAL_COVERAGE_FRACTION,
        score_exponent: float = 0.5,
        refine_period: int = 2,
        smoothing_lambda: float = 0.4,
        area_exponent: float = 1.0,
    ):
        self.refinement_threshold = float(refinement_threshold)
        self.seed = seed
        self.global_coverage_fraction = float(global_coverage_fraction)
        self.score_exponent = float(score_exponent)
        self.refine_period = max(int(refine_period), 1)
        self.smoothing_lambda = float(smoothing_lambda)
        self.area_exponent = float(area_exponent)
        self._rng = np.random.RandomState(seed)
        self._sampling_state: dict[str, np.ndarray] | None = None
        self._last_refinement_stats: dict[str, float | int | bool] = {}

    def _extract_triangle_data(self, mesh: Any) -> tuple[np.ndarray, np.ndarray]:
        triangles: list[np.ndarray] = []
        areas: list[float] = []
        element_ids: list[int] = []

        for el in mesh.Elements(VOL):
            vertices = []
            for vertex in el.vertices:
                point = mesh[vertex].point
                if hasattr(point, "x"):
                    vertices.append((float(point.x), float(point.y)))
                else:
                    vertices.append((float(point[0]), float(point[1])))

            if len(vertices) != 3:
                continue

            triangle = np.asarray(vertices, dtype=float)
            edge_ab = triangle[1] - triangle[0]
            edge_ac = triangle[2] - triangle[0]
            area = 0.5 * abs(edge_ab[0] * edge_ac[1] - edge_ab[1] * edge_ac[0])
            if area <= 0.0:
                continue

            triangles.append(triangle)
            areas.append(float(area))
            element_ids.append(int(el.nr))

        if not triangles:
            raise ValueError("mesh does not contain any positive-area triangles")

        return (
            np.asarray(triangles, dtype=float),
            np.asarray(areas, dtype=float),
            np.asarray(element_ids, dtype=int),
        )

    def _build_triangle_neighbors(
        self, mesh: Any, element_ids: np.ndarray
    ) -> list[list[int]]:
        id_to_local = {int(global_id): idx for idx, global_id in enumerate(element_ids)}
        neighbors: list[set[int]] = [set() for _ in range(len(element_ids))]

        for edge in mesh.edges:
            adjacent_elements = []
            for element_id in getattr(edge, "elements", ()):
                try:
                    if element_id.VB() != VOL:
                        continue
                    local_idx = id_to_local.get(int(element_id.nr))
                    if local_idx is not None:
                        adjacent_elements.append(local_idx)
                except Exception:
                    continue

            if len(adjacent_elements) < 2:
                continue

            for idx in adjacent_elements:
                for nbr in adjacent_elements:
                    if nbr != idx:
                        neighbors[idx].add(nbr)

        return [sorted(local_neighbors) for local_neighbors in neighbors]

    def _smooth_scores(
        self, scores: np.ndarray, neighbors: list[list[int]], areas: np.ndarray
    ) -> np.ndarray:
        lam = min(max(self.smoothing_lambda, 0.0), 1.0)
        if lam <= 0.0:
            return np.asarray(scores, dtype=float)

        scores = np.asarray(scores, dtype=float)
        smoothed = scores.copy()

        for idx, nbrs in enumerate(neighbors):
            if not nbrs:
                continue
            nbr_scores = scores[nbrs]
            nbr_areas = areas[nbrs]
            denom = float(np.sum(nbr_areas))
            if denom <= 0.0:
                continue
            neighbor_mean = float(np.dot(nbr_scores, nbr_areas) / denom)
            smoothed[idx] = (1.0 - lam) * scores[idx] + lam * neighbor_mean

        return smoothed

    def _evaluate_residual_scores(
        self, mesh: Any, model: Any
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        triangles, areas, element_ids = self._extract_triangle_data(mesh)

        bary = self._QUADRATURE_BARYCENTRIC
        quad_points = np.einsum("qb,ebc->eqc", bary, triangles)
        flat_points = quad_points.reshape(-1, 2)

        x = torch.tensor(flat_points[:, 0], dtype=torch.float32, device=model.mesh_x.device)
        y = torch.tensor(flat_points[:, 1], dtype=torch.float32, device=model.mesh_y.device)

        with torch.enable_grad():
            residual = model.PDE_residual(x, y)

        residual_sq = torch.square(residual).detach().cpu().numpy().reshape(
            len(triangles), len(bary)
        )
        pointwise_r2 = np.mean(residual_sq, axis=1)
        pointwise_r2 = np.nan_to_num(pointwise_r2, nan=0.0, posinf=0.0, neginf=0.0)
        area_factor = np.power(np.clip(areas, 1e-12, None), self.area_exponent)
        raw_scores = area_factor * np.clip(pointwise_r2, 0.0, None)
        neighbors = self._build_triangle_neighbors(mesh, element_ids)
        smoothed_scores = self._smooth_scores(raw_scores, neighbors, areas)
        total_residual = float(np.sum(raw_scores))
        return triangles, areas, raw_scores, smoothed_scores, total_residual

    def _set_sampling_distribution(
        self, triangles: np.ndarray, areas: np.ndarray, scores: np.ndarray
    ) -> None:
        scores = np.asarray(scores, dtype=float)
        areas = np.asarray(areas, dtype=float)

        clipped_scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        clipped_scores = np.clip(clipped_scores, 0.0, None)
        area_probs = areas / np.sum(areas)

        if np.sum(clipped_scores) <= 0.0:
            adaptive_probs = area_probs
        else:
            weighted_scores = np.power(clipped_scores + 1e-12, self.score_exponent)
            adaptive_probs = weighted_scores / np.sum(weighted_scores)

        self._sampling_state = {
            "triangles": triangles,
            "areas": areas,
            "adaptive_probs": adaptive_probs,
            "area_probs": area_probs,
        }

    def _sample_from_distribution(self, num_points: int) -> np.ndarray:
        if self._sampling_state is None:
            raise RuntimeError("sampling distribution has not been initialized")

        triangles = self._sampling_state["triangles"]
        adaptive_probs = self._sampling_state["adaptive_probs"]
        area_probs = self._sampling_state["area_probs"]

        global_count = int(round(self.global_coverage_fraction * num_points))
        global_count = min(max(global_count, 0), num_points)
        adaptive_count = max(num_points - global_count, 0)

        adaptive_alloc = np.zeros(len(triangles), dtype=int)
        if adaptive_count > 0:
            sorted_indices = np.argsort(-adaptive_probs)
            cumulative = np.cumsum(adaptive_probs[sorted_indices])
            prefix_count = int(
                np.searchsorted(cumulative, self._ADAPTIVE_MASS_TARGET, side="left")
            ) + 1
            selected_count = min(
                adaptive_count,
                max(self._MIN_SELECTED_TRIANGLES, prefix_count),
            )
            selected_indices = sorted_indices[:selected_count]

            # Diversification guard: every selected triangle gets at least one point.
            adaptive_alloc[selected_indices] = 1
            remaining_adaptive = adaptive_count - selected_count

            if remaining_adaptive > 0:
                selected_probs = adaptive_probs[selected_indices]
                prob_sum = float(np.sum(selected_probs))
                if prob_sum <= 0.0 or not np.isfinite(prob_sum):
                    selected_probs = np.full(
                        len(selected_indices), 1.0 / len(selected_indices)
                    )
                else:
                    selected_probs = selected_probs / prob_sum
                adaptive_alloc[selected_indices] += self._rng.multinomial(
                    remaining_adaptive, selected_probs
                )

        global_alloc = self._rng.multinomial(global_count, area_probs)
        total_alloc = adaptive_alloc + global_alloc

        sampled_batches = []
        for triangle, count in zip(triangles, total_alloc):
            if count <= 0:
                continue
            sampled_batches.append(
                sample_uniform_points_in_triangle(triangle, int(count), self._rng)
            )

        if not sampled_batches:
            raise ValueError("adaptive sampler produced zero collocation points")

        return np.vstack(sampled_batches)

    def get_collocation_points(
        self,
        mesh: Any,
        model: Optional[Any] = None,
        iteration: int = 0,
        num_points: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if num_points is None:
            num_points = len(mesh.vertices)

        if (
            self._sampling_state is None
            or len(self._sampling_state["triangles"]) != int(getattr(mesh, "ne", 0))
        ):
            triangles, areas, _ = self._extract_triangle_data(mesh)
            self._set_sampling_distribution(triangles, areas, areas)

        points = self._sample_from_distribution(int(num_points))
        return points_to_tensors(points)

    def refine_mesh(
        self, mesh: Any, model: Any, iteration: int = 0
    ) -> Tuple[Any, bool]:
        triangles, areas, raw_scores, smoothed_scores, total_residual = (
            self._evaluate_residual_scores(mesh, model)
        )

        if not hasattr(model, "total_residual_history"):
            model.total_residual_history = []
        if not hasattr(model, "boundary_residual_history"):
            model.boundary_residual_history = []
        model.total_residual_history.append(float(total_residual))
        model.boundary_residual_history.append(float("nan"))

        max_score = float(np.max(smoothed_scores)) if len(smoothed_scores) else 0.0
        refine_mask = np.zeros_like(smoothed_scores, dtype=bool)
        was_refined = False
        should_refine = ((iteration + 1) % self.refine_period) == 0
        if max_score > 0.0 and should_refine:
            refine_mask = smoothed_scores > self.refinement_threshold * max_score
            mesh.ngmesh.Elements2D().NumPy()["refine"] = refine_mask
            mesh.Refine()
            was_refined = bool(np.any(refine_mask))
        else:
            mesh.ngmesh.Elements2D().NumPy()["refine"] = refine_mask

        (
            refined_triangles,
            refined_areas,
            refined_raw_scores,
            refined_smoothed_scores,
            _,
        ) = self._evaluate_residual_scores(mesh, model)
        self._set_sampling_distribution(
            refined_triangles, refined_areas, refined_smoothed_scores
        )

        self._last_refinement_stats = {
            "iteration": int(iteration),
            "total_residual": float(total_residual),
            "boundary_residual": float("nan"),
            "max_indicator": float(max_score),
            "mean_indicator": float(np.mean(smoothed_scores))
            if len(smoothed_scores)
            else 0.0,
            "mean_raw_indicator": float(np.mean(raw_scores)) if len(raw_scores) else 0.0,
            "mean_smoothed_indicator": float(np.mean(smoothed_scores))
            if len(smoothed_scores)
            else 0.0,
            "refined_elements": int(np.count_nonzero(refine_mask)),
            "was_refined": bool(was_refined),
            "refine_period": int(self.refine_period),
            "score_exponent": float(self.score_exponent),
            "area_exponent": float(self.area_exponent),
            "global_coverage_fraction": float(self.global_coverage_fraction),
            "smoothing_lambda": float(self.smoothing_lambda),
        }

        return mesh, was_refined

    def get_error_indicators(self, mesh: Any, model: Any) -> torch.Tensor:
        _, _, _, smoothed_scores, _ = self._evaluate_residual_scores(mesh, model)
        return torch.tensor(smoothed_scores, dtype=torch.float32)

    def log_iteration(self, iteration: int, mesh: Any, model: Any) -> dict:
        base_log = super().log_iteration(iteration, mesh, model)
        base_log["sampling_strategy"] = "adaptive_element_sampling"

        if self._last_refinement_stats:
            base_log.update(self._last_refinement_stats)

        if self._sampling_state is not None:
            base_log["sampling_elements"] = int(len(self._sampling_state["triangles"]))

        return base_log
