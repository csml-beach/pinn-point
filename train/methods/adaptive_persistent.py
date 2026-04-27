"""
Persistence-weighted adaptive residual-guided collocation sampling.

This variant augments the mesh-scaffold adaptive sampler with a persistence
signal: triangles that remain difficult over multiple refinement rounds are
weighted more heavily than transient residual spikes.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .adaptive import AdaptiveMethod


class AdaptivePersistentMethod(AdaptiveMethod):
    """Residual-guided sampler with cross-iteration persistent scores."""

    name = "adaptive_persistent"
    description = "Residual-guided interior sampling with persistence-weighted scoring"

    def __init__(
        self,
        *args,
        persistence_alpha: float = 0.6,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.persistence_alpha = float(persistence_alpha)
        self._previous_score_state: dict[str, np.ndarray] | None = None

    def initialize_run_state(self, **kwargs) -> None:
        self._previous_score_state = None
        self._sampling_state = None
        self._last_refinement_stats = {}
        return None

    @staticmethod
    def _triangle_centroids(triangles: np.ndarray) -> np.ndarray:
        return np.mean(np.asarray(triangles, dtype=float), axis=1)

    def _match_previous_scores(self, triangles: np.ndarray) -> np.ndarray | None:
        if self._previous_score_state is None:
            return None

        previous_centroids = self._previous_score_state.get("centroids")
        previous_scores = self._previous_score_state.get("scores")
        if previous_centroids is None or previous_scores is None:
            return None
        if len(previous_centroids) == 0 or len(previous_scores) == 0:
            return None

        current_centroids = self._triangle_centroids(triangles)
        deltas = current_centroids[:, None, :] - previous_centroids[None, :, :]
        nearest_indices = np.argmin(np.sum(deltas * deltas, axis=2), axis=1)
        return np.asarray(previous_scores[nearest_indices], dtype=float)

    def _blend_persistent_scores(
        self, triangles: np.ndarray, current_scores: np.ndarray
    ) -> np.ndarray:
        current_scores = np.asarray(current_scores, dtype=float)
        previous_scores = self._match_previous_scores(triangles)
        if previous_scores is None:
            return current_scores

        alpha = min(max(self.persistence_alpha, 0.0), 1.0)
        return alpha * previous_scores + (1.0 - alpha) * current_scores

    def refine_mesh(self, mesh: Any, model: Any, iteration: int = 0):
        triangles, areas, raw_scores, smoothed_scores, total_residual = (
            self._evaluate_residual_scores(mesh, model, iteration=iteration)
        )

        if not hasattr(model, "total_residual_history"):
            model.total_residual_history = []
        if not hasattr(model, "boundary_residual_history"):
            model.boundary_residual_history = []
        model.total_residual_history.append(float(total_residual))
        model.boundary_residual_history.append(float("nan"))

        persistent_scores = self._blend_persistent_scores(triangles, smoothed_scores)

        max_score = float(np.max(persistent_scores)) if len(persistent_scores) else 0.0
        refine_mask = np.zeros_like(persistent_scores, dtype=bool)
        was_refined = False
        should_refine = ((iteration + 1) % self.refine_period) == 0
        if max_score > 0.0 and should_refine:
            refine_mask = persistent_scores > self.refinement_threshold * max_score
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
        ) = self._evaluate_residual_scores(mesh, model, iteration=iteration)
        refined_persistent_scores = self._blend_persistent_scores(
            refined_triangles, refined_smoothed_scores
        )
        self._set_sampling_distribution(
            refined_triangles, refined_areas, refined_persistent_scores
        )
        self._previous_score_state = {
            "centroids": self._triangle_centroids(refined_triangles),
            "scores": np.asarray(refined_persistent_scores, dtype=float),
        }

        self._last_refinement_stats = {
            "iteration": int(iteration),
            "total_residual": float(total_residual),
            "boundary_residual": float("nan"),
            "max_indicator": float(max_score),
            "mean_indicator": float(np.mean(refined_persistent_scores))
            if len(refined_persistent_scores)
            else 0.0,
            "mean_raw_indicator": float(np.mean(refined_raw_scores))
            if len(refined_raw_scores)
            else 0.0,
            "mean_smoothed_indicator": float(np.mean(refined_smoothed_scores))
            if len(refined_smoothed_scores)
            else 0.0,
            "mean_persistent_indicator": float(np.mean(refined_persistent_scores))
            if len(refined_persistent_scores)
            else 0.0,
            "refined_elements": int(np.count_nonzero(refine_mask)),
            "was_refined": bool(was_refined),
            "refine_period": int(self.refine_period),
            "score_exponent": float(self.score_exponent),
            "area_exponent": float(self.area_exponent),
            "global_coverage_fraction": float(self.global_coverage_fraction),
            "smoothing_lambda": float(self.smoothing_lambda),
            "persistence_alpha": float(self.persistence_alpha),
        }

        return mesh, was_refined

    def get_error_indicators(self, mesh: Any, model: Any) -> torch.Tensor:
        triangles, _, _, smoothed_scores, _ = self._evaluate_residual_scores(mesh, model)
        persistent_scores = self._blend_persistent_scores(triangles, smoothed_scores)
        return torch.tensor(persistent_scores, dtype=torch.float32)
