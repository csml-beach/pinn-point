"""Entropy-balanced adaptive residual sampling."""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np
import torch

from .adaptive_persistent import AdaptivePersistentMethod
from .sampling import points_to_tensors, sample_uniform_points_in_triangle


class AdaptiveEntropyBalancedMethod(AdaptivePersistentMethod):
    """Mesh-native adaptive sampling with entropy-controlled coverage.

    This method avoids an external quasi-random backbone. Instead, it constructs
    one element-level sampling distribution that interpolates between geometric
    area coverage and rank-persistent residual focus. The interpolation strength
    is driven by residual concentration: flat residuals keep broad coverage,
    concentrated residuals increase adaptive focus.
    """

    name = "adaptive_entropy_balanced"
    description = "Entropy-balanced rank-persistent adaptive residual sampling"

    def __init__(
        self,
        *args: Any,
        persistence_alpha: float = 0.5,
        lambda_min: float = 0.25,
        lambda_max: float = 0.75,
        rank_gamma: float = 1.0,
        warmup_iterations: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, persistence_alpha=persistence_alpha, **kwargs)
        self.lambda_min = float(np.clip(lambda_min, 0.0, 1.0))
        self.lambda_max = float(np.clip(lambda_max, self.lambda_min, 1.0))
        self.rank_gamma = max(float(rank_gamma), 1e-6)
        self.warmup_iterations = max(0, int(warmup_iterations))
        self._active_iteration = 0
        self._last_entropy_stats: dict[str, float] = {}

    def initialize_run_state(self, **kwargs) -> None:
        super().initialize_run_state(**kwargs)
        self._active_iteration = 0
        self._last_entropy_stats = {}
        return None

    @staticmethod
    def _rank_normalize_scores(scores: np.ndarray) -> np.ndarray:
        scores = np.nan_to_num(
            np.asarray(scores, dtype=float),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        n_scores = len(scores)
        if n_scores == 0:
            return scores
        if n_scores == 1:
            return np.ones_like(scores, dtype=float)

        order = np.argsort(scores, kind="mergesort")
        sorted_scores = scores[order]
        sorted_ranks = np.empty(n_scores, dtype=float)

        start = 0
        while start < n_scores:
            end = start + 1
            while end < n_scores and sorted_scores[end] == sorted_scores[start]:
                end += 1
            sorted_ranks[start:end] = 0.5 * (start + end - 1) / (n_scores - 1)
            start = end

        ranks = np.empty(n_scores, dtype=float)
        ranks[order] = sorted_ranks
        return ranks

    def _blend_persistent_scores(
        self, triangles: np.ndarray, current_scores: np.ndarray
    ) -> np.ndarray:
        current_rank_scores = self._rank_normalize_scores(current_scores)
        previous_scores = self._match_previous_scores(triangles)
        if previous_scores is None or self._active_iteration < self.warmup_iterations:
            return current_rank_scores

        alpha = min(max(self.persistence_alpha, 0.0), 1.0)
        return alpha * previous_scores + (1.0 - alpha) * current_rank_scores

    def _entropy_weight(self, adaptive_probs: np.ndarray) -> tuple[float, float, float]:
        n_probs = len(adaptive_probs)
        if n_probs <= 1:
            entropy_norm = 1.0
        else:
            probs = np.clip(np.asarray(adaptive_probs, dtype=float), 1e-12, None)
            probs = probs / np.sum(probs)
            entropy = -float(np.sum(probs * np.log(probs)))
            entropy_norm = entropy / float(np.log(n_probs))
            entropy_norm = float(np.clip(entropy_norm, 0.0, 1.0))

        concentration = 1.0 - entropy_norm
        adaptive_lambda = self.lambda_min + (
            self.lambda_max - self.lambda_min
        ) * concentration
        return float(adaptive_lambda), float(concentration), float(entropy_norm)

    def _set_entropy_balanced_distribution(
        self, triangles: np.ndarray, areas: np.ndarray, scores: np.ndarray
    ) -> None:
        areas = np.nan_to_num(np.asarray(areas, dtype=float), nan=0.0)
        areas = np.clip(areas, 1e-12, None)
        area_probs = areas / np.sum(areas)

        scores = np.nan_to_num(
            np.asarray(scores, dtype=float),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        scores = np.clip(scores, 0.0, None)
        if np.sum(scores) <= 0.0:
            adaptive_probs = area_probs.copy()
        else:
            weights = np.power(scores + 1e-12, self.rank_gamma)
            adaptive_probs = weights / np.sum(weights)

        adaptive_lambda, concentration, entropy_norm = self._entropy_weight(
            adaptive_probs
        )
        balanced_probs = (
            (1.0 - adaptive_lambda) * area_probs
            + adaptive_lambda * adaptive_probs
        )
        balanced_probs = balanced_probs / np.sum(balanced_probs)

        self._sampling_state = {
            "triangles": triangles,
            "areas": areas,
            "adaptive_probs": adaptive_probs,
            "area_probs": area_probs,
            "balanced_probs": balanced_probs,
        }
        self._last_entropy_stats = {
            "adaptive_lambda": float(adaptive_lambda),
            "residual_concentration": float(concentration),
            "normalized_entropy": float(entropy_norm),
        }

    @staticmethod
    def _deterministic_quota_alloc(probs: np.ndarray, num_points: int) -> np.ndarray:
        expected = np.asarray(probs, dtype=float) * int(num_points)
        alloc = np.floor(expected).astype(int)
        remaining = int(num_points) - int(np.sum(alloc))
        if remaining > 0:
            fractional = expected - alloc
            order = np.argsort(-fractional, kind="mergesort")
            alloc[order[:remaining]] += 1
        return alloc

    def _sample_from_distribution(self, num_points: int) -> np.ndarray:
        if self._sampling_state is None:
            raise RuntimeError("sampling distribution has not been initialized")

        triangles = self._sampling_state["triangles"]
        probs = self._sampling_state.get("balanced_probs")
        if probs is None:
            probs = self._sampling_state["adaptive_probs"]

        alloc = self._deterministic_quota_alloc(probs, int(num_points))
        sampled_batches = []
        for triangle, count in zip(triangles, alloc):
            if count <= 0:
                continue
            sampled_batches.append(
                sample_uniform_points_in_triangle(triangle, int(count), self._rng)
            )

        if not sampled_batches:
            raise ValueError("entropy-balanced sampler produced zero collocation points")

        points = np.vstack(sampled_batches)
        return points[self._rng.permutation(len(points))]

    def get_collocation_points(
        self,
        mesh: Any,
        model: Optional[Any] = None,
        iteration: int = 0,
        num_points: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if num_points is None:
            num_points = len(list(mesh.vertices))
        self._active_iteration = int(iteration)

        if (
            self._sampling_state is None
            or len(self._sampling_state["triangles"]) != int(getattr(mesh, "ne", 0))
        ):
            triangles, areas, _ = self._extract_triangle_data(mesh)
            self._set_entropy_balanced_distribution(
                triangles, areas, np.zeros_like(areas)
            )

        points = self._sample_from_distribution(int(num_points))
        return points_to_tensors(points)

    def refine_mesh(self, mesh: Any, model: Any, iteration: int = 0):
        self._active_iteration = int(iteration)
        triangles, areas, raw_scores, smoothed_scores, total_residual = (
            self._evaluate_residual_scores(mesh, model, iteration=iteration)
        )

        if not hasattr(model, "total_residual_history"):
            model.total_residual_history = []
        if not hasattr(model, "boundary_residual_history"):
            model.boundary_residual_history = []
        model.total_residual_history.append(float(total_residual))
        model.boundary_residual_history.append(float("nan"))

        sampling_scores = self._blend_persistent_scores(triangles, smoothed_scores)

        max_refine_score = float(np.max(smoothed_scores)) if len(smoothed_scores) else 0.0
        refine_mask = np.zeros_like(smoothed_scores, dtype=bool)
        was_refined = False
        should_refine = ((iteration + 1) % self.refine_period) == 0
        if max_refine_score > 0.0 and should_refine:
            refine_mask = smoothed_scores > self.refinement_threshold * max_refine_score
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
        refined_sampling_scores = self._blend_persistent_scores(
            refined_triangles, refined_smoothed_scores
        )
        self._set_entropy_balanced_distribution(
            refined_triangles, refined_areas, refined_sampling_scores
        )
        self._previous_score_state = {
            "centroids": self._triangle_centroids(refined_triangles),
            "scores": np.asarray(refined_sampling_scores, dtype=float),
        }

        self._last_refinement_stats = {
            "iteration": int(iteration),
            "total_residual": float(total_residual),
            "boundary_residual": float("nan"),
            "max_indicator": float(np.max(refined_sampling_scores))
            if len(refined_sampling_scores)
            else 0.0,
            "mean_indicator": float(np.mean(refined_sampling_scores))
            if len(refined_sampling_scores)
            else 0.0,
            "mean_raw_indicator": float(np.mean(refined_raw_scores))
            if len(refined_raw_scores)
            else 0.0,
            "mean_smoothed_indicator": float(np.mean(refined_smoothed_scores))
            if len(refined_smoothed_scores)
            else 0.0,
            "mean_persistent_indicator": float(np.mean(refined_sampling_scores))
            if len(refined_sampling_scores)
            else 0.0,
            "max_refinement_indicator": float(max_refine_score),
            "refined_elements": int(np.count_nonzero(refine_mask)),
            "was_refined": bool(was_refined),
            "refine_period": int(self.refine_period),
            "score_exponent": float(self.score_exponent),
            "area_exponent": float(self.area_exponent),
            "global_coverage_fraction": 0.0,
            "smoothing_lambda": float(self.smoothing_lambda),
            "persistence_alpha": float(self.persistence_alpha),
            "lambda_min": float(self.lambda_min),
            "lambda_max": float(self.lambda_max),
            "rank_gamma": float(self.rank_gamma),
            "warmup_iterations": int(self.warmup_iterations),
            "sampling_indicator": "entropy_balanced_rank_persistent_residual",
            "refinement_indicator": "current_smoothed_residual",
        }
        self._last_refinement_stats.update(self._last_entropy_stats)

        return mesh, was_refined

    def get_error_indicators(self, mesh: Any, model: Any) -> torch.Tensor:
        triangles, _, _, smoothed_scores, _ = self._evaluate_residual_scores(mesh, model)
        sampling_scores = self._blend_persistent_scores(triangles, smoothed_scores)
        return torch.tensor(sampling_scores, dtype=torch.float32)

    def log_iteration(self, iteration: int, mesh: Any, model: Any) -> dict:
        log = super().log_iteration(iteration, mesh, model)
        log.update(
            {
                "sampling_strategy": "entropy_balanced_rank_persistent_adaptive",
                "lambda_min": float(self.lambda_min),
                "lambda_max": float(self.lambda_max),
                "rank_gamma": float(self.rank_gamma),
                "warmup_iterations": int(self.warmup_iterations),
            }
        )
        log.update(self._last_entropy_stats)
        return log
