"""Power-tempered adaptive residual sampling."""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np
import torch

from .adaptive_persistent import AdaptivePersistentMethod
from .sampling import points_to_tensors, sample_uniform_points_in_triangle


class AdaptivePowerTemperedMethod(AdaptivePersistentMethod):
    """Mesh-native adaptive sampling with exponential residual tilting.

    The method uses a single element distribution:

        p_i proportional to area_i^a * exp(beta * z_i)

    where z_i is a rank-persistent residual score in [0, 1]. The temperature
    beta is increased when the raw residual field is concentrated and kept mild
    when residuals look diffuse/noisy. This keeps broad geometric coverage
    without importing an external low-discrepancy sequence.
    """

    name = "adaptive_power_tempered"
    description = "Power-tempered rank-persistent adaptive residual sampling"

    def __init__(
        self,
        *args: Any,
        persistence_alpha: float = 0.5,
        beta_min: float = 1.0,
        beta_max: float = 4.0,
        coverage_area_exponent: float = 0.5,
        coverage_floor: float = 0.0,
        warmup_iterations: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, persistence_alpha=persistence_alpha, **kwargs)
        self.beta_min = float(beta_min)
        self.beta_max = max(float(beta_max), self.beta_min)
        self.coverage_area_exponent = float(coverage_area_exponent)
        self.coverage_floor = float(np.clip(coverage_floor, 0.0, 1.0))
        self.warmup_iterations = max(0, int(warmup_iterations))
        self._active_iteration = 0
        self._last_tempering_stats: dict[str, float] = {}

    def initialize_run_state(self, **kwargs) -> None:
        super().initialize_run_state(**kwargs)
        self._active_iteration = 0
        self._last_tempering_stats = {}
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

    @staticmethod
    def _normalized_entropy_concentration(scores: np.ndarray) -> tuple[float, float]:
        scores = np.nan_to_num(
            np.asarray(scores, dtype=float),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        scores = np.clip(scores, 0.0, None)
        n_scores = len(scores)
        score_sum = float(np.sum(scores))
        if n_scores <= 1 or score_sum <= 0.0:
            return 0.0, 1.0

        probs = np.clip(scores / score_sum, 1e-12, None)
        probs = probs / np.sum(probs)
        entropy = -float(np.sum(probs * np.log(probs)))
        normalized_entropy = entropy / float(np.log(n_scores))
        normalized_entropy = float(np.clip(normalized_entropy, 0.0, 1.0))
        concentration = 1.0 - normalized_entropy
        return float(concentration), normalized_entropy

    def _set_power_tempered_distribution(
        self,
        triangles: np.ndarray,
        areas: np.ndarray,
        rank_scores: np.ndarray,
        concentration_scores: np.ndarray,
    ) -> None:
        areas = np.nan_to_num(np.asarray(areas, dtype=float), nan=0.0)
        areas = np.clip(areas, 1e-12, None)
        true_area_probs = areas / np.sum(areas)
        area_weights = np.power(areas, self.coverage_area_exponent)
        area_probs = area_weights / np.sum(area_weights)

        z = np.nan_to_num(
            np.asarray(rank_scores, dtype=float),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        )
        z = np.clip(z, 0.0, 1.0)

        concentration, normalized_entropy = self._normalized_entropy_concentration(
            concentration_scores
        )
        beta = self.beta_min + (self.beta_max - self.beta_min) * concentration
        tilt = np.exp(beta * (z - float(np.max(z)) if len(z) else z))
        weights = area_weights * tilt
        if float(np.sum(weights)) <= 0.0 or not np.all(np.isfinite(weights)):
            power_probs = area_probs
        else:
            power_probs = weights / np.sum(weights)

        if self.coverage_floor > 0.0:
            probs = (
                (1.0 - self.coverage_floor) * power_probs
                + self.coverage_floor * true_area_probs
            )
            probs = probs / np.sum(probs)
        else:
            probs = power_probs

        self._sampling_state = {
            "triangles": triangles,
            "areas": areas,
            "adaptive_probs": probs,
            "area_probs": area_probs,
            "true_area_probs": true_area_probs,
            "unfloored_power_tempered_probs": power_probs,
            "power_tempered_probs": probs,
        }
        self._last_tempering_stats = {
            "tempering_beta": float(beta),
            "residual_concentration": float(concentration),
            "normalized_entropy": float(normalized_entropy),
            "coverage_floor": float(self.coverage_floor),
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
        probs = self._sampling_state.get("power_tempered_probs")
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
            raise ValueError("power-tempered sampler produced zero collocation points")

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
            zero_scores = np.zeros_like(areas)
            self._set_power_tempered_distribution(
                triangles, areas, zero_scores, zero_scores
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
        self._set_power_tempered_distribution(
            refined_triangles,
            refined_areas,
            refined_sampling_scores,
            refined_smoothed_scores,
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
            "coverage_area_exponent": float(self.coverage_area_exponent),
            "global_coverage_fraction": float(self.coverage_floor),
            "smoothing_lambda": float(self.smoothing_lambda),
            "persistence_alpha": float(self.persistence_alpha),
            "beta_min": float(self.beta_min),
            "beta_max": float(self.beta_max),
            "coverage_floor": float(self.coverage_floor),
            "warmup_iterations": int(self.warmup_iterations),
            "sampling_indicator": "power_tempered_rank_persistent_residual",
            "refinement_indicator": "current_smoothed_residual",
        }
        self._last_refinement_stats.update(self._last_tempering_stats)

        return mesh, was_refined

    def get_error_indicators(self, mesh: Any, model: Any) -> torch.Tensor:
        triangles, _, _, smoothed_scores, _ = self._evaluate_residual_scores(mesh, model)
        sampling_scores = self._blend_persistent_scores(triangles, smoothed_scores)
        return torch.tensor(sampling_scores, dtype=torch.float32)

    def log_iteration(self, iteration: int, mesh: Any, model: Any) -> dict:
        log = super().log_iteration(iteration, mesh, model)
        log.update(
            {
                "sampling_strategy": "power_tempered_rank_persistent_adaptive",
                "beta_min": float(self.beta_min),
                "beta_max": float(self.beta_max),
                "coverage_area_exponent": float(self.coverage_area_exponent),
                "coverage_floor": float(self.coverage_floor),
                "warmup_iterations": int(self.warmup_iterations),
            }
        )
        log.update(self._last_tempering_stats)
        return log


class AdaptivePowerTemperedBeta25Method(AdaptivePowerTemperedMethod):
    """Power-tempered sampler with a conservative beta_max=2.5 cap."""

    name = "adaptive_power_tempered_beta25"
    description = "Power-tempered adaptive residual sampling (beta_max=2.5)"

    def __init__(self, *args: Any, beta_max: float = 2.5, **kwargs: Any) -> None:
        super().__init__(*args, beta_max=beta_max, **kwargs)


class AdaptivePowerTemperedBeta30Method(AdaptivePowerTemperedMethod):
    """Power-tempered sampler with a conservative beta_max=3.0 cap."""

    name = "adaptive_power_tempered_beta30"
    description = "Power-tempered adaptive residual sampling (beta_max=3.0)"

    def __init__(self, *args: Any, beta_max: float = 3.0, **kwargs: Any) -> None:
        super().__init__(*args, beta_max=beta_max, **kwargs)


class AdaptivePowerTemperedFloor15Method(AdaptivePowerTemperedMethod):
    """Power-tempered sampler with a 15% true-area coverage floor."""

    name = "adaptive_power_tempered_floor15"
    description = "Power-tempered adaptive residual sampling (coverage_floor=0.15)"

    def __init__(
        self, *args: Any, coverage_floor: float = 0.15, **kwargs: Any
    ) -> None:
        super().__init__(*args, coverage_floor=coverage_floor, **kwargs)


class AdaptivePowerTemperedFloor25Method(AdaptivePowerTemperedMethod):
    """Power-tempered sampler with a 25% true-area coverage floor."""

    name = "adaptive_power_tempered_floor25"
    description = "Power-tempered adaptive residual sampling (coverage_floor=0.25)"

    def __init__(
        self, *args: Any, coverage_floor: float = 0.25, **kwargs: Any
    ) -> None:
        super().__init__(*args, coverage_floor=coverage_floor, **kwargs)
