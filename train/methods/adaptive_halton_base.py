"""Adaptive residual sampling with a fixed Halton coverage backbone."""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np
import torch

from .adaptive_persistent import AdaptivePersistentMethod
from .quasi_random import HAS_QMC
from .sampling import points_to_tensors, sample_points_in_domain

if HAS_QMC:
    from scipy.stats import qmc
else:
    qmc = None


class AdaptiveHaltonBaseMethod(AdaptivePersistentMethod):
    """Persistent adaptive sampling anchored by a Halton coverage budget.

    The method reserves a fixed fraction of the collocation budget for Halton
    coverage, then spends the remainder using residual-guided mesh sampling.
    Residual scores are rank-normalized before persistence so the persistence
    mechanism depends on relative difficulty rather than raw PDE scale.
    """

    name = "adaptive_halton_base"
    description = "Halton-backed persistent adaptive residual sampling"

    def __init__(
        self,
        *args: Any,
        domain_bounds: Tuple[float, float, float, float] = (0.0, 5.0, 0.0, 5.0),
        backbone_fraction: float = 0.5,
        persistence_alpha: float = 0.5,
        warmup_iterations: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, persistence_alpha=persistence_alpha, **kwargs)
        self.domain_bounds = domain_bounds
        self.backbone_fraction = float(np.clip(backbone_fraction, 0.0, 1.0))
        self.warmup_iterations = max(0, int(warmup_iterations))
        self._halton_backbone_cache: dict[int, np.ndarray] = {}
        self._active_iteration = 0

    def initialize_run_state(self, **kwargs) -> None:
        super().initialize_run_state(**kwargs)
        self._halton_backbone_cache = {}
        self._active_iteration = 0
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
        self._set_sampling_distribution(
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
            "global_coverage_fraction": float(self.global_coverage_fraction),
            "smoothing_lambda": float(self.smoothing_lambda),
            "persistence_alpha": float(self.persistence_alpha),
            "backbone_fraction": float(self.backbone_fraction),
            "warmup_iterations": int(self.warmup_iterations),
            "sampling_indicator": "rank_persistent_residual",
            "refinement_indicator": "current_smoothed_residual",
        }

        return mesh, was_refined

    def _generate_halton_backbone(
        self, mesh: Any, num_points: int, iteration: int
    ) -> np.ndarray:
        if num_points <= 0:
            return np.empty((0, 2), dtype=float)
        if num_points in self._halton_backbone_cache:
            return self._halton_backbone_cache[num_points].copy()
        if not HAS_QMC or qmc is None:
            raise ImportError("scipy.stats.qmc required for adaptive_halton_base")

        x_min, x_max, y_min, y_max = self.domain_bounds
        x_range = x_max - x_min
        y_range = y_max - y_min
        base_seed = 0 if self.seed is None else int(self.seed)
        offset = 0

        def batch_generator(size: int) -> np.ndarray:
            nonlocal offset
            sampler = qmc.Halton(d=2, seed=base_seed + offset)
            unit_points = sampler.random(size)
            scaled_points = np.empty_like(unit_points)
            scaled_points[:, 0] = unit_points[:, 0] * x_range + x_min
            scaled_points[:, 1] = unit_points[:, 1] * y_range + y_min
            offset += size
            return scaled_points

        points = sample_points_in_domain(
            mesh,
            num_points,
            batch_generator,
            batch_size=max(1000, num_points * 4),
            max_batches=100,
            warn_label="Halton backbone points",
            method_name=self.name,
            iteration=iteration,
        )
        self._halton_backbone_cache[num_points] = points.copy()
        return points

    def get_collocation_points(
        self,
        mesh: Any,
        model: Optional[Any] = None,
        iteration: int = 0,
        num_points: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if num_points is None:
            num_points = len(list(mesh.vertices))
        num_points = int(num_points)
        self._active_iteration = int(iteration)

        if (
            self._sampling_state is None
            or len(self._sampling_state["triangles"]) != int(getattr(mesh, "ne", 0))
        ):
            triangles, areas, _ = self._extract_triangle_data(mesh)
            self._set_sampling_distribution(triangles, areas, areas)

        backbone_count = int(round(self.backbone_fraction * num_points))
        backbone_count = min(max(backbone_count, 0), num_points)
        adaptive_count = num_points - backbone_count

        point_blocks = []
        if backbone_count:
            point_blocks.append(
                self._generate_halton_backbone(mesh, backbone_count, iteration)
            )
        if adaptive_count:
            point_blocks.append(self._sample_from_distribution(adaptive_count))

        if point_blocks:
            points = np.vstack(point_blocks)
            points = points[self._rng.permutation(len(points))]
        else:
            points = np.empty((0, 2), dtype=float)
        return points_to_tensors(points)

    def get_error_indicators(self, mesh: Any, model: Any) -> torch.Tensor:
        triangles, _, _, smoothed_scores, _ = self._evaluate_residual_scores(mesh, model)
        sampling_scores = self._blend_persistent_scores(triangles, smoothed_scores)
        return torch.tensor(sampling_scores, dtype=torch.float32)

    def log_iteration(self, iteration: int, mesh: Any, model: Any) -> dict:
        log = super().log_iteration(iteration, mesh, model)
        log.update(
            {
                "sampling_strategy": "halton_backbone_rank_persistent_adaptive",
                "backbone_fraction": float(self.backbone_fraction),
                "warmup_iterations": int(self.warmup_iterations),
                "rank_persistent_scores": True,
            }
        )
        return log
