"""
Hybrid adaptive mesh refinement using residual and anchor-point data error.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from config import DEVICE
from .adaptive import AdaptiveMethod
from .sampling import sample_points_in_domain


class AdaptiveHybridAnchorMethod(AdaptiveMethod):
    """Adaptive refinement driven by residual and fixed-anchor supervised error."""

    name = "adaptive_hybrid_anchor"
    description = "Adaptive refinement using residual + fixed anchor supervised error"

    def __init__(
        self,
        refinement_threshold: float = 0.5,
        domain_bounds: tuple[float, float, float, float] = (0.0, 5.0, 0.0, 5.0),
        anchor_count: int = 512,
        alpha: float = 1.0,
        beta: float = 1.0,
        normalization_quantile: float = 0.95,
        seed: int | None = None,
    ):
        super().__init__(refinement_threshold=refinement_threshold)
        self.domain_bounds = domain_bounds
        self.anchor_count = int(anchor_count)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.normalization_quantile = float(normalization_quantile)
        self.seed = seed
        self._rng = np.random.RandomState(seed)
        self._anchor_points: np.ndarray | None = None
        self._anchor_values: torch.Tensor | None = None
        self._anchor_x_device: torch.Tensor | None = None
        self._anchor_y_device: torch.Tensor | None = None
        self._anchor_values_device: torch.Tensor | None = None

    def initialize_run_state(
        self, initial_mesh: Any, fem_solution: Any | None = None, **kwargs
    ) -> None:
        """Create the fixed anchor set once per run from the initial FEM solution."""
        if self._anchor_points is not None and self._anchor_values is not None:
            return
        if fem_solution is None:
            raise ValueError("Hybrid adaptive method requires an initial FEM solution")

        x_min, x_max, y_min, y_max = self.domain_bounds
        points = sample_points_in_domain(
            initial_mesh,
            self.anchor_count,
            lambda batch_size: np.column_stack(
                (
                    self._rng.uniform(x_min, x_max, size=batch_size),
                    self._rng.uniform(y_min, y_max, size=batch_size),
                )
            ),
            batch_size=max(256, self.anchor_count * 4),
            max_batches=40,
            warn_label="hybrid anchor points",
            method_name=self.name,
            iteration=-1,
        )

        values = []
        for x_coord, y_coord in points:
            try:
                values.append(float(fem_solution(initial_mesh(float(x_coord), float(y_coord)))))
            except TypeError:
                values.append(float(fem_solution(float(x_coord), float(y_coord))))

        self._anchor_points = points.astype(float, copy=False)
        self._anchor_values = torch.tensor(values, dtype=torch.float32)
        self._anchor_x_device = None
        self._anchor_y_device = None
        self._anchor_values_device = None
        print(f"Initialized fixed hybrid anchor set with {len(points):,} points")

    def _get_anchor_tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._anchor_points is None or self._anchor_values is None:
            raise RuntimeError("Hybrid anchor set has not been initialized")
        if (
            self._anchor_x_device is None
            or self._anchor_y_device is None
            or self._anchor_values_device is None
        ):
            self._anchor_x_device = torch.tensor(
                self._anchor_points[:, 0], dtype=torch.float32, device=DEVICE
            )
            self._anchor_y_device = torch.tensor(
                self._anchor_points[:, 1], dtype=torch.float32, device=DEVICE
            )
            self._anchor_values_device = self._anchor_values.to(DEVICE)
        return self._anchor_x_device, self._anchor_y_device, self._anchor_values_device

    def _compute_anchor_point_errors(self, model: Any) -> np.ndarray:
        anchor_x, anchor_y, anchor_values = self._get_anchor_tensors()
        with torch.no_grad():
            pred = model.forward(anchor_x, anchor_y).reshape(-1)
        errors = torch.square(pred - anchor_values).detach().cpu().numpy()
        return np.clip(errors, 0.0, None)

    def _aggregate_anchor_errors_to_elements(
        self, mesh: Any, point_errors: np.ndarray
    ) -> tuple[np.ndarray, int]:
        if self._anchor_points is None:
            raise RuntimeError("Hybrid anchor set has not been initialized")

        num_elements = int(getattr(mesh, "ne", 0))
        element_error_sum = np.zeros(num_elements, dtype=float)
        element_counts = np.zeros(num_elements, dtype=float)
        assigned = 0

        for (x_coord, y_coord), error_value in zip(self._anchor_points, point_errors):
            try:
                element_nr = int(mesh(float(x_coord), float(y_coord)).nr)
            except Exception:
                continue
            if 0 <= element_nr < num_elements:
                element_error_sum[element_nr] += float(error_value)
                element_counts[element_nr] += 1.0
                assigned += 1

        with np.errstate(divide="ignore", invalid="ignore"):
            element_errors = np.divide(
                element_error_sum,
                np.maximum(element_counts, 1.0),
                out=np.zeros_like(element_error_sum),
            )

        return element_errors, assigned

    def _robust_normalize(self, values: np.ndarray) -> tuple[np.ndarray, float]:
        values = np.asarray(values, dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return np.zeros_like(values), 1.0

        q = min(max(self.normalization_quantile, 0.0), 1.0)
        scale = float(np.quantile(finite, q))
        if not np.isfinite(scale) or scale <= 1e-12:
            scale = float(np.max(finite)) if finite.size else 1.0
        if not np.isfinite(scale) or scale <= 1e-12:
            scale = 1.0

        clipped = np.clip(values, 0.0, scale)
        return clipped / (scale + 1e-12), scale

    def refine_mesh(self, mesh: Any, model: Any, iteration: int = 0):
        _, areas, residual_scores, total_residual = self._evaluate_residual_scores(
            mesh, model
        )
        if not hasattr(model, "total_residual_history"):
            model.total_residual_history = []
        if not hasattr(model, "boundary_residual_history"):
            model.boundary_residual_history = []
        model.total_residual_history.append(total_residual)
        model.boundary_residual_history.append(float("nan"))

        point_errors = self._compute_anchor_point_errors(model)
        element_anchor_errors, assigned_anchor_points = self._aggregate_anchor_errors_to_elements(
            mesh, point_errors
        )

        residual_norm, residual_scale = self._robust_normalize(residual_scores)
        anchor_norm, anchor_scale = self._robust_normalize(element_anchor_errors)
        score = self.alpha * residual_norm + self.beta * anchor_norm

        max_score = float(np.max(score)) if score.size else 0.0
        if max_score > 0.0:
            refine_mask = score > self.refinement_threshold * max_score
            mesh.ngmesh.Elements2D().NumPy()["refine"] = refine_mask
            mesh.Refine()
            was_refined = bool(refine_mask.any())
        else:
            refine_mask = np.zeros_like(score, dtype=bool)
            mesh.ngmesh.Elements2D().NumPy()["refine"] = refine_mask
            was_refined = False

        refined_triangles, refined_areas, refined_residual_scores, _ = (
            self._evaluate_residual_scores(mesh, model)
        )
        refined_anchor_errors, refined_assigned_anchor_points = (
            self._aggregate_anchor_errors_to_elements(mesh, point_errors)
        )
        refined_residual_norm, refined_residual_scale = self._robust_normalize(
            refined_residual_scores
        )
        refined_anchor_norm, refined_anchor_scale = self._robust_normalize(
            refined_anchor_errors
        )
        refined_score = (
            self.alpha * refined_residual_norm + self.beta * refined_anchor_norm
        )
        self._set_sampling_distribution(
            refined_triangles, refined_areas, refined_areas * refined_score
        )

        self._last_refinement_stats = {
            "iteration": int(iteration),
            "total_residual": float(total_residual),
            "boundary_residual": float("nan"),
            "max_indicator": float(max_score),
            "mean_indicator": float(np.mean(score)) if score.size else 0.0,
            "refined_elements": int(np.count_nonzero(refine_mask)),
            "was_refined": bool(was_refined),
        }

        if not hasattr(model, "hybrid_refinement_stats_history"):
            model.hybrid_refinement_stats_history = []
        model.hybrid_refinement_stats_history.append(
            {
                "iteration": int(iteration),
                "anchor_count": int(self.anchor_count),
                "assigned_anchor_points": int(assigned_anchor_points),
                "assigned_anchor_points_refined": int(refined_assigned_anchor_points),
                "alpha": float(self.alpha),
                "beta": float(self.beta),
                "residual_scale_q": float(refined_residual_scale),
                "anchor_error_scale_q": float(refined_anchor_scale),
                "score_max": float(max_score),
                "score_mean": float(np.mean(refined_score)) if refined_score.size else 0.0,
                "refined_elements": int(np.count_nonzero(refine_mask)),
            }
        )

        print(
            "Hybrid refinement stats: "
            f"assigned_anchors={refined_assigned_anchor_points}, "
            f"residual_q={refined_residual_scale:.3e}, "
            f"anchor_q={refined_anchor_scale:.3e}, "
            f"refined_elements={int(np.count_nonzero(refine_mask))}"
        )

        return mesh, was_refined

    def log_iteration(self, iteration: int, mesh: Any, model: Any) -> dict:
        base_log = super().log_iteration(iteration, mesh, model)
        if getattr(model, "hybrid_refinement_stats_history", None):
            base_log.update(model.hybrid_refinement_stats_history[-1])
        base_log["sampling_strategy"] = "hybrid_anchor"
        return base_log
