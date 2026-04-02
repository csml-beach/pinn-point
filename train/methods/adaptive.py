"""
Adaptive mesh refinement method.

This method refines the mesh based on PDE residual error indicators,
concentrating collocation points in regions with high residual.
"""

import torch
from typing import Tuple, Optional, Any
from ngsolve import GridFunction, BaseVector, Integrate, VOL, BND, H1
from .base import TrainingMethod
from config import DEVICE


class AdaptiveMethod(TrainingMethod):
    """Residual-based adaptive mesh refinement method.

    Uses PINN residuals to identify high-error regions and refines
    the mesh locally to improve solution accuracy.
    """

    name = "adaptive"
    description = "Residual-based adaptive mesh refinement"

    def __init__(self, refinement_threshold: float = 0.5):
        """Initialize adaptive method.

        Args:
            refinement_threshold: Fraction of max error for refinement
                                  (elements with error > threshold * max_error are refined)
        """
        self.refinement_threshold = refinement_threshold

    def _compute_residual_indicators(
        self, mesh: Any, model: Any
    ) -> tuple[Any, Any, object, float, float]:
        """Build the residual field and elementwise indicators on the current mesh."""
        res = model.PDE_residual(model.mesh_x, model.mesh_y).detach().cpu().numpy()

        fe_space = H1(mesh, order=1, dirichlet=".*")
        residuals = GridFunction(fe_space)
        residuals.vec[:] = BaseVector(res.flatten())
        residuals = residuals * residuals

        eta2 = Integrate(residuals, mesh, VOL, element_wise=True)
        total_residual = float(Integrate(residuals, mesh, VOL))
        boundary_residual = float(Integrate(residuals, mesh, BND))
        return fe_space, residuals, eta2, total_residual, boundary_residual

    def get_collocation_points(
        self, mesh: Any, model: Optional[Any] = None, iteration: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get collocation points from current mesh vertices.

        For adaptive method, points come directly from the mesh vertices
        which are concentrated in high-residual regions after refinement.

        Args:
            mesh: NGSolve mesh object
            model: Optional model (not used for vertex extraction)
            iteration: Current iteration

        Returns:
            Tuple of (x, y) tensors with mesh vertex coordinates
        """
        from geometry import export_vertex_coordinates

        coords = export_vertex_coordinates(mesh)
        x, y = coords.unbind(1)

        return x.float(), y.float()

    def refine_mesh(
        self, mesh: Any, model: Any, iteration: int = 0
    ) -> Tuple[Any, bool]:
        """Refine mesh based on PINN residual error indicators.

        Marks elements for refinement where the squared residual exceeds
        a threshold fraction of the maximum element error.

        Args:
            mesh: NGSolve mesh to refine
            model: Trained PINN model for residual computation
            iteration: Current iteration

        Returns:
            Tuple of (refined_mesh, was_refined)
        """
        _, _, eta2, total_residual, boundary_residual = self._compute_residual_indicators(
            mesh, model
        )

        if not hasattr(model, "total_residual_history"):
            model.total_residual_history = []
        if not hasattr(model, "boundary_residual_history"):
            model.boundary_residual_history = []
        model.total_residual_history.append(total_residual)
        model.boundary_residual_history.append(boundary_residual)

        # Mark elements for refinement
        eta2_np = eta2.NumPy()
        maxerr = float(max(eta2)) if len(eta2_np) else 0.0
        if maxerr > 0.0:
            refine_mask = eta2_np > self.refinement_threshold * maxerr
            mesh.ngmesh.Elements2D().NumPy()["refine"] = refine_mask
            mesh.Refine()
            was_refined = bool(refine_mask.any())
        else:
            mesh.ngmesh.Elements2D().NumPy()["refine"] = eta2_np > 0.0
            was_refined = False

        return mesh, was_refined

    def get_error_indicators(self, mesh: Any, model: Any) -> torch.Tensor:
        """Compute element-wise residual error indicators.

        Args:
            mesh: Current mesh
            model: Trained model

        Returns:
            Tensor of squared residuals per element
        """
        from geometry import export_vertex_coordinates

        coords = export_vertex_coordinates(mesh)
        mesh_x, mesh_y = coords.unbind(1)

        res = model.PDE_residual(mesh_x.to(DEVICE), mesh_y.to(DEVICE)).detach()
        return res**2

    def log_iteration(self, iteration: int, mesh: Any, model: Any) -> dict:
        """Log adaptive-specific information."""
        base_log = super().log_iteration(iteration, mesh, model)

        # Add residual statistics
        try:
            from geometry import export_vertex_coordinates

            coords = export_vertex_coordinates(mesh)
            mesh_x, mesh_y = coords.unbind(1)
            res = model.PDE_residual(mesh_x.to(DEVICE), mesh_y.to(DEVICE)).detach()

            base_log.update(
                {
                    "max_residual": float(res.abs().max()),
                    "mean_residual": float(res.abs().mean()),
                    "num_elements": int(getattr(mesh, "ne", 0)),
                }
            )
        except Exception:
            pass

        return base_log
