"""
Mesh refinement and error computation functions.
Contains functions for adaptive mesh refinement based on PINN residuals and error analysis.
"""

import torch
import numpy as np
import os

# Explicit ngsolve imports for static analyzers and clarity
from ngsolve import GridFunction, BaseVector, Integrate, VOL, BND
from geometry import export_vertex_coordinates
from fem_solver import solve_FEM
from training import train_model
from config import DEVICE, DIRECTORY, MESH_CONFIG
import math


def compute_model_residual_on_reference(
    model, reference_mesh, reference_solution, export_images=False, iteration=None
):
    """Integrate PDE residual^2 over the same fine reference mesh used for error.

    This is evaluation-only and must not be used for mesh refinement.
    Records totals on model.fixed_total_residual_history and
    model.fixed_boundary_residual_history.
    """
    # Coordinates on reference mesh (same approach as compute_model_error)
    ref_coords = export_vertex_coordinates(reference_mesh)
    try:
        ref_x, ref_y = ref_coords.unbind(1)
        tx, ty = ref_x.to(DEVICE).float(), ref_y.to(DEVICE).float()
    except AttributeError:
        # Fallback if export returns ndarray
        import torch as _torch

        tx = _torch.tensor(ref_coords[:, 0], dtype=_torch.float32, device=DEVICE)
        ty = _torch.tensor(ref_coords[:, 1], dtype=_torch.float32, device=DEVICE)

    # Evaluate PDE residual at reference vertices (requires gradients for autograd-based PDE terms)
    tx.requires_grad_(True)
    ty.requires_grad_(True)
    with torch.enable_grad():
        # Support alternate attribute names if present
        if hasattr(model, "PDE_residual"):
            r = model.PDE_residual(tx, ty)  # shape [N]
        elif hasattr(model, "pde_residual"):
            # Some models may expose a lowercase variant taking stacked coords
            coords = torch.stack([tx, ty], dim=1)
            coords.requires_grad_(True)
            r = model.pde_residual(coords)
        else:
            raise AttributeError("Model does not expose PDE_residual or pde_residual")
    r2 = (r * r).detach().cpu().numpy().reshape(-1)

    # Create GridFunction for residual^2 on the reference FE space
    reference_fes = reference_solution.space
    residuals_gf = GridFunction(reference_fes)
    # Try to set vector; if sizes differ, attempt anyway and let integration handle or raise
    try:
        residuals_gf.vec[:] = BaseVector(r2.flatten())
    except Exception as e:
        # Initialize histories if missing, then append NaN to keep CSV aligned
        if not hasattr(model, "fixed_total_residual_history"):
            model.fixed_total_residual_history = []
        if not hasattr(model, "fixed_boundary_residual_history"):
            model.fixed_boundary_residual_history = []
        model.fixed_total_residual_history.append(float("nan"))
        model.fixed_boundary_residual_history.append(float("nan"))
        print(
            f"[Fixed residual] Failed to set GridFunction vector (len={r2.size}): {e}"
        )
        return

    # Integrate over domain and boundary
    try:
        total_residual = Integrate(residuals_gf, reference_mesh, VOL)
        boundary_residual = Integrate(residuals_gf, reference_mesh, BND)
    except Exception as e:
        # Append NaN on integration failure to avoid empty CSV columns
        if not hasattr(model, "fixed_total_residual_history"):
            model.fixed_total_residual_history = []
        if not hasattr(model, "fixed_boundary_residual_history"):
            model.fixed_boundary_residual_history = []
        model.fixed_total_residual_history.append(float("nan"))
        model.fixed_boundary_residual_history.append(float("nan"))
        print(f"[Fixed residual] Integration failed: {e}")
        return

    # Store histories (initialize if missing)
    if not hasattr(model, "fixed_total_residual_history"):
        model.fixed_total_residual_history = []
    if not hasattr(model, "fixed_boundary_residual_history"):
        model.fixed_boundary_residual_history = []
    if not hasattr(model, "fixed_rms_residual_history"):
        model.fixed_rms_residual_history = []
    model.fixed_total_residual_history.append(float(total_residual))
    model.fixed_boundary_residual_history.append(float(boundary_residual))

    # Compute and store RMS residual (area-normalized) for interpretability
    try:
        if not hasattr(model, "_reference_area"):
            area = Integrate(1, reference_mesh, VOL)
            model._reference_area = float(area)
        area = getattr(model, "_reference_area", None)
        if area and area > 0:
            rms = math.sqrt(float(total_residual) / area)
        else:
            rms = float("nan")
    except Exception:
        rms = float("nan")
    model.fixed_rms_residual_history.append(rms)

    # Optional visualization
    if export_images and iteration is not None:
        from visualization import export_to_png

        export_to_png(
            reference_mesh,
            residuals_gf,
            fieldname="fixed_residual",
            filename=f"fixed_residual_{iteration}.png",
        )

    try:
        msg = f"[Fixed residual] Total: {total_residual:.6e}, Boundary: {boundary_residual:.6e}, RMS: {rms:.6e}"
    except Exception:
        msg = f"[Fixed residual] Total: {total_residual}, Boundary: {boundary_residual}, RMS: {rms}"
    print(msg)


def compute_model_residual_on_reference_quadrature(
    model, reference_mesh, export_images=False, iteration=None
):
    """Robust fixed-grid residual integral on the reference mesh.

    Primary path: for each triangle, evaluate residual at its three vertices,
    use mean(r^2) as element value, and sum area * mean(r^2).
    Fallback: Monte Carlo over the domain if mesh APIs fail.
    """
    # Domain area (if available)
    try:
        domain_area = float(Integrate(1.0, reference_mesh, VOL))
    except Exception:
        domain_area = float("nan")

    # Mesh-based per-element accumulation
    try:
        total = 0.0
        area_sum = 0.0
        for el in reference_mesh.Elements2D():
            verts = []
            for v in el.vertices:
                p = reference_mesh[v].point
                verts.append((p.x, p.y))
            if len(verts) != 3:
                continue
            (x1, y1), (x2, y2), (x3, y3) = verts
            area = abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)) * 0.5
            if area <= 0:
                continue

            # Evaluate residual at vertices
            r2_vals = []
            for xq, yq in ((x1, y1), (x2, y2), (x3, y3)):
                tx = torch.tensor(
                    [xq], dtype=torch.float32, device=DEVICE, requires_grad=True
                )
                ty = torch.tensor(
                    [yq], dtype=torch.float32, device=DEVICE, requires_grad=True
                )
                with torch.enable_grad():
                    if hasattr(model, "PDE_residual"):
                        rq = model.PDE_residual(tx, ty)
                    elif hasattr(model, "pde_residual"):
                        coords = torch.stack([tx, ty], dim=1).requires_grad_(True)
                        rq = model.pde_residual(coords)
                    else:
                        raise AttributeError("Model missing PDE_residual/pde_residual")
                r2_val = float((rq * rq).detach().cpu().numpy().reshape(-1)[0])
                if np.isfinite(r2_val):
                    r2_vals.append(r2_val)

            if not r2_vals:
                continue
            mean_r2 = float(np.mean(r2_vals))
            total += area * mean_r2
            area_sum += area

        if area_sum <= 0:
            raise RuntimeError("No positive-area triangles processed")
        if not np.isfinite(domain_area) or domain_area <= 0:
            domain_area = area_sum
        rms = float(math.sqrt(total / domain_area))

        if not hasattr(model, "fixed_total_residual_history"):
            model.fixed_total_residual_history = []
        model.fixed_total_residual_history.append(float(total))

        print(f"[Fixed residual quad] Total: {total:.6e}, RMS(est): {rms:.6e}")
        return
    except Exception as e:
        print(
            f"[Fixed residual quad] mesh-based integration failed: {e}. Falling back to Monte Carlo."
        )

    # Fallback: Monte Carlo sampling inside the domain bounding box
    try:
        # Get bounding box from mesh coordinates
        coords = export_vertex_coordinates(reference_mesh)
        try:
            xs, ys = coords[:, 0].cpu().numpy(), coords[:, 1].cpu().numpy()
        except Exception:
            xs, ys = np.asarray(coords[:, 0]), np.asarray(coords[:, 1])
        xmin, xmax = float(np.min(xs)), float(np.max(xs))
        ymin, ymax = float(np.min(ys)), float(np.max(ys))

        # Sample points with rejection using mesh(...) membership
        rng = np.random.default_rng()
        target = 10000
        acc_x, acc_y = [], []
        attempts = 0
        max_attempts = target * 20
        while len(acc_x) < target and attempts < max_attempts:
            rx = rng.uniform(xmin, xmax, size=2048)
            ry = rng.uniform(ymin, ymax, size=2048)
            for xq, yq in zip(rx, ry):
                try:
                    if reference_mesh(xq, yq).nr != -1:
                        acc_x.append(xq)
                        acc_y.append(yq)
                        if len(acc_x) >= target:
                            break
                except Exception:
                    pass
            attempts += 2048

        if len(acc_x) < 1000:
            raise RuntimeError(f"Too few accepted samples: {len(acc_x)}")

        tx = torch.tensor(acc_x, dtype=torch.float32, device=DEVICE, requires_grad=True)
        ty = torch.tensor(acc_y, dtype=torch.float32, device=DEVICE, requires_grad=True)
        with torch.enable_grad():
            if hasattr(model, "PDE_residual"):
                r = model.PDE_residual(tx, ty)
            elif hasattr(model, "pde_residual"):
                r = model.pde_residual(
                    torch.stack([tx, ty], dim=1).requires_grad_(True)
                )
            else:
                raise AttributeError("Model missing PDE_residual/pde_residual")
        r2 = (r * r).detach().float().cpu().numpy()
        r2 = r2[np.isfinite(r2)]
        if r2.size == 0:
            raise RuntimeError("All Monte Carlo residual^2 values are non-finite")

        if not np.isfinite(domain_area) or domain_area <= 0:
            domain_area = float(Integrate(1.0, reference_mesh, VOL))
        total = float(np.mean(r2) * domain_area)
        rms = float(math.sqrt(total / domain_area))

        if not hasattr(model, "fixed_total_residual_history"):
            model.fixed_total_residual_history = []
        model.fixed_total_residual_history.append(total)

        print(f"[Fixed residual MC] Total: {total:.6e}, RMS(est): {rms:.6e}")
    except Exception as e:
        if not hasattr(model, "fixed_total_residual_history"):
            model.fixed_total_residual_history = []
        model.fixed_total_residual_history.append(float("nan"))
        print(f"[Fixed residual MC] failed: {e}")


def compute_model_residual_rms_on_reference(model, reference_mesh, iteration=None):
    """Compute RMS of PDE residual over reference mesh vertices (no integration or area weighting).

    This provides a fair, mesh-independent scalar by sampling residuals at the reference mesh points
    used for error evaluation.
    """
    # Get coordinates as tensors
    ref_coords = export_vertex_coordinates(reference_mesh)
    try:
        ref_x, ref_y = ref_coords.unbind(1)
        tx, ty = ref_x.to(DEVICE).float(), ref_y.to(DEVICE).float()
    except Exception:
        import torch as _torch

        tx = _torch.tensor(ref_coords[:, 0], dtype=_torch.float32, device=DEVICE)
        ty = _torch.tensor(ref_coords[:, 1], dtype=_torch.float32, device=DEVICE)

    tx.requires_grad_(True)
    ty.requires_grad_(True)
    with torch.enable_grad():
        if hasattr(model, "PDE_residual"):
            r = model.PDE_residual(tx, ty)
        elif hasattr(model, "pde_residual"):
            coords = torch.stack([tx, ty], dim=1).requires_grad_(True)
            r = model.pde_residual(coords)
        else:
            raise AttributeError("Model does not expose PDE_residual or pde_residual")

    r2 = (r * r).detach().cpu().numpy().reshape(-1)
    # Filter non-finite entries
    import numpy as _np

    r2 = r2[_np.isfinite(r2)]
    if r2.size == 0:
        rms = float("nan")
    else:
        rms = float(_np.sqrt(_np.mean(r2)))

    if not hasattr(model, "fixed_rms_residual_history"):
        model.fixed_rms_residual_history = []
    model.fixed_rms_residual_history.append(rms)

    try:
        print(f"[Fixed residual RMS] RMS: {rms:.6e}")
    except Exception:
        print(f"[Fixed residual RMS] RMS: {rms}")


def compute_model_error_rms_on_reference(
    model, reference_mesh, reference_solution, iteration=None
):
    """Compute RMS of model error (u_pred - u_ref) sampled at reference mesh vertices.

    Keeps existing integrated error logic intact; this just adds an RMS time series.
    """
    # Coordinates
    ref_coords = export_vertex_coordinates(reference_mesh)
    try:
        ref_x, ref_y = ref_coords.unbind(1)
        tx, ty = ref_x.to(DEVICE).float(), ref_y.to(DEVICE).float()
    except Exception:
        import torch as _torch

        tx = _torch.tensor(ref_coords[:, 0], dtype=_torch.float32, device=DEVICE)
        ty = _torch.tensor(ref_coords[:, 1], dtype=_torch.float32, device=DEVICE)

    with torch.no_grad():
        u_pred = model.forward(tx, ty).detach().cpu().numpy().reshape(-1)

    # Reference solution at dofs; assume P1 => dofs ~ vertices
    try:
        u_ref = reference_solution.vec.FV().NumPy().reshape(-1)
    except Exception:
        try:
            u_ref = np.array([v for v in reference_solution.vec]).reshape(-1)
        except Exception:
            u_ref = None

    import numpy as _np

    if u_ref is None or u_ref.size != u_pred.size:
        # Length mismatch; compute NaN and log once
        rms = float("nan")
        print(
            f"[Error RMS] length mismatch (pred={u_pred.size}, ref={0 if u_ref is None else u_ref.size}); set NaN"
        )
    else:
        diff2 = (u_pred - u_ref) ** 2
        diff2 = diff2[_np.isfinite(diff2)]
        rms = float(_np.sqrt(_np.mean(diff2))) if diff2.size > 0 else float("nan")

    if not hasattr(model, "total_error_rms_history"):
        model.total_error_rms_history = []
    model.total_error_rms_history.append(rms)


def compute_model_error(
    model, reference_mesh, reference_solution, export_images=False, iteration=None
):
    """Compute model error against a high-fidelity reference FEM solution.

    Args:
        model: PINN model
        reference_mesh: Fine reference mesh (computed once)
        reference_solution: Reference FEM solution on fine mesh (computed once)
        export_images: Whether to export error visualization
        iteration: Current iteration number for file naming

    Returns:
        None (updates model error history)
    """
    from geometry import export_vertex_coordinates

    # Get reference mesh coordinates
    ref_coords = export_vertex_coordinates(reference_mesh)
    ref_x, ref_y = ref_coords.unbind(1)

    # Get PINN predictions at reference mesh points
    with torch.no_grad():
        u_pred = model.forward(ref_x.to(DEVICE).float(), ref_y.to(DEVICE).float())
    u_pred = u_pred.detach().cpu().numpy()

    # Create GridFunction with PINN predictions on reference mesh
    reference_fes = (
        reference_solution.space
    )  # Get finite element space from reference solution
    u_pinn_on_ref = GridFunction(reference_fes)
    u_pinn_on_ref.vec[:] = BaseVector(u_pred.flatten())

    # Ensure output directory exists
    if not os.path.exists(DIRECTORY):
        os.makedirs(DIRECTORY)

    # Compute squared error against reference solution
    error = (u_pinn_on_ref - reference_solution) * (u_pinn_on_ref - reference_solution)

    # Integrate errors over different regions
    total_error = Integrate(error, reference_mesh, VOL)
    boundary_error = Integrate(error, reference_mesh, BND)

    # Store error history
    model.total_error_history.append(total_error)
    model.boundary_error_history.append(boundary_error)

    # Also compute RMS of residuals on the same reference mesh (evaluation-only)
    try:
        compute_model_residual_rms_on_reference(
            model, reference_mesh, iteration=iteration
        )
    except Exception as e:
        print(f"Warning: failed fixed residual RMS evaluation (adaptive): {e}")

    # Compute integration-based fixed residual as well (for comparison/plots)
    try:
        compute_model_residual_on_reference_quadrature(
            model, reference_mesh, export_images=False, iteration=iteration
        )
    except Exception as e:
        print(f"Warning: failed fixed residual integral evaluation (adaptive): {e}")

    # And compute RMS of error at reference vertices for a DOF-agnostic metric
    try:
        compute_model_error_rms_on_reference(
            model, reference_mesh, reference_solution, iteration=iteration
        )
    except Exception as e:
        print(f"Warning: failed fixed error RMS evaluation (adaptive): {e}")

    # Export visualization if requested
    if export_images and iteration is not None:
        from visualization import export_to_png

        # Project error CoefficientFunction to GridFunction for visualization
        error_gf = GridFunction(reference_fes)
        error_gf.Set(error)  # Project the coefficient function to grid function

        # Debug: Check error field values
        error_values = error_gf.vec.FV().NumPy()
        print(
            f"Error visualization - Range: {error_values.min():.2e} to {error_values.max():.2e}, Mean: {error_values.mean():.2e}"
        )

        export_to_png(
            reference_mesh,
            error_gf,  # Use GridFunction instead of CoefficientFunction
            fieldname="errors",
            filename=f"errors_{iteration}.png",
        )

    print(
        f"Total Error (vs reference): {total_error:.6e}, Boundary Error: {boundary_error:.6e}"
    )


def refine_mesh(model, fe_space, mesh, export_images=False, iteration=None):
    """Refine mesh based on PINN residual error.

    Args:
        model: PINN model
        fe_space: Finite element space
        mesh: NGSolve mesh to refine
        export_images: Whether to export residual visualization
        iteration: Current iteration number for file naming

    Returns:
        None (modifies mesh in-place and updates model)
    """
    # Compute PDE residuals at mesh points
    res = model.PDE_residual(model.mesh_x, model.mesh_y).detach().numpy()

    # Create GridFunction with residuals
    residuals = GridFunction(fe_space)
    residuals.vec[:] = BaseVector(res.flatten())
    residuals = residuals * residuals  # Square the residuals

    # Compute element-wise and total residuals
    eta2 = Integrate(residuals, mesh, VOL, element_wise=True)
    total_residual = Integrate(residuals, mesh, VOL)
    boundary_residual = Integrate(residuals, mesh, BND)

    # Store residual history
    model.boundary_residual_history.append(boundary_residual)
    model.total_residual_history.append(total_residual)

    # Export visualization if requested
    if export_images and iteration is not None:
        from visualization import export_to_png

        export_to_png(
            mesh,
            residuals,
            fieldname="residuals",
            filename=f"residuals_{iteration}.png",
        )

    # Mark elements for refinement based on error indicator
    maxerr = max(eta2)
    refinement_threshold = MESH_CONFIG["refinement_threshold"]
    mesh.ngmesh.Elements2D().NumPy()["refine"] = (
        eta2.NumPy() > refinement_threshold * maxerr
    )

    # Refine the mesh
    mesh.Refine()

    # Update mesh coordinates in model
    mesh_x, mesh_y = export_vertex_coordinates(mesh).unbind(1)
    model.mesh_x = mesh_x
    model.mesh_y = mesh_y

    # Update mesh history
    model.mesh_point_history.append((model.mesh_x.numpy(), model.mesh_y.numpy()))
    model.mesh_point_count_history.append(len(model.mesh_x))

    print(f"Mesh refined: {len(model.mesh_x)} points, Max Error: {maxerr:.6e}")


def adapt_mesh_and_train(
    model,
    mesh,
    dataset,
    reference_mesh,
    reference_solution,
    epochs=None,
    export_images=False,
    iteration=None,
):
    """Perform one iteration of mesh adaptation and training.

    Args:
        model: PINN model
        mesh: NGSolve mesh (for refinement)
        dataset: Training dataset
        reference_mesh: Fine reference mesh for error computation
        reference_solution: Reference FEM solution on fine mesh
        epochs: Number of training epochs
        export_images: Whether to export visualizations
        iteration: Current iteration number

    Returns:
        None (modifies model and mesh in-place)
    """
    print(
        f"\n--- Adaptation Iteration {iteration + 1 if iteration is not None else 'N/A'} ---"
    )

    # Create finite element space for residual computation (without solving FEM)
    from ngsolve import H1

    fe_space = H1(mesh, order=1, dirichlet=".*")
    model.fes = fe_space  # Store finite element space for residual computation

    # Train PINN model
    train_model(model, dataset, epochs)
    # Fine-tune for a few extra epochs to stabilize after mesh changes
    try:
        base = epochs if isinstance(epochs, int) and epochs > 0 else 0
        extra_epochs = max(1, int(0.2 * base)) if base > 0 else 1
    except Exception:
        extra_epochs = 1
    if extra_epochs > 0:
        print(f"Fine-tuning for {extra_epochs} extra epochs before evaluation")
        train_model(model, dataset, extra_epochs)

    # Compute model error against high-fidelity reference solution
    compute_model_error(
        model, reference_mesh, reference_solution, export_images, iteration
    )

    # Refine mesh based on PINN residuals
    refine_mesh(model, fe_space, mesh, export_images, iteration)


def create_reference_solution(mesh_size_factor=0.05):
    """Create a high-fidelity reference mesh and FEM solution.

    This should be called once at the beginning to create a very fine reference
    solution that will be used for accurate error assessment throughout the experiment.

    Args:
        mesh_size_factor: Factor to create very fine mesh (smaller = finer)

    Returns:
        tuple: (reference_mesh, reference_solution)
    """
    from geometry import create_initial_mesh

    print(
        f"Creating high-fidelity reference solution with mesh size factor {mesh_size_factor}..."
    )
    print("Warning: This may take significant time and memory!")

    # Create very fine mesh
    reference_mesh = create_initial_mesh(maxh=mesh_size_factor)

    # Solve FEM on reference mesh
    reference_solution, reference_fes = solve_FEM(reference_mesh)

    # Count points for information
    from geometry import export_vertex_coordinates

    ref_coords = export_vertex_coordinates(reference_mesh)
    num_ref_points = len(ref_coords)

    print(f"Reference solution created with {num_ref_points:,} points")
    print("This reference solution will be used for all error computations")

    return reference_mesh, reference_solution


def compute_mesh_quality_metrics(mesh):
    """Compute various mesh quality metrics.

    Args:
        mesh: NGSolve mesh

    Returns:
        dict: Dictionary containing mesh quality metrics
    """
    num_vertices = len(mesh.vertices)
    num_elements = len(mesh.elements)

    # Compute element areas/volumes
    element_areas = []
    for el in mesh.elements:
        # This is a simplified calculation - in practice, you'd compute actual areas
        element_areas.append(1.0)  # Placeholder

    metrics = {
        "num_vertices": num_vertices,
        "num_elements": num_elements,
        "avg_element_area": np.mean(element_areas) if element_areas else 0,
        "min_element_area": np.min(element_areas) if element_areas else 0,
        "max_element_area": np.max(element_areas) if element_areas else 0,
    }

    return metrics


def analyze_refinement_patterns(model):
    """Analyze mesh refinement patterns over iterations.

    Args:
        model: PINN model with refinement history

    Returns:
        dict: Analysis results
    """
    if not model.mesh_point_count_history:
        return {"error": "No refinement history available"}

    point_counts = model.mesh_point_count_history
    refinement_rates = []

    for i in range(1, len(point_counts)):
        rate = (point_counts[i] - point_counts[i - 1]) / point_counts[i - 1]
        refinement_rates.append(rate)

    analysis = {
        "initial_points": point_counts[0] if point_counts else 0,
        "final_points": point_counts[-1] if point_counts else 0,
        "total_refinement_factor": (
            point_counts[-1] / point_counts[0]
            if point_counts and point_counts[0] > 0
            else 0
        ),
        "avg_refinement_rate": np.mean(refinement_rates) if refinement_rates else 0,
        "refinement_history": point_counts,
    }

    return analysis


def compute_random_residuals(
    model, initial_mesh, fe_space, export_images=False, iteration=None
):
    """Compute residuals for random point model and export visualization.

    Args:
        model: PINN model with random points
        initial_mesh: Initial mesh (for domain bounds and visualization)
        fe_space: Finite element space for GridFunction creation
        export_images: Whether to export residual visualization
        iteration: Current iteration number for file naming

    Returns:
        None (updates model residual history)
    """
    # Compute PDE residuals at random points
    res = model.PDE_residual(model.mesh_x, model.mesh_y).detach().numpy()

    # For random points, we need to interpolate residuals to the mesh for visualization
    # Create GridFunction and interpolate residuals from random points to mesh
    residuals = GridFunction(fe_space)

    # Get mesh coordinates
    from fem_solver import export_vertex_coordinates

    mesh_coords = export_vertex_coordinates(initial_mesh)
    mesh_x, mesh_y = mesh_coords.T

    # Simple interpolation: find nearest random point for each mesh point
    import numpy as np

    random_coords = torch.stack([model.mesh_x, model.mesh_y], dim=1).cpu().numpy()

    interpolated_res = []
    for mx, my in zip(mesh_x, mesh_y):
        # Find nearest random point
        distances = np.sum((random_coords - np.array([mx, my])) ** 2, axis=1)
        nearest_idx = np.argmin(distances)
        interpolated_res.append(res[nearest_idx])

    # Set interpolated residuals in GridFunction
    from ngsolve import BaseVector

    residuals.vec[:] = BaseVector(np.array(interpolated_res).flatten())
    residuals = residuals * residuals  # Square the residuals

    # Compute total residuals for tracking
    from ngsolve import Integrate, VOL, BND

    total_residual = Integrate(residuals, initial_mesh, VOL)
    boundary_residual = Integrate(residuals, initial_mesh, BND)

    # Store residual history
    if not hasattr(model, "boundary_residual_history"):
        model.boundary_residual_history = []
    if not hasattr(model, "total_residual_history"):
        model.total_residual_history = []

    model.boundary_residual_history.append(boundary_residual)
    model.total_residual_history.append(total_residual)

    # Export visualization if requested
    if export_images and iteration is not None:
        from visualization import export_to_png

        export_to_png(
            initial_mesh,
            residuals,
            fieldname="random_residuals",
            filename=f"random_residuals_{iteration}.png",
        )

    print(
        f"Random residuals computed - Total: {total_residual:.2e}, Boundary: {boundary_residual:.2e}"
    )


def compute_random_model_error(
    model, reference_mesh, reference_solution, export_images=False, iteration=None
):
    """Compute error for random model against reference solution.

    Args:
        model: Random PINN model
        reference_mesh: High-fidelity reference mesh
        reference_solution: High-fidelity reference solution
        export_images: Whether to export error visualization
        iteration: Current iteration number for file naming

    Returns:
        None (updates model error history)
    """
    # Get reference mesh coordinates
    from fem_solver import export_vertex_coordinates

    ref_coords = export_vertex_coordinates(reference_mesh)
    ref_x, ref_y = ref_coords.T

    # Evaluate PINN on reference mesh
    from config import DEVICE

    u_pred = (
        model.forward(
            torch.tensor(ref_x, dtype=torch.float32).to(DEVICE),
            torch.tensor(ref_y, dtype=torch.float32).to(DEVICE),
        )
        .detach()
        .cpu()
        .numpy()
    )

    # Create GridFunction for PINN solution on reference mesh
    reference_fes = reference_solution.space
    u_pinn_on_ref = GridFunction(reference_fes)
    u_pinn_on_ref.vec[:] = BaseVector(u_pred.flatten())

    # Compute error: (PINN - reference)^2
    error = (u_pinn_on_ref - reference_solution) * (u_pinn_on_ref - reference_solution)

    # Compute integrated error metrics
    total_error = Integrate(error, reference_mesh, VOL)
    boundary_error = Integrate(error, reference_mesh, BND)

    # Store error history in model
    if not hasattr(model, "total_error_history"):
        model.total_error_history = []
    if not hasattr(model, "boundary_error_history"):
        model.boundary_error_history = []

    model.total_error_history.append(total_error)
    model.boundary_error_history.append(boundary_error)

    # Also compute RMS of residuals on the same reference mesh (evaluation-only)
    try:
        compute_model_residual_rms_on_reference(
            model, reference_mesh, iteration=iteration
        )
    except Exception as e:
        print(f"Warning: failed fixed residual RMS evaluation (random): {e}")

    # Compute integration-based fixed residual as well (for comparison/plots)
    try:
        compute_model_residual_on_reference_quadrature(
            model, reference_mesh, export_images=False, iteration=iteration
        )
    except Exception as e:
        print(f"Warning: failed fixed residual integral evaluation (random): {e}")

    # And compute RMS of error at reference vertices for a DOF-agnostic metric
    try:
        compute_model_error_rms_on_reference(
            model, reference_mesh, reference_solution, iteration=iteration
        )
    except Exception as e:
        print(f"Warning: failed fixed error RMS evaluation (random): {e}")

    # Export visualization if requested
    if export_images and iteration is not None:
        from visualization import export_to_png

        # Project error CoefficientFunction to GridFunction for visualization
        error_gf = GridFunction(reference_fes)
        error_gf.Set(error)  # Project the coefficient function to grid function

        # Debug: Check error field values
        error_values = error_gf.vec.FV().NumPy()
        print(
            f"Random error visualization - Range: {error_values.min():.2e} to {error_values.max():.2e}, Mean: {error_values.mean():.2e}"
        )

        export_to_png(
            reference_mesh,
            error_gf,  # Use GridFunction instead of CoefficientFunction
            fieldname="random_errors",
            filename=f"random_errors_{iteration}.png",
        )

    print(
        f"Random Model - Total Error (vs reference): {total_error:.6e}, Boundary Error: {boundary_error:.6e}"
    )
