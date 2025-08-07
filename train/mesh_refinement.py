"""
Mesh refinement and error computation functions.
Contains functions for adaptive mesh refinement based on PINN residuals and error analysis.
"""

import torch
import numpy as np
import os
from ngsolve import *
from geometry import export_vertex_coordinates
from fem_solver import solve_FEM
from training import train_model
from config import DEVICE, DIRECTORY, MESH_CONFIG


def compute_model_error(model, reference_mesh, reference_solution, export_images=False, iteration=None):
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
    reference_fes = reference_solution.space  # Get finite element space from reference solution
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

    # Export visualization if requested
    if export_images and iteration is not None:
        from visualization import export_to_png
        
        # Project error CoefficientFunction to GridFunction for visualization
        error_gf = GridFunction(reference_fes)
        error_gf.Set(error)  # Project the coefficient function to grid function
        
        # Debug: Check error field values
        error_values = error_gf.vec.FV().NumPy()
        print(f"Error visualization - Range: {error_values.min():.2e} to {error_values.max():.2e}, Mean: {error_values.mean():.2e}")
        
        export_to_png(
            reference_mesh,
            error_gf,  # Use GridFunction instead of CoefficientFunction
            fieldname="errors",
            filename=f"errors_{iteration}.png",
        )

    print(f"Total Error (vs reference): {total_error:.6e}, Boundary Error: {boundary_error:.6e}")


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
    mesh.ngmesh.Elements2D().NumPy()["refine"] = eta2.NumPy() > refinement_threshold * maxerr
    
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


def adapt_mesh_and_train(model, mesh, dataset, reference_mesh, reference_solution, 
                         epochs=None, export_images=False, iteration=None):
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
    print(f"\n--- Adaptation Iteration {iteration + 1 if iteration is not None else 'N/A'} ---")
    
    # Create finite element space for residual computation (without solving FEM)
    from ngsolve import H1
    fe_space = H1(mesh, order=1, dirichlet=".*")
    model.fes = fe_space  # Store finite element space for residual computation
    
    # Train PINN model
    train_model(model, dataset, epochs)
    
    # Compute model error against high-fidelity reference solution
    compute_model_error(model, reference_mesh, reference_solution, export_images, iteration)
    
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
    
    print(f"Creating high-fidelity reference solution with mesh size factor {mesh_size_factor}...")
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
        rate = (point_counts[i] - point_counts[i-1]) / point_counts[i-1]
        refinement_rates.append(rate)
    
    analysis = {
        "initial_points": point_counts[0] if point_counts else 0,
        "final_points": point_counts[-1] if point_counts else 0,
        "total_refinement_factor": point_counts[-1] / point_counts[0] if point_counts and point_counts[0] > 0 else 0,
        "avg_refinement_rate": np.mean(refinement_rates) if refinement_rates else 0,
        "refinement_history": point_counts,
    }
    
    return analysis


def compute_random_residuals(model, initial_mesh, fe_space, export_images=False, iteration=None):
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
        distances = np.sum((random_coords - np.array([mx, my]))**2, axis=1)
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
    if not hasattr(model, 'boundary_residual_history'):
        model.boundary_residual_history = []
    if not hasattr(model, 'total_residual_history'):
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
    
    print(f"Random residuals computed - Total: {total_residual:.2e}, Boundary: {boundary_residual:.2e}")


def compute_random_model_error(model, reference_mesh, reference_solution, export_images=False, iteration=None):
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
    u_pred = model.forward(
        torch.tensor(ref_x, dtype=torch.float32).to(DEVICE),
        torch.tensor(ref_y, dtype=torch.float32).to(DEVICE),
    ).detach().cpu().numpy()

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

    # Export visualization if requested
    if export_images and iteration is not None:
        from visualization import export_to_png
        
        # Project error CoefficientFunction to GridFunction for visualization
        error_gf = GridFunction(reference_fes)
        error_gf.Set(error)  # Project the coefficient function to grid function
        
        # Debug: Check error field values
        error_values = error_gf.vec.FV().NumPy()
        print(f"Random error visualization - Range: {error_values.min():.2e} to {error_values.max():.2e}, Mean: {error_values.mean():.2e}")
        
        export_to_png(
            reference_mesh,
            error_gf,  # Use GridFunction instead of CoefficientFunction
            fieldname="random_errors",
            filename=f"random_errors_{iteration}.png",
        )

    print(f"Random Model - Total Error (vs reference): {total_error:.6e}, Boundary Error: {boundary_error:.6e}")
