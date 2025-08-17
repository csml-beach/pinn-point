"""
Utility functions for PINN adaptive mesh training.
Contains helper functions and utilities used across different modules.
"""

import numpy as np
import random
import torch
import os
from ngsolve import *
from config import DEVICE, DIRECTORY


def fix_random_model_error(model, mesh, gfu, mesh_x, mesh_y, fe_space):
    """Fixed version of get_random_model_error function.
    
    Args:
        model: PINN model
        mesh: NGSolve mesh
        gfu: FEM solution (GridFunction)
        mesh_x: x-coordinates for evaluation
        mesh_y: y-coordinates for evaluation
        fe_space: Finite element space
        
    Returns:
        None (updates model error and residual history)
    """
    # Make predictions at the specified points
    u_pred = model.forward(mesh_x.to(DEVICE).float(), mesh_y.to(DEVICE).float())
    u_pred = u_pred.detach().cpu().numpy()
    
    # Create GridFunction with predictions
    u_plot = GridFunction(fe_space)
    u_plot.vec[:] = BaseVector(u_pred.flatten())

    # Ensure output directory exists
    if not os.path.exists(DIRECTORY):
        os.makedirs(DIRECTORY)

    # Compute errors
    error = (u_plot - gfu) * (u_plot - gfu)
    total_error = Integrate(error, mesh, VOL)
    boundary_error = Integrate(error, mesh, BND)

    # Compute residuals
    res = model.PDE_residual(mesh_x, mesh_y).detach().numpy()
    residuals = GridFunction(fe_space)
    residuals.vec[:] = BaseVector(res.flatten())
    residuals = residuals * residuals

    total_residual = Integrate(residuals, mesh, VOL)
    boundary_residual = Integrate(residuals, mesh, BND)

    # Update model history
    model.total_error_history.append(total_error)
    model.boundary_error_history.append(boundary_error)
    model.total_residual_history.append(total_residual)
    model.boundary_residual_history.append(boundary_residual)

    print(f"Random model - Total Error: {total_error:.6e}, Boundary Error: {boundary_error:.6e}")


def validate_mesh_points(mesh_x, mesh_y, mesh):
    """Validate that mesh points are within the domain.
    
    Args:
        mesh_x: x-coordinates
        mesh_y: y-coordinates
        mesh: NGSolve mesh
        
    Returns:
        bool: True if all points are valid
    """
    if len(mesh_x) != len(mesh_y):
        return False
    
    valid_count = 0
    for x, y in zip(mesh_x, mesh_y):
        try:
            if not mesh(float(x), float(y)).nr == -1:
                valid_count += 1
        except Exception:
            pass
    
    return valid_count == len(mesh_x)


def tensor_to_numpy_safe(tensor):
    """Safely convert tensor to numpy array.
    
    Args:
        tensor: PyTorch tensor
        
    Returns:
        numpy.ndarray: Converted array
    """
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return np.array(tensor)


def ensure_tensor(data, device=None):
    """Ensure data is a PyTorch tensor on the specified device.
    
    Args:
        data: Input data (tensor, array, or list)
        device: Target device (uses config default if None)
        
    Returns:
        torch.Tensor: Converted tensor
    """
    if device is None:
        device = DEVICE
    
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data)
    
    return data.to(device)


def set_global_seed(seed: int) -> None:
    """Set seeds for Python, NumPy, and Torch for reproducibility."""
    try:
        random.seed(seed)
    except Exception:
        pass
    try:
        np.random.seed(seed)
    except Exception:
        pass
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def create_directory_structure(base_dir=None):
    """Create the required directory structure for the project.
    
    Args:
        base_dir: Base directory (uses config default if None)
        
    Returns:
        dict: Created directories
    """
    if base_dir is None:
        base_dir = DIRECTORY
    
    directories = {
        "main": base_dir,
        "images": base_dir,
        "models": os.path.join(base_dir, "models"),
        "results": os.path.join(base_dir, "results"),
        "logs": os.path.join(base_dir, "logs"),
    }
    
    for name, path in directories.items():
        try:
            os.makedirs(path, exist_ok=True)
            print(f"Created/verified directory: {path}")
        except Exception as e:
            print(f"Warning: Could not create directory {path}: {e}")
    
    return directories


def save_model_checkpoint(model, filepath, additional_info=None):
    """Save model checkpoint with metadata.
    
    Args:
        model: PINN model
        filepath: Path to save checkpoint
        additional_info: Additional information to save
        
    Returns:
        bool: Success status
    """
    try:
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "mesh_x": tensor_to_numpy_safe(model.mesh_x),
            "mesh_y": tensor_to_numpy_safe(model.mesh_y),
            "total_error_history": model.total_error_history,
            "boundary_error_history": model.boundary_error_history,
            "train_loss_history": model.train_loss_history,
            "total_residual_history": model.total_residual_history,
            "boundary_residual_history": model.boundary_residual_history,
            "mesh_point_history": model.mesh_point_history,
            "mesh_point_count_history": model.mesh_point_count_history,
        }
        
        if additional_info:
            checkpoint.update(additional_info)
        
        torch.save(checkpoint, filepath)
        print(f"Model checkpoint saved to {filepath}")
        return True
        
    except Exception as e:
        print(f"Error saving checkpoint: {e}")
        return False


def load_model_checkpoint(model, filepath):
    """Load model checkpoint.
    
    Args:
        model: PINN model to load into
        filepath: Path to checkpoint file
        
    Returns:
        bool: Success status
    """
    try:
        checkpoint = torch.load(filepath, map_location=DEVICE)
        
        model.load_state_dict(checkpoint["model_state_dict"])
        model.mesh_x = torch.tensor(checkpoint["mesh_x"])
        model.mesh_y = torch.tensor(checkpoint["mesh_y"])
        model.total_error_history = checkpoint.get("total_error_history", [])
        model.boundary_error_history = checkpoint.get("boundary_error_history", [])
        model.train_loss_history = checkpoint.get("train_loss_history", [])
        model.total_residual_history = checkpoint.get("total_residual_history", [])
        model.boundary_residual_history = checkpoint.get("boundary_residual_history", [])
        model.mesh_point_history = checkpoint.get("mesh_point_history", [])
        model.mesh_point_count_history = checkpoint.get("mesh_point_count_history", [])
        
        print(f"Model checkpoint loaded from {filepath}")
        return True
        
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return False


def print_model_summary(model):
    """Print a summary of the model and its training history.
    
    Args:
        model: PINN model
        
    Returns:
        None
    """
    print("\n" + "="*50)
    print("PINN MODEL SUMMARY")
    print("="*50)
    
    # Model architecture
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Architecture: {model.b1.in_features} -> {model.hidden_size} -> {model.hidden_size} -> 1")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Current mesh info
    if hasattr(model, 'mesh_x') and model.mesh_x is not None:
        print(f"Current mesh points: {len(model.mesh_x):,}")
    
    # Training history
    if model.mesh_point_count_history:
        initial_points = model.mesh_point_count_history[0]
        current_points = model.mesh_point_count_history[-1]
        refinement_factor = current_points / initial_points if initial_points > 0 else 0
        print(f"Mesh refinement: {initial_points:,} -> {current_points:,} (×{refinement_factor:.2f})")
    
    if model.total_error_history:
        final_error = model.total_error_history[-1]
        print(f"Final total error: {final_error:.6e}")
    
    if model.train_loss_history:
        print(f"Training epochs completed: {len(model.train_loss_history)}")
    
    print("="*50)


def get_system_info():
    """Get system information for debugging and logging.
    
    Returns:
        dict: System information
    """
    info = {
        "device": str(DEVICE),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    
    if torch.cuda.is_available():
        info["cuda_device_count"] = torch.cuda.device_count()
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
    
    try:
        import ngsolve
        info["ngsolve_available"] = True
    except ImportError:
        info["ngsolve_available"] = False
    
    try:
        import pyvista
        info["pyvista_available"] = True
    except ImportError:
        info["pyvista_available"] = False
    
    return info


def log_experiment_info(model, config_info=None, filepath=None):
    """Log experiment information to a file.
    
    Args:
        model: PINN model
        config_info: Configuration information
        filepath: Log file path
        
    Returns:
        None
    """
    if filepath is None:
        from config import REPORTS_DIRECTORY
        os.makedirs(REPORTS_DIRECTORY, exist_ok=True)
        filepath = os.path.join(REPORTS_DIRECTORY, "experiment_log.txt")
    
    with open(filepath, 'w') as f:
        f.write("PINN Adaptive Mesh Experiment Log\n")
        f.write("="*50 + "\n\n")
        
        # System info
        system_info = get_system_info()
        f.write("System Information:\n")
        for key, value in system_info.items():
            f.write(f"  {key}: {value}\n")
        f.write("\n")
        
        # Configuration
        if config_info:
            f.write("Configuration:\n")
            for key, value in config_info.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")
        
        # Model summary
        f.write("Model Summary:\n")
        total_params = sum(p.numel() for p in model.parameters())
        f.write(f"  Total parameters: {total_params:,}\n")
        f.write(f"  Hidden size: {model.hidden_size}\n")
        f.write(f"  Loss weights: data={model.w_data}, interior={model.w_interior}, bc={model.w_bc}\n")
        
        # Training results
        if model.mesh_point_count_history:
            f.write(f"  Initial mesh points: {model.mesh_point_count_history[0]:,}\n")
            f.write(f"  Final mesh points: {model.mesh_point_count_history[-1]:,}\n")
        
        if model.total_error_history:
            f.write(f"  Final total error: {model.total_error_history[-1]:.6e}\n")
    
    print(f"Experiment log saved to {filepath}")


def cleanup_gif_png_files(directory=None, patterns=None, dry_run=False):
    """
    Clean up PNG files that were generated for GIF creation.
    
    This function removes intermediate PNG files created during adaptive mesh
    visualization, which can accumulate and take up significant disk space.
    
    Args:
        directory: Directory to clean (uses config DIRECTORY if None)
        patterns: List of filename patterns to match (uses defaults if None)
        dry_run: If True, only shows what would be deleted without deleting
        
    Returns:
        dict: Summary of cleanup results
    """
    if directory is None:
        directory = DIRECTORY
    
    # Default patterns for GIF-related PNG files
    if patterns is None:
        patterns = [
            "*residuals*.png",      # Residual field images  
            "*errors*.png",         # Error field images
            "*iter_*.png",          # Iteration-specific images
            "*_step_*.png",         # Step-specific images
            "*adaptation_*.png",    # Adaptation images
        ]
    
    import glob
    
    results = {
        "files_found": [],
        "files_deleted": [],
        "errors": [],
        "total_size_freed": 0
    }
    
    print(f"{'DRY RUN: ' if dry_run else ''}Cleaning up GIF PNG files in {directory}")
    
    # Find all matching files
    for pattern in patterns:
        search_path = os.path.join(directory, pattern)
        matching_files = glob.glob(search_path)
        results["files_found"].extend(matching_files)
    
    # Remove duplicates and sort
    results["files_found"] = sorted(list(set(results["files_found"])))
    
    if not results["files_found"]:
        print("No GIF PNG files found to clean up")
        return results
    
    print(f"Found {len(results['files_found'])} files matching patterns:")
    for pattern in patterns:
        print(f"  {pattern}")
    
    # Process each file
    for file_path in results["files_found"]:
        try:
            # Get file size
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                results["total_size_freed"] += file_size
                
                if dry_run:
                    print(f"  Would delete: {os.path.basename(file_path)} ({file_size} bytes)")
                else:
                    os.remove(file_path)
                    results["files_deleted"].append(file_path)
                    print(f"  Deleted: {os.path.basename(file_path)} ({file_size} bytes)")
            
        except Exception as e:
            error_msg = f"Error processing {file_path}: {e}"
            results["errors"].append(error_msg)
            print(f"  Error: {error_msg}")
    
    # Summary
    if dry_run:
        print("\nDry run complete:")
        print(f"  {len(results['files_found'])} files would be deleted")
        print(f"  {results['total_size_freed']} bytes would be freed")
    else:
        print("\nCleanup complete:")
        print(f"  {len(results['files_deleted'])} files deleted")
        print(f"  {results['total_size_freed']} bytes freed")
        if results["errors"]:
            print(f"  {len(results['errors'])} errors encountered")
    
    return results


def cleanup_all_temp_files(directory=None, dry_run=False):
    """
    Clean up all temporary files including PNG files, VTK exports, and cache files.
    
    Args:
        directory: Directory to clean (uses config DIRECTORY if None) 
        dry_run: If True, shows what would be deleted without deleting
        
    Returns:
        dict: Summary of cleanup results
    """
    if directory is None:
        directory = DIRECTORY
    
    temp_patterns = [
        # GIF-related PNG files
        "*residuals*.png",
        "*errors*.png", 
        "*iter_*.png",
        "*_step_*.png",
        "*adaptation_*.png",
        # VTK export files
        "*.vtu",
        "*.vtk", 
        "vtk_export*",
        # Cache files
        "*.pyc",
        "__pycache__",
    ]
    
    print(f"{'DRY RUN: ' if dry_run else ''}Cleaning up all temporary files...")
    
    # Clean PNG files first
    png_results = cleanup_gif_png_files(directory, patterns=temp_patterns[:5], dry_run=dry_run)
    
    # Clean other temp files
    import glob
    import shutil
    
    other_files = []
    for pattern in temp_patterns[5:]:
        search_path = os.path.join(directory, pattern)
        matching_files = glob.glob(search_path)
        other_files.extend(matching_files)
    
    total_cleaned = len(png_results["files_deleted"])
    
    for file_path in other_files:
        try:
            if dry_run:
                if os.path.isdir(file_path):
                    print(f"  Would remove directory: {os.path.basename(file_path)}")
                else:
                    print(f"  Would delete: {os.path.basename(file_path)}")
            else:
                if os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                    print(f"  Removed directory: {os.path.basename(file_path)}")
                else:
                    os.remove(file_path)
                    print(f"  Deleted: {os.path.basename(file_path)}")
                total_cleaned += 1
        except Exception as e:
            print(f"  Error processing {file_path}: {e}")
    
    print(f"\n{'Dry run' if dry_run else 'Cleanup'} complete: {total_cleaned} items processed")
    
    return png_results
