"""
Utility functions for PINN adaptive mesh training.
Contains helper functions and utilities used across different modules.
"""

import numpy as np
import random
import torch
import os
from config import DEVICE, REQUESTED_DEVICE


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
            "relative_l2_error_history": getattr(
                model, "relative_l2_error_history", []
            ),
            "total_error_rms_history": getattr(model, "total_error_rms_history", []),
            "relative_error_rms_history": getattr(
                model, "relative_error_rms_history", []
            ),
            "boundary_error_history": model.boundary_error_history,
            "train_loss_history": model.train_loss_history,
            "total_residual_history": model.total_residual_history,
            "boundary_residual_history": model.boundary_residual_history,
            "fixed_total_residual_history": getattr(
                model, "fixed_total_residual_history", []
            ),
            "relative_fixed_l2_residual_history": getattr(
                model, "relative_fixed_l2_residual_history", []
            ),
            "fixed_boundary_residual_history": getattr(
                model, "fixed_boundary_residual_history", []
            ),
            "fixed_rms_residual_history": getattr(
                model, "fixed_rms_residual_history", []
            ),
            "relative_fixed_rms_residual_history": getattr(
                model, "relative_fixed_rms_residual_history", []
            ),
            "mesh_point_history": model.mesh_point_history,
            "mesh_point_count_history": model.mesh_point_count_history,
            "iteration_point_count_history": getattr(
                model, "iteration_point_count_history", []
            ),
            "iteration_runtime_history": getattr(
                model, "iteration_runtime_history", []
            ),
            "cumulative_runtime_history": getattr(
                model, "cumulative_runtime_history", []
            ),
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
        if hasattr(model, "set_mesh_points"):
            model.set_mesh_points(checkpoint["mesh_x"], checkpoint["mesh_y"])
        else:
            model.mesh_x = torch.tensor(
                checkpoint["mesh_x"], dtype=torch.float32, device=DEVICE
            )
            model.mesh_y = torch.tensor(
                checkpoint["mesh_y"], dtype=torch.float32, device=DEVICE
            )
        model.total_error_history = checkpoint.get("total_error_history", [])
        model.relative_l2_error_history = checkpoint.get(
            "relative_l2_error_history", []
        )
        model.total_error_rms_history = checkpoint.get("total_error_rms_history", [])
        model.relative_error_rms_history = checkpoint.get(
            "relative_error_rms_history", []
        )
        model.boundary_error_history = checkpoint.get("boundary_error_history", [])
        model.train_loss_history = checkpoint.get("train_loss_history", [])
        model.total_residual_history = checkpoint.get("total_residual_history", [])
        model.boundary_residual_history = checkpoint.get(
            "boundary_residual_history", []
        )
        model.fixed_total_residual_history = checkpoint.get(
            "fixed_total_residual_history", []
        )
        model.relative_fixed_l2_residual_history = checkpoint.get(
            "relative_fixed_l2_residual_history", []
        )
        model.fixed_boundary_residual_history = checkpoint.get(
            "fixed_boundary_residual_history", []
        )
        model.fixed_rms_residual_history = checkpoint.get(
            "fixed_rms_residual_history", []
        )
        model.relative_fixed_rms_residual_history = checkpoint.get(
            "relative_fixed_rms_residual_history", []
        )
        model.mesh_point_history = checkpoint.get("mesh_point_history", [])
        model.mesh_point_count_history = checkpoint.get("mesh_point_count_history", [])
        model.iteration_point_count_history = checkpoint.get(
            "iteration_point_count_history", []
        )
        model.iteration_runtime_history = checkpoint.get(
            "iteration_runtime_history", []
        )
        model.cumulative_runtime_history = checkpoint.get(
            "cumulative_runtime_history", []
        )

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
    print("\n" + "=" * 50)
    print("PINN MODEL SUMMARY")
    print("=" * 50)

    # Model architecture
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(
        f"Architecture: {model.b1.in_features} -> {model.hidden_size} -> {model.hidden_size} -> 1"
    )
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Current mesh info
    if hasattr(model, "mesh_x") and model.mesh_x is not None:
        print(f"Current mesh points: {len(model.mesh_x):,}")

    # Training history
    if model.mesh_point_count_history:
        initial_points = model.mesh_point_count_history[0]
        current_points = model.mesh_point_count_history[-1]
        refinement_factor = current_points / initial_points if initial_points > 0 else 0
        print(
            f"Mesh refinement: {initial_points:,} -> {current_points:,} (×{refinement_factor:.2f})"
        )

    if model.total_error_history:
        final_error = model.total_error_history[-1]
        print(f"Final error integral: {final_error:.6e}")
    if getattr(model, "relative_l2_error_history", None):
        print(f"Final relative L2 error: {model.relative_l2_error_history[-1]:.6e}")
    if getattr(model, "relative_error_rms_history", None):
        print(
            f"Final relative RMS error: {model.relative_error_rms_history[-1]:.6e}"
        )

    if model.train_loss_history:
        print(f"Training epochs completed: {len(model.train_loss_history)}")

    print("=" * 50)


def get_system_info():
    """Get system information for debugging and logging.

    Returns:
        dict: System information
    """
    info = {
        "requested_device": REQUESTED_DEVICE,
        "device": str(DEVICE),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }

    if torch.cuda.is_available():
        info["cuda_device_count"] = torch.cuda.device_count()
        current_index = DEVICE.index if DEVICE.type == "cuda" else 0
        info["cuda_device_name"] = torch.cuda.get_device_name(current_index)
        info["cuda_selected_index"] = current_index

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
        from paths import reports_dir

        rdir = reports_dir()
        os.makedirs(rdir, exist_ok=True)
        filepath = os.path.join(rdir, "experiment_log.txt")

    with open(filepath, "w") as f:
        f.write("PINN Adaptive Mesh Experiment Log\n")
        f.write("=" * 50 + "\n\n")

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
        f.write(
            f"  Loss weights: data={model.w_data}, interior={model.w_interior}, bc={model.w_bc}\n"
        )

        # Training results
        if model.mesh_point_count_history:
            f.write(f"  Initial mesh points: {model.mesh_point_count_history[0]:,}\n")
            f.write(f"  Final mesh points: {model.mesh_point_count_history[-1]:,}\n")

        if model.total_error_history:
            f.write(f"  Final error integral: {model.total_error_history[-1]:.6e}\n")
        if getattr(model, "relative_l2_error_history", None):
            f.write(
                f"  Final relative L2 error: {model.relative_l2_error_history[-1]:.6e}\n"
            )
        if getattr(model, "relative_error_rms_history", None):
            f.write(
                "  Final relative RMS error: "
                f"{model.relative_error_rms_history[-1]:.6e}\n"
            )

    print(f"Experiment log saved to {filepath}")
