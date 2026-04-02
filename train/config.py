"""
Configuration file for PINN adaptive mesh training.
Contains all constants, hyperparameters, and configuration settings.
"""

import os
import torch
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")


def _resolve_device_from_env():
    requested = os.environ.get("PINN_DEVICE", "auto").strip() or "auto"
    lowered = requested.lower()

    if lowered == "auto":
        if torch.cuda.is_available():
            return requested, torch.device("cuda:0")
        return requested, torch.device("cpu")

    if lowered == "cpu":
        return requested, torch.device("cpu")

    if lowered == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "PINN_DEVICE requested 'mps' but PyTorch MPS is not available"
            )
        return requested, torch.device("mps")

    try:
        device = torch.device(requested)
    except Exception as exc:
        raise RuntimeError(f"Invalid PINN_DEVICE value '{requested}'") from exc

    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"PINN_DEVICE requested '{requested}' but CUDA is not available"
            )
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError(
                f"PINN_DEVICE requested '{requested}' but only "
                f"{torch.cuda.device_count()} CUDA device(s) are available"
            )

    return requested, device


# Device configuration
REQUESTED_DEVICE, DEVICE = _resolve_device_from_env()
RUNTIME_CONFIG = {
    "requested_device": REQUESTED_DEVICE,
    "active_device": str(DEVICE),
}

# Project configuration
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))

# Model hyperparameters
MODEL_CONFIG = {
    "hidden_size": 50,
    "num_data": 1000,  #  Size of data loss computation (how many FEM training points to use)
    "num_bd": 5000,  # Size of boundary condition enforcement (how many boundary points to check)
    "w_data": 1.0,  # loss_data weight
    "w_interior": 1.0,  # loss_interior weight
    "w_bc": 1.0,  # loss_bc weight
}

# Training parameters
TRAINING_CONFIG = {
    "epochs": 500,  # number of epochs where the model is trained on data + interior points at each iteration before mesh refinement
    "iterations": 10,  # number of mesh refinements
    "lr": 1e-3,
    "optimizer": "LBFGS",
    "seed": None,
}

# Mesh parameters
MESH_CONFIG = {
    "maxh": 0.5,  # Initial mesh size
    "refinement_threshold": 0.7,  # Threshold for mesh refinement (fraction of max error)
    "reference_mesh_factor": 0.01,  # Factor for creating reference mesh (smaller = finer)
}

# Geometry parameters
GEOMETRY_CONFIG = {
    "base_l": 0.5,
    "base_w": 1.5,
    "offset": "auto",
    "domain_size": 5,
    "grid_n": 6,
    "pattern_scale": "auto",
    "circle_radius": 1.0,
    "cell_fill": 0.6,
}

# Visualization parameters
VIZ_CONFIG = {
    "image_size": 600,
    "gif_duration": 1000,
    "gif_loop": 0,
    # Optional fixed color ranges for consistency across plots
    # Set to a tuple (vmin, vmax) to fix colorbar range; leave as None for auto-scaling
    "residual_clim": (0.0, 30.0),  # e.g., (0.0, 1e-2)
    "error_clim": (0.0, 200.0),  # e.g., (0.0, 1e-3)
}

# Random point generation
RANDOM_CONFIG = {
    "default_point_count": 200,
    "domain_bounds": "auto",  # (min, max) for x and y coordinates; set to "auto" to use mesh bbox
    "log_sampling_stats": True,  # write sampling stats to reports/point_sampling_stats.txt
}

# RAD (Residual-based Adaptive Distribution) parameters - Wu et al. 2022
RAD_CONFIG = {
    "k": 2.0,  # Exponent for residual weighting (higher = more focus on high-error regions)
    "c": 0.0,  # Regularization constant (higher = more uniform coverage)
    "num_candidates": 2000,  # Size of candidate set for residual evaluation (smaller than paper's 10k)
    "resample_period": 1,  # Resample points every N iterations
}

# Random-R (Random with Resampling) parameters
RANDOM_R_CONFIG = {
    "resample_period": 1,  # Resample points every N iterations (1 = every iteration)
}

# Quasi-random sampling parameters
QUASI_RANDOM_CONFIG = {
    "seed": 42,  # Seed for reproducible low-discrepancy sequences
}
