"""
Configuration file for PINN adaptive mesh training.
Contains all constants, hyperparameters, and configuration settings.
"""

import os
import torch
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Device configuration
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Directory configuration
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DIRECTORY = os.path.join(PROJECT_ROOT, "images")
REPORTS_DIRECTORY = os.path.join(PROJECT_ROOT, "reports")

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
    "offset": 3,
    "domain_size": 5,
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
