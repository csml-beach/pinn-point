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
DIRECTORY = os.path.join(os.path.dirname(os.path.dirname(__file__)), "images")

# Model hyperparameters
MODEL_CONFIG = {
    "hidden_size": 50,
    "num_data": 1000,  #  Size of data loss computation (how many FEM training points to use)
    "num_bd": 5000,   # Size of boundary condition enforcement (how many boundary points to check)
    "w_data": 1.0,      # loss_data weight
    "w_interior": 1.0,   # loss_interior weight
    "w_bc": 1.0,        # loss_bc weight
}

# Training parameters
TRAINING_CONFIG = {
    "epochs": 100,
    "iterations": 10,
    "lr": 1e-3,
    "optimizer": "Adam",
}

# Mesh parameters
MESH_CONFIG = {
    "maxh": 0.5,                    # Initial mesh size
    "refinement_threshold": 0.7,     # Threshold for mesh refinement (fraction of max error)
    "reference_mesh_factor": 0.02,   # Factor for creating reference mesh (smaller = finer)
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
}

# Random point generation
RANDOM_CONFIG = {
    "default_point_count": 1000,
    "domain_bounds": (0, 5),  # (min, max) for x and y coordinates
}
