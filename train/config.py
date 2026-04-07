"""
Configuration file for PINN adaptive mesh training.
Contains all constants, hyperparameters, and configuration settings.
"""

import json
import os
import sys
import warnings


def _spec_from_argv():
    args = sys.argv[1:]
    try:
        spec_index = args.index("--spec")
    except ValueError:
        return None

    if spec_index + 1 >= len(args):
        return None

    spec_path = args[spec_index + 1]
    try:
        with open(spec_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _get_requested_device():
    env_requested = os.environ.get("PINN_DEVICE")
    if env_requested is not None and env_requested.strip():
        return env_requested.strip()

    spec_requested = None if _SPEC_ARGV is None else _SPEC_ARGV.get("device")
    if spec_requested is not None:
        spec_requested = str(spec_requested).strip()
        if spec_requested:
            os.environ["PINN_DEVICE"] = spec_requested
            return spec_requested

    return "auto"


def _get_requested_num_threads():
    env_threads = os.environ.get("PINN_NUM_THREADS")
    if env_threads is not None and env_threads.strip():
        try:
            return max(1, int(env_threads))
        except ValueError:
            return None

    if _SPEC_ARGV is None:
        return None

    spec_threads = _SPEC_ARGV.get("num_threads")
    if spec_threads is None:
        return None

    try:
        spec_threads = max(1, int(spec_threads))
    except (TypeError, ValueError):
        return None

    os.environ["PINN_NUM_THREADS"] = str(spec_threads)
    return spec_threads


_SPEC_ARGV = _spec_from_argv()
_EARLY_REQUESTED_DEVICE = _get_requested_device()
_REQUESTED_NUM_THREADS = _get_requested_num_threads()

if _REQUESTED_NUM_THREADS is not None:
    thread_vars = (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
    for var in thread_vars:
        os.environ[var] = str(_REQUESTED_NUM_THREADS)

# If the user requested CPU, hide CUDA before importing torch so the entire
# process stays on CPU instead of partially initializing a GPU context.
if _EARLY_REQUESTED_DEVICE.lower() == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch

if _REQUESTED_NUM_THREADS is not None:
    try:
        torch.set_num_threads(_REQUESTED_NUM_THREADS)
    except Exception:
        pass
    try:
        torch.set_num_interop_threads(max(1, min(4, _REQUESTED_NUM_THREADS)))
    except Exception:
        pass

# Suppress warnings
warnings.filterwarnings("ignore")


def _resolve_device_from_env():
    requested = _EARLY_REQUESTED_DEVICE.strip() or "auto"
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
    "num_threads": _REQUESTED_NUM_THREADS,
}

# Project configuration
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))

# Model hyperparameters
MODEL_CONFIG = {
    "hidden_size": 64,
    "num_data": 128,  # Labeled coarse-mesh batch size per optimizer step
    "num_bd": 1000,  # Size of boundary condition enforcement (how many boundary points to check)
    "w_data": 0.25,  # loss_data weight
    "w_interior": 1.0,  # loss_interior weight
    "w_bc": 1.0,  # loss_bc weight
}

# Training parameters
TRAINING_CONFIG = {
    "epochs": 100,  # number of epochs where the model is trained on data + interior points at each iteration before mesh refinement
    "iterations": 4,  # number of mesh refinements
    "lr": 1e-3,
    "optimizer": "Adam",
    "seed": None,
}

# Mesh parameters
MESH_CONFIG = {
    "maxh": 0.7,  # Initial mesh size
    "refinement_threshold": 0.7,  # Threshold for mesh refinement (fraction of max error)
    "reference_mesh_factor": 0.05,  # Factor for creating reference mesh (smaller = finer)
}

# Geometry parameters
GEOMETRY_CONFIG = {
    "base_l": 0.5,
    "base_w": 1.5,
    "offset": "auto",
    "domain_size": 5,
    "grid_n": 3,
    "pattern_scale": "auto",
    "circle_radius": 0.7,
    "cell_fill": 0.45,
}

# Visualization parameters
VIZ_CONFIG = {
    "image_size": 600,
    "gif_duration": 1000,
    "gif_loop": 0,
    # Optional fixed color ranges for consistency across plots
    # Set to a tuple (vmin, vmax) to fix colorbar range; leave as None for auto-scaling
    "residual_clim": (0.0, 30.0),  # e.g., (0.0, 1e-2)
    # Error images show pointwise squared error, not the normalized headline metric.
    # The previous 200.0 cap was from a much rougher regime and flattened current plots.
    "error_clim": (0.0, 5.0),
}

# Random point generation
RANDOM_CONFIG = {
    "default_point_count": 200,
    "domain_bounds": "auto",  # (min, max) for x and y coordinates; set to "auto" to use mesh bbox
    "log_sampling_stats": True,  # write sampling stats to reports/methods/<method>/sampling_stats.txt
}

# RAD (Residual-based Adaptive Distribution) parameters - Wu et al. 2022
RAD_CONFIG = {
    "k": 2.0,  # Exponent for residual weighting (higher = more focus on high-error regions)
    "c": 0.0,  # Regularization constant (higher = more uniform coverage)
    "num_candidates": 500,  # Size of candidate set for residual evaluation during lean iteration
    "resample_period": 2,  # Resample points every N iterations
}

# Quasi-random sampling parameters
QUASI_RANDOM_CONFIG = {
    "seed": 42,  # Seed for reproducible low-discrepancy sequences
}

# Hybrid adaptive refinement parameters
HYBRID_ADAPTIVE_CONFIG = {
    "anchor_count": 128,  # Fixed FEM-labeled anchor points sampled once per run
    "alpha": 1.0,  # Weight for normalized residual indicator
    "beta": 0.5,  # Weight for normalized anchor-error indicator
    "normalization_quantile": 0.95,  # Quantile used for robust clipping/normalization
    "refinement_threshold": 0.9,  # Hybrid-specific refinement aggressiveness
    "anchor_seed_offset": 7919,  # Offset from method seed for deterministic anchor sampling
}
