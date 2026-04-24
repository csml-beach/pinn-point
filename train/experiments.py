"""
High-level experiment runners for PINN adaptive mesh training.
Contains the main experiment orchestration functions.
"""

import copy
import csv
import itertools
import json
import os
import time

import torch
from ngsolve import H1, Mesh
from torch.utils.data import TensorDataset

from config import (
    DEVICE,
    GEOMETRY_CONFIG,
    HYBRID_ADAPTIVE_CONFIG,
    MESH_CONFIG,
    QUASI_RANDOM_CONFIG,
    RAD_CONFIG,
    TRAINING_CONFIG,
    VALIDATION_CONFIG,
)
from fem_solver import create_dataset, export_fem_solution, solve_FEM
from geometry import export_vertex_coordinates
from methods import get_method, list_methods
from paths import (
    comparison_images_dir,
    generate_run_id,
    method_reports_dir,
    reports_dir,
    set_active_run,
    write_run_manifest,
    write_run_metadata,
)
from pinn_model import FeedForward
from problems import get_problem
from training import train_model
from utils import (
    build_model_checkpoint,
    get_selected_history_value,
    print_model_summary,
    restore_model_state_from_checkpoint,
    set_global_seed,
)
from visualization import create_multi_method_visualizations


MESH_REFINEMENT_METHODS = {
    "adaptive",
    "adaptive_entropy_balanced",
    "adaptive_halton_base",
    "adaptive_persistent",
    "adaptive_power_tempered",
    "adaptive_power_tempered_beta25",
    "adaptive_power_tempered_beta30",
    "adaptive_hybrid_anchor",
}


def _build_problem(problem_name: str = "poisson", problem_kwargs: dict | None = None):
    kwargs = dict(problem_kwargs or {})
    if problem_name in {"poisson", "advection_diffusion"}:
        kwargs.setdefault("domain_size", GEOMETRY_CONFIG["domain_size"])
    return get_problem(problem_name, **kwargs)


def _clone_mesh(mesh):
    return Mesh(mesh.ngmesh.Copy())


def _method_seed(base_seed: int, method_name: str) -> int:
    offset = sum((idx + 1) * ord(ch) for idx, ch in enumerate(method_name))
    return (int(base_seed) + offset) % (2**31 - 1)


def _build_initial_model_state(problem, vertex_array, base_seed: int):
    set_global_seed(base_seed)
    mesh_x, mesh_y = vertex_array.T
    prototype = FeedForward(mesh_x=mesh_x, mesh_y=mesh_y, problem=problem).to(DEVICE)
    state = copy.deepcopy(prototype.state_dict())
    for mesh_buffer_key in ("mesh_x", "mesh_y", "mesh_t"):
        state.pop(mesh_buffer_key, None)
    del prototype
    return state


def _split_training_and_validation_dataset(dataset, seed: int, validation_config: dict):
    if not validation_config.get("enabled", True):
        return dataset, None

    if not hasattr(dataset, "tensors") or len(dataset.tensors) != 2:
        return dataset, None

    xy, u = dataset.tensors
    total_count = len(xy)
    if total_count < 2:
        return dataset, None

    holdout_fraction = float(validation_config.get("data_holdout_fraction", 0.0))
    if holdout_fraction <= 0.0:
        return dataset, None

    val_count = max(1, int(round(total_count * holdout_fraction)))
    val_count = min(val_count, total_count - 1)

    generator = torch.Generator()
    generator.manual_seed(int(seed) + 2027)
    permutation = torch.randperm(total_count, generator=generator)
    val_idx = permutation[:val_count]
    train_idx = permutation[val_count:]

    train_dataset = TensorDataset(xy[train_idx], u[train_idx])
    validation_dataset = TensorDataset(xy[val_idx], u[val_idx])
    if hasattr(dataset, "flux_supervision"):
        train_dataset.flux_supervision = dataset.flux_supervision
    return train_dataset, validation_dataset


def _augment_points_for_problem(
    problem,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    mesh,
    iteration: int,
    seed: int | None,
    purpose: str,
):
    if problem is None or not hasattr(problem, "augment_collocation_points"):
        return x, y, None

    augmented = problem.augment_collocation_points(
        x,
        y,
        mesh=mesh,
        iteration=int(iteration),
        seed=seed,
        purpose=purpose,
    )
    if not isinstance(augmented, tuple):
        raise ValueError("augment_collocation_points must return a tuple")
    if len(augmented) == 2:
        return augmented[0], augmented[1], None
    if len(augmented) == 3:
        return augmented
    raise ValueError("augment_collocation_points must return (x, y) or (x, y, t)")


def _build_fixed_residual_validation_points(
    problem, mesh, collocation_budget, seed: int, validation_config: dict
):
    if not validation_config.get("enabled", True):
        return None

    point_count = validation_config.get("interior_point_count")
    if point_count is None:
        point_count = collocation_budget
    point_count = max(1, int(point_count))

    method = _build_method_instance("halton", problem, method_seed=int(seed) + 4099)
    x, y = method.get_collocation_points(
        mesh,
        model=None,
        iteration=0,
        num_points=point_count,
    )
    return _augment_points_for_problem(
        problem,
        x,
        y,
        mesh=mesh,
        iteration=0,
        seed=int(seed) + 4099,
        purpose="validation",
    )


def _append_validation_history(model, validation_result):
    if validation_result is None:
        return
    model.validation_score_history.append(validation_result["validation_score"])
    model.validation_data_loss_history.append(validation_result["validation_data_loss"])
    model.validation_residual_loss_history.append(
        validation_result["validation_residual_loss"]
    )


def _resolve_validation_config(validation_options: dict | None = None):
    config = dict(VALIDATION_CONFIG)
    if validation_options:
        config.update(validation_options)
    return config


def _record_iteration_runtime(model, runtime_sec: float):
    runtime_sec = float(runtime_sec)
    cumulative = runtime_sec
    if getattr(model, "cumulative_runtime_history", None):
        cumulative += model.cumulative_runtime_history[-1]
    model.iteration_runtime_history.append(runtime_sec)
    model.cumulative_runtime_history.append(cumulative)


def _json_safe(value):
    if value is None:
        return None
    if isinstance(value, (str, bool, int, float)):
        return value
    try:
        return float(value)
    except Exception:
        return str(value)


def _record_method_iteration_log(model, method, iteration: int, mesh):
    if not hasattr(model, "method_iteration_logs"):
        model.method_iteration_logs = []
    try:
        raw_log = method.log_iteration(iteration, mesh, model) or {}
    except Exception as exc:
        raw_log = {
            "method": getattr(method, "name", "unknown"),
            "iteration": int(iteration),
            "log_error": str(exc),
        }
    normalized = {key: _json_safe(val) for key, val in raw_log.items()}
    normalized.setdefault("method", getattr(method, "name", "unknown"))
    normalized.setdefault("iteration", int(iteration))
    model.method_iteration_logs.append(normalized)


def _log_comparison_budget_policy(epochs: int, collocation_budget: int):
    print("\nComparison budget policy:")
    print(f"  Exact optimizer budget per iteration: {epochs} epochs for every method")
    print("  Adaptive-only extra fine-tuning: disabled")
    print(
        "  Fixed interior collocation budget: "
        f"{int(collocation_budget):,} points per iteration for every method"
    )
    print(
        "  Runtime metric scope: point selection/refinement, training, and evaluation"
    )


def _build_method_instance(method_name: str, problem, method_seed: int | None = None):
    domain_bounds = problem.get_sampling_bounds()

    if method_name == "adaptive":
        method = get_method(
            method_name,
            refinement_threshold=MESH_CONFIG["refinement_threshold"],
            seed=method_seed,
            area_exponent=0.5,
        )
        method.description = (
            "Residual-guided interior sampling with mixed area/density scoring"
        )
    elif method_name == "adaptive_persistent":
        method = get_method(
            method_name,
            refinement_threshold=MESH_CONFIG["refinement_threshold"],
            seed=method_seed,
            area_exponent=0.5,
            persistence_alpha=0.6,
        )
        method.description = (
            "Residual-guided interior sampling with persistence-weighted scoring"
        )
    elif method_name == "adaptive_halton_base":
        method = get_method(
            method_name,
            refinement_threshold=MESH_CONFIG["refinement_threshold"],
            seed=method_seed,
            domain_bounds=domain_bounds,
            area_exponent=0.5,
            persistence_alpha=0.5,
            backbone_fraction=0.5,
            warmup_iterations=1,
        )
        method.description = (
            "Halton-backed persistent adaptive residual sampling"
        )
    elif method_name == "adaptive_entropy_balanced":
        method = get_method(
            method_name,
            refinement_threshold=MESH_CONFIG["refinement_threshold"],
            seed=method_seed,
            area_exponent=0.5,
            persistence_alpha=0.5,
            lambda_min=0.25,
            lambda_max=0.75,
            rank_gamma=1.0,
            warmup_iterations=1,
        )
        method.description = (
            "Entropy-balanced rank-persistent adaptive residual sampling"
        )
    elif method_name in {
        "adaptive_power_tempered",
        "adaptive_power_tempered_beta25",
        "adaptive_power_tempered_beta30",
    }:
        beta_max_by_method = {
            "adaptive_power_tempered": 4.0,
            "adaptive_power_tempered_beta25": 2.5,
            "adaptive_power_tempered_beta30": 3.0,
        }
        beta_max = beta_max_by_method[method_name]
        method = get_method(
            method_name,
            refinement_threshold=MESH_CONFIG["refinement_threshold"],
            seed=method_seed,
            area_exponent=0.5,
            persistence_alpha=0.5,
            beta_min=1.0,
            beta_max=beta_max,
            coverage_area_exponent=0.5,
            warmup_iterations=1,
        )
        method.description = (
            f"Power-tempered rank-persistent adaptive residual sampling "
            f"(beta_max={beta_max:g})"
        )
    elif method_name == "adaptive_hybrid_anchor":
        method = get_method(
            method_name,
            refinement_threshold=HYBRID_ADAPTIVE_CONFIG.get(
                "refinement_threshold", MESH_CONFIG["refinement_threshold"]
            ),
            domain_bounds=domain_bounds,
            anchor_count=HYBRID_ADAPTIVE_CONFIG["anchor_count"],
            alpha=HYBRID_ADAPTIVE_CONFIG["alpha"],
            beta=HYBRID_ADAPTIVE_CONFIG["beta"],
            normalization_quantile=HYBRID_ADAPTIVE_CONFIG["normalization_quantile"],
            seed=(
                None
                if method_seed is None
                else method_seed + HYBRID_ADAPTIVE_CONFIG["anchor_seed_offset"]
            ),
        )
    elif method_name == "rad":
        method = get_method(
            method_name,
            domain_bounds=domain_bounds,
            k=RAD_CONFIG["k"],
            c=RAD_CONFIG["c"],
            num_candidates=RAD_CONFIG["num_candidates"],
            resample_period=RAD_CONFIG["resample_period"],
            seed=method_seed,
        )
    elif method_name in ("halton", "sobol"):
        method = get_method(
            method_name,
            domain_bounds=domain_bounds,
            seed=method_seed if method_seed is not None else QUASI_RANDOM_CONFIG["seed"],
        )
    else:
        try:
            method = get_method(method_name, domain_bounds=domain_bounds)
        except TypeError:
            method = get_method(method_name)

    method.set_problem(problem)
    return method


def run_mesh_refinement_method_training_fair(
    method_name: str,
    problem,
    initial_mesh,
    initial_fem_solution,
    shared_dataset,
    validation_dataset,
    validation_residual_points,
    validation_config,
    reference_mesh,
    reference_solution,
    num_adaptations,
    epochs,
    collocation_budget,
    export_images,
    learning_rate,
    initial_state_dict=None,
    method_seed=None,
):
    """
    Run a mesh-refining method using shared components for fair comparison.

    Args:
        initial_mesh: Initial mesh object
        shared_dataset: Shared training dataset (from initial mesh FEM solution)
        reference_mesh: High-fidelity reference mesh for error assessment
        reference_solution: High-fidelity reference solution for error assessment
        num_adaptations: Number of adaptation iterations
        epochs: Number of training epochs per iteration
        export_images: Whether to export visualization images

    Returns:
        FeedForward: Trained PINN model
    """
    from mesh_refinement import compute_model_error

    method = _build_method_instance(method_name, problem, method_seed)
    method.initialize_run_state(
        initial_mesh=initial_mesh,
        fem_solution=initial_fem_solution,
        method_seed=method_seed,
    )

    print(f"Starting {method_name} training on device:", DEVICE)
    print(f"Configuration: {num_adaptations} iterations, {epochs} epochs per iteration")
    if method_seed is not None:
        set_global_seed(method_seed)

    current_mesh = initial_mesh
    init_x, init_y = method.get_collocation_points(
        current_mesh,
        model=None,
        iteration=0,
        num_points=collocation_budget,
    )
    init_x, init_y, init_t = _augment_points_for_problem(
        problem,
        init_x,
        init_y,
        mesh=current_mesh,
        iteration=0,
        seed=method_seed,
        purpose="train",
    )
    print(f"Initial collocation budget: {len(init_x):,} points")

    model = FeedForward(
        mesh_x=init_x, mesh_y=init_y, mesh_t=init_t, problem=problem
    ).to(DEVICE)
    if initial_state_dict is not None:
        model.load_state_dict(copy.deepcopy(initial_state_dict), strict=False)
    model.method_name = method_name

    # Store reference solution in model for consistent error assessment
    model.reference_mesh = reference_mesh
    model.reference_solution = reference_solution

    # Initialize tracking
    initial_points = torch.stack([init_x, init_y], dim=1).detach().cpu().clone()
    model.mesh_point_history = [initial_points]
    model.mesh_point_count_history = [len(init_x)]

    # Adaptation iterations
    best_iteration_checkpoint = None
    best_iteration_score = None
    for iteration in range(num_adaptations):
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration + 1}/{num_adaptations}")
        print(f"{'='*60}")

        print(f"\n--- {method_name} Iteration {iteration + 1} ---")
        if iteration > 0:
            x, y = method.get_collocation_points(
                current_mesh,
                model=model,
                iteration=iteration,
                num_points=collocation_budget,
            )
            x, y, t = _augment_points_for_problem(
                problem,
                x,
                y,
                mesh=current_mesh,
                iteration=iteration,
                seed=method_seed,
                purpose="train",
            )
            model.set_mesh_points(x, y, mesh_t=t)
            sampled_points = torch.stack([x, y], dim=1).detach().cpu().clone()
            model.mesh_point_history.append(sampled_points)
            model.mesh_point_count_history.append(len(x))

        model.iteration_point_count_history.append(len(model.mesh_x))
        iter_start = time.perf_counter()

        validation_result = train_model(
            model,
            shared_dataset,
            epochs,
            optimizer=TRAINING_CONFIG["optimizer"],
            lr=learning_rate,
            validation_dataset=validation_dataset,
            validation_residual_points=validation_residual_points,
            restore_best_epoch_checkpoint=validation_config.get(
                "restore_best_epoch_checkpoint", True
            ),
        )

        compute_model_error(
            model,
            reference_mesh,
            reference_solution,
            export_images=export_images,
            iteration=iteration,
        )
        _append_validation_history(model, validation_result)

        current_mesh, _ = method.refine_mesh(current_mesh, model, iteration=iteration)
        _record_iteration_runtime(model, time.perf_counter() - iter_start)
        _record_method_iteration_log(model, method, iteration, current_mesh)
        if validation_result is not None:
            validation_score = validation_result["validation_score"]
            if (
                best_iteration_score is None
                or validation_score < best_iteration_score
            ):
                best_iteration_score = validation_score
                model.selected_iteration_index = iteration
                model.best_validation_score = validation_score
                best_iteration_checkpoint = build_model_checkpoint(
                    model,
                    additional_info={
                        "selected_iteration_index": iteration,
                        "best_validation_score": validation_score,
                    },
                )
        print(
            f"Iteration {iteration + 1} completed. Current mesh: {len(model.mesh_x):,} points"
        )

    if (
        best_iteration_checkpoint is not None
        and validation_config.get("restore_best_iteration_checkpoint", True)
    ):
        restore_model_state_from_checkpoint(model, best_iteration_checkpoint)
        print(
            "Restored best validation iteration checkpoint: "
            f"iteration={model.selected_iteration_index + 1}, "
            f"score={model.best_validation_score:.6e}"
        )

    print(f"\n{'='*60}")
    print(f"{method_name.upper()} TRAINING COMPLETED")
    print(f"{'='*60}")

    return model


def run_adaptive_training_fair(
    problem,
    initial_mesh,
    initial_fem_solution,
    shared_dataset,
    validation_dataset,
    validation_residual_points,
    validation_config,
    reference_mesh,
    reference_solution,
    num_adaptations,
    epochs,
    collocation_budget,
    export_images,
    learning_rate,
    initial_state_dict=None,
    method_seed=None,
):
    """Backward-compatible wrapper for the residual-only adaptive baseline."""
    return run_mesh_refinement_method_training_fair(
        method_name="adaptive",
        problem=problem,
        initial_mesh=initial_mesh,
        initial_fem_solution=initial_fem_solution,
        shared_dataset=shared_dataset,
        validation_dataset=validation_dataset,
        validation_residual_points=validation_residual_points,
        validation_config=validation_config,
        reference_mesh=reference_mesh,
        reference_solution=reference_solution,
        num_adaptations=num_adaptations,
        epochs=epochs,
        collocation_budget=collocation_budget,
        export_images=export_images,
        learning_rate=learning_rate,
        initial_state_dict=initial_state_dict,
        method_seed=method_seed,
    )


def run_method_training_fair(
    method_name: str,
    problem,
    initial_mesh,
    shared_dataset,
    validation_dataset,
    validation_residual_points,
    validation_config,
    reference_mesh,
    reference_solution,
    num_adaptations: int,
    epochs: int,
    collocation_budget: int,
    export_images: bool,
    learning_rate: float,
    initial_state_dict=None,
    method_seed=None,
):
    """
    Run training with any registered sampling method using shared components for fair comparison.

    This is the generalized training function that works with all registered
    fixed-budget sampling methods (halton, sobol, rad, adaptive, random).

    Args:
        method_name: Name of method from methods registry
        initial_mesh: Initial mesh object
        shared_dataset: Shared training dataset (from initial mesh FEM solution)
        reference_mesh: High-fidelity reference mesh for error assessment
        reference_solution: High-fidelity reference solution for error assessment
        num_adaptations: Number of adaptation iterations
        epochs: Number of training epochs per iteration
        export_images: Whether to export visualization images
        collocation_budget: Fixed interior collocation budget used by all methods

    Returns:
        FeedForward: Trained PINN model
    """
    from mesh_refinement import (
        compute_model_error,
        compute_random_model_error,
        compute_random_residuals,
    )
    print(f"\n{'='*60}")
    print(f"METHOD TRAINING: {method_name.upper()}")
    print(f"{'='*60}")
    if method_seed is not None:
        set_global_seed(method_seed)

    method = _build_method_instance(method_name, problem, method_seed)

    print(f"Method: {method.name} - {method.description}")

    # Get initial mesh coordinates
    vertex_array = export_vertex_coordinates(initial_mesh)
    mesh_x, mesh_y = vertex_array.T
    initial_point_count = len(mesh_x)

    print(f"Initial mesh: {initial_point_count:,} points")

    # Initialize the model on the actual iteration-0 collocation set so the
    # fixed-budget method history starts from the same accepted-point budget
    # used in training.
    initial_x, initial_y = method.get_collocation_points(
        initial_mesh,
        model=None,
        iteration=0,
        num_points=collocation_budget,
    )
    initial_x, initial_y, initial_t = _augment_points_for_problem(
        problem,
        initial_x,
        initial_y,
        mesh=initial_mesh,
        iteration=0,
        seed=method_seed,
        purpose="train",
    )
    print(f"Initial collocation budget: {len(initial_x):,} points")

    model = FeedForward(
        mesh_x=initial_x,
        mesh_y=initial_y,
        mesh_t=initial_t,
        problem=problem,
    ).to(DEVICE)
    if initial_state_dict is not None:
        model.load_state_dict(copy.deepcopy(initial_state_dict), strict=False)
    model.method_name = method_name
    model.reference_mesh = reference_mesh
    model.reference_solution = reference_solution

    # Initialize tracking histories from the actual method-specific collocation set,
    # not from the full initial mesh vertex list.
    initial_points = torch.stack([initial_x, initial_y], dim=1).detach().cpu().clone()
    model.mesh_point_history = [initial_points]
    model.mesh_point_count_history = [len(initial_x)]
    model.total_error_history = []
    model.boundary_error_history = []
    random_fe_space = None
    if method_name == "random" and not getattr(problem, "has_time_input", False):
        random_fe_space = H1(initial_mesh, order=1, dirichlet=".*")

    # Training iterations
    best_iteration_checkpoint = None
    best_iteration_score = None
    for iteration in range(num_adaptations):
        print(f"\n--- {method_name} Iteration {iteration + 1}/{num_adaptations} ---")
        iter_start = time.perf_counter()

        if iteration > 0:
            # Get collocation points using the method
            x, y = method.get_collocation_points(
                initial_mesh,
                model=model,
                iteration=iteration,
                num_points=collocation_budget,
            )
            x, y, t = _augment_points_for_problem(
                problem,
                x,
                y,
                mesh=initial_mesh,
                iteration=iteration,
                seed=method_seed,
                purpose="train",
            )

            # Update model's residual computation points
            model.set_mesh_points(x, y, mesh_t=t)

        actual_count = len(model.mesh_x)
        print(f"Using {actual_count:,} {method_name} points for residual computation")
        model.iteration_point_count_history.append(actual_count)

        # Train model
        validation_result = train_model(
            model,
            shared_dataset,
            epochs,
            optimizer=TRAINING_CONFIG["optimizer"],
            lr=learning_rate,
            validation_dataset=validation_dataset,
            validation_residual_points=validation_residual_points,
            restore_best_epoch_checkpoint=validation_config.get(
                "restore_best_epoch_checkpoint", True
            ),
        )

        # Compute and record error
        if method_name == "random" and random_fe_space is not None:
            compute_random_residuals(
                model,
                initial_mesh,
                random_fe_space,
                export_images=export_images,
                iteration=iteration,
            )
            if export_images:
                compute_random_model_error(
                    model,
                    reference_mesh,
                    reference_solution,
                    export_images=True,
                    iteration=iteration,
                )
            else:
                compute_model_error(
                    model,
                    reference_mesh,
                    reference_solution,
                    export_images=False,
                    iteration=iteration,
                )
        else:
            compute_model_error(
                model,
                reference_mesh,
                reference_solution,
                export_images=export_images,
                iteration=iteration,
            )

        _append_validation_history(model, validation_result)
        _record_iteration_runtime(model, time.perf_counter() - iter_start)
        _record_method_iteration_log(model, method, iteration, initial_mesh)

        # Record point count
        model.mesh_point_count_history.append(actual_count)
        if validation_result is not None:
            validation_score = validation_result["validation_score"]
            if (
                best_iteration_score is None
                or validation_score < best_iteration_score
            ):
                best_iteration_score = validation_score
                model.selected_iteration_index = iteration
                model.best_validation_score = validation_score
                best_iteration_checkpoint = build_model_checkpoint(
                    model,
                    additional_info={
                        "selected_iteration_index": iteration,
                        "best_validation_score": validation_score,
                    },
                )

    if (
        best_iteration_checkpoint is not None
        and validation_config.get("restore_best_iteration_checkpoint", True)
    ):
        restore_model_state_from_checkpoint(model, best_iteration_checkpoint)
        print(
            "Restored best validation iteration checkpoint: "
            f"iteration={model.selected_iteration_index + 1}, "
            f"score={model.best_validation_score:.6e}"
        )

    print(f"\n{method_name} training completed")
    return model


def run_complete_experiment(
    mesh_size=None,
    num_adaptations=None,
    epochs=None,
    export_images=False,
    create_gifs=True,
    generate_report=True,
    methods_to_run=None,
    problem_name: str = "poisson",
    problem_kwargs: dict | None = None,
    validation_options: dict | None = None,
    reference_mesh_factor: float | None = None,
    seed: int | None = None,
    learning_rate: float | None = None,
):
    """Run the complete PINN adaptive mesh experiment.

    Args:
        mesh_size: Initial mesh size parameter
        num_adaptations: Number of adaptation iterations
        epochs: Number of training epochs per iteration
        export_images: Whether to export visualization images
        create_gifs: Whether to create animated GIFs
        generate_report: Whether to generate summary report
        methods_to_run: List of methods to test ['adaptive', 'random', 'gradient_based', 'ml_guided']

    Returns:
        dict: Dictionary of trained models keyed by method name
    """
    if mesh_size is None:
        mesh_size = MESH_CONFIG["maxh"]
    if num_adaptations is None:
        num_adaptations = TRAINING_CONFIG["iterations"]
    if epochs is None:
        epochs = TRAINING_CONFIG["epochs"]
    if learning_rate is None:
        learning_rate = float(TRAINING_CONFIG["lr"])
    if methods_to_run is None:
        methods_to_run = ["adaptive", "random"]  # Default to current methods
    if epochs is None:
        epochs = TRAINING_CONFIG["epochs"]
    problem = _build_problem(problem_name, problem_kwargs)
    validation_config = _resolve_validation_config(validation_options)
    if reference_mesh_factor is None:
        reference_mesh_factor = MESH_CONFIG["reference_mesh_factor"]
    if seed is None:
        cfg_seed = TRAINING_CONFIG.get("seed")
        if cfg_seed is not None:
            seed = int(cfg_seed)
        else:
            seed = int(torch.initial_seed() % (2**31 - 1))
    else:
        seed = int(seed)

    print(f"\n{'='*80}")
    print("STARTING COMPLETE PINN ADAPTIVE MESH EXPERIMENT")
    print(f"{'='*80}")
    print(f"Mesh size: {mesh_size}")
    print(f"Iterations: {num_adaptations}")
    print(f"Epochs per iteration: {epochs}")
    print(f"Export images: {export_images}")
    print(f"Methods to test: {', '.join(methods_to_run)}")
    print(f"Problem: {problem.name}")
    print(f"Base seed: {seed}")
    print(f"Device: {DEVICE}")
    print(f"Learning rate: {learning_rate}")

    # Create shared components for fair comparison
    print("\n" + "=" * 60)
    print("CREATING SHARED COMPONENTS FOR FAIR COMPARISON")
    print("=" * 60)

    # 1. Create initial mesh and training dataset (shared by all methods)
    print("Creating initial mesh and training dataset...")
    initial_mesh = problem.create_mesh(maxh=mesh_size)
    gfu = None
    fes = None
    shared_training_dataset = problem.create_training_dataset(initial_mesh, seed=seed)
    if shared_training_dataset is None:
        gfu, fes = solve_FEM(initial_mesh, problem=problem)
        vertex_array = export_vertex_coordinates(initial_mesh)
        solution_array = export_fem_solution(initial_mesh, gfu, problem=problem)
        shared_training_dataset = create_dataset(vertex_array, solution_array)
    else:
        gfu, fes = solve_FEM(initial_mesh, problem=problem)
        vertex_array = export_vertex_coordinates(initial_mesh)

    training_dataset, validation_dataset = _split_training_and_validation_dataset(
        shared_training_dataset, seed, validation_config
    )
    mesh_x, mesh_y = vertex_array.T
    shared_label_count = len(training_dataset)
    print(f"Shared training dataset: {shared_label_count:,} labels")
    collocation_budget = problem.get_collocation_budget(
        initial_mesh,
        vertex_array,
        training_dataset=training_dataset,
    )
    if collocation_budget is None:
        collocation_budget = len(mesh_x)
    collocation_budget = max(1, int(collocation_budget))
    _log_comparison_budget_policy(epochs, collocation_budget)
    validation_residual_points = _build_fixed_residual_validation_points(
        problem, initial_mesh, collocation_budget, seed, validation_config
    )
    if validation_dataset is not None:
        print(
            "Validation policy: "
            f"{len(training_dataset)} coarse training labels + "
            f"{len(validation_dataset)} held-out validation labels, "
            f"{len(validation_residual_points[0]) if validation_residual_points else 0} "
            "fixed interior residual-validation points"
        )
        if not validation_config.get("restore_best_epoch_checkpoint", True):
            print("  Epoch-level best-checkpoint restore: disabled")
        if not validation_config.get("restore_best_iteration_checkpoint", True):
            print("  Iteration-level best-checkpoint restore: disabled")
    initial_model_state = _build_initial_model_state(problem, vertex_array, seed)

    # 2. Create high-fidelity reference solution (shared by both methods)
    print("Creating high-fidelity reference solution...")
    from mesh_refinement import create_reference_solution

    reference_mesh, reference_solution = create_reference_solution(
        problem, mesh_size_factor=reference_mesh_factor
    )
    ref_vertex_count = len(export_vertex_coordinates(reference_mesh))
    print(f"Reference solution: {ref_vertex_count:,} points")
    print("This reference solution will be used for all error computations")

    # Dictionary to store all trained models
    trained_models = {}
    execution_order = list(methods_to_run)

    # Run each requested method with shared components
    for method in execution_order:
        print(f"\n{'='*60}")
        print(f"RUNNING METHOD: {method.upper()}")
        print(f"{'='*60}")
        method_seed = _method_seed(seed, method)
        method_mesh = _clone_mesh(initial_mesh)

        if method in MESH_REFINEMENT_METHODS:
            print(f"Starting {method} mesh-refinement training...")
            model = run_mesh_refinement_method_training_fair(
                method_name=method,
                problem=problem,
                initial_mesh=method_mesh,
                initial_fem_solution=gfu,
                shared_dataset=training_dataset,
                validation_dataset=validation_dataset,
                validation_residual_points=validation_residual_points,
                validation_config=validation_config,
                reference_mesh=reference_mesh,
                reference_solution=reference_solution,
                num_adaptations=num_adaptations,
                epochs=epochs,
                collocation_budget=collocation_budget,
                export_images=export_images,
                learning_rate=learning_rate,
                initial_state_dict=initial_model_state,
                method_seed=method_seed,
            )
            trained_models[method] = model

        elif method in list_methods():
            print(f"Starting {method} training...")
            model = run_method_training_fair(
                method_name=method,
                problem=problem,
                initial_mesh=method_mesh,
                shared_dataset=training_dataset,
                validation_dataset=validation_dataset,
                validation_residual_points=validation_residual_points,
                validation_config=validation_config,
                reference_mesh=reference_mesh,
                reference_solution=reference_solution,
                num_adaptations=num_adaptations,
                epochs=epochs,
                collocation_budget=collocation_budget,
                export_images=export_images,
                learning_rate=learning_rate,
                initial_state_dict=initial_model_state,
                method_seed=method_seed,
            )
            trained_models[method] = model

        else:
            print(f"Warning: Unknown method '{method}' - skipping")
            print(f"Available methods: {list_methods()}")

    # Print summaries for all models
    for method_name, model in trained_models.items():
        print(f"\n{method_name.title()} Model Summary:")
        print_model_summary(model)

    if generate_report and trained_models:
        try:
            write_multi_method_histories_csv(trained_models)
        except Exception as e:
            print(f"Warning: Failed to write multi-method histories CSV: {e}")
        try:
            print("\nGenerating multi-method visualizations...")
            create_multi_method_visualizations(
                trained_models,
                dataset_size=len(training_dataset),
                output_dir=comparison_images_dir(),
                include_gifs=create_gifs and export_images,
                cleanup_pngs=True,
            )
        except Exception as e:
            print(f"Warning: Failed to create multi-method visualizations: {e}")
    try:
        write_method_report_bundle(trained_models)
    except Exception as e:
        print(f"Warning: Failed to write per-method report bundle: {e}")
    try:
        write_run_manifest(
            methods=sorted(trained_models.keys()),
            extra={
                "problem_name": problem.name,
                "seed": seed,
                "mesh_size": mesh_size,
                "iterations": num_adaptations,
                "epochs": epochs,
                "collocation_budget": collocation_budget,
                "training_dataset_size": len(training_dataset),
                "validation_dataset_size": (
                    len(validation_dataset) if validation_dataset is not None else 0
                ),
                "validation_residual_point_count": (
                    len(validation_residual_points[0])
                    if validation_residual_points is not None
                    else 0
                ),
                "validation_config": validation_config,
                "learning_rate": float(learning_rate),
                "export_images": export_images,
                "generate_report": generate_report,
            },
        )
    except Exception as e:
        print(f"Warning: Failed to write run manifest: {e}")

    print(f"\n{'='*80}")
    print("EXPERIMENT COMPLETED SUCCESSFULLY")
    print(f"{'='*80}")

    return trained_models


def write_multi_method_histories_csv(trained_models: dict):
    """Write training histories for all methods to a single CSV file.

    Args:
        trained_models: Dictionary mapping method names to trained models
    """
    output_path = os.path.join(reports_dir(), "all_methods_histories.csv")

    # Collect all data
    rows = []
    for method_name, model in trained_models.items():
        rows.extend(_collect_method_history_rows(method_name, model))

    _write_history_csv(rows, output_path)

    print(f"Multi-method histories written to: {output_path}")


def _history_csv_fieldnames():
    return [
        "method",
        "iteration",
        "total_error",
        "relative_l2_error",
        "total_error_rms",
        "relative_error_rms",
        "boundary_error",
        "fixed_total_residual",
        "relative_fixed_l2_residual",
        "fixed_boundary_residual",
        "fixed_rms_residual",
        "relative_fixed_rms_residual",
        "validation_score",
        "validation_data_loss",
        "validation_residual_loss",
        "is_selected_checkpoint",
        "point_count",
        "iteration_runtime_sec",
        "cumulative_runtime_sec",
    ]


def _collect_method_history_rows(method_name: str, model) -> list[dict]:
    total_errors = getattr(model, "total_error_history", [])
    relative_l2_errors = getattr(model, "relative_l2_error_history", [])
    total_error_rms = getattr(model, "total_error_rms_history", [])
    relative_error_rms = getattr(model, "relative_error_rms_history", [])
    boundary_errors = getattr(model, "boundary_error_history", [])
    fixed_total_residuals = getattr(model, "fixed_total_residual_history", [])
    relative_fixed_l2_residuals = getattr(
        model, "relative_fixed_l2_residual_history", []
    )
    fixed_boundary_residuals = getattr(model, "fixed_boundary_residual_history", [])
    fixed_rms_residuals = getattr(model, "fixed_rms_residual_history", [])
    relative_fixed_rms_residuals = getattr(
        model, "relative_fixed_rms_residual_history", []
    )
    validation_scores = getattr(model, "validation_score_history", [])
    validation_data_losses = getattr(model, "validation_data_loss_history", [])
    validation_residual_losses = getattr(
        model, "validation_residual_loss_history", []
    )
    iteration_point_counts = getattr(model, "iteration_point_count_history", [])
    point_counts = getattr(model, "mesh_point_count_history", [])
    iteration_runtime = getattr(model, "iteration_runtime_history", [])
    cumulative_runtime = getattr(model, "cumulative_runtime_history", [])
    selected_iteration_index = getattr(model, "selected_iteration_index", None)

    num_rows = max(
        len(total_errors),
        len(relative_l2_errors),
        len(total_error_rms),
        len(relative_error_rms),
        len(boundary_errors),
        len(fixed_total_residuals),
        len(relative_fixed_l2_residuals),
        len(fixed_boundary_residuals),
        len(fixed_rms_residuals),
        len(relative_fixed_rms_residuals),
        len(validation_scores),
        len(validation_data_losses),
        len(validation_residual_losses),
        len(iteration_point_counts),
        len(iteration_runtime),
        len(cumulative_runtime),
        max(len(point_counts) - 1, 0),
    )

    rows = []
    for i in range(num_rows):
        if i < len(iteration_point_counts):
            point_count = iteration_point_counts[i]
        elif i < len(point_counts):
            point_count = point_counts[i]
        else:
            point_count = None
        rows.append(
            {
                "method": method_name,
                "iteration": i,
                "total_error": total_errors[i] if i < len(total_errors) else None,
                "relative_l2_error": (
                    relative_l2_errors[i] if i < len(relative_l2_errors) else None
                ),
                "total_error_rms": (
                    total_error_rms[i] if i < len(total_error_rms) else None
                ),
                "relative_error_rms": (
                    relative_error_rms[i] if i < len(relative_error_rms) else None
                ),
                "boundary_error": (
                    boundary_errors[i] if i < len(boundary_errors) else None
                ),
                "fixed_total_residual": (
                    fixed_total_residuals[i] if i < len(fixed_total_residuals) else None
                ),
                "relative_fixed_l2_residual": (
                    relative_fixed_l2_residuals[i]
                    if i < len(relative_fixed_l2_residuals)
                    else None
                ),
                "fixed_boundary_residual": (
                    fixed_boundary_residuals[i]
                    if i < len(fixed_boundary_residuals)
                    else None
                ),
                "fixed_rms_residual": (
                    fixed_rms_residuals[i] if i < len(fixed_rms_residuals) else None
                ),
                "relative_fixed_rms_residual": (
                    relative_fixed_rms_residuals[i]
                    if i < len(relative_fixed_rms_residuals)
                    else None
                ),
                "validation_score": (
                    validation_scores[i] if i < len(validation_scores) else None
                ),
                "validation_data_loss": (
                    validation_data_losses[i]
                    if i < len(validation_data_losses)
                    else None
                ),
                "validation_residual_loss": (
                    validation_residual_losses[i]
                    if i < len(validation_residual_losses)
                    else None
                ),
                "is_selected_checkpoint": (
                    selected_iteration_index is not None and i == selected_iteration_index
                ),
                "point_count": point_count,
                "iteration_runtime_sec": (
                    iteration_runtime[i] if i < len(iteration_runtime) else None
                ),
                "cumulative_runtime_sec": (
                    cumulative_runtime[i] if i < len(cumulative_runtime) else None
                ),
            }
        )
    return rows


def _write_history_csv(rows: list[dict], output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_history_csv_fieldnames())
        writer.writeheader()
        writer.writerows(rows)


def _method_diagnostics_payload(method_name: str, model) -> dict:
    def _selected(history_name: str):
        return _json_safe(get_selected_history_value(model, history_name))

    return {
        "method": method_name,
        "selected_iteration_index": getattr(model, "selected_iteration_index", None),
        "best_validation_score": _json_safe(
            getattr(model, "best_validation_score", None)
        ),
        "final_metrics": {
            "total_error": _selected("total_error_history"),
            "relative_l2_error": _selected("relative_l2_error_history"),
            "total_error_rms": _selected("total_error_rms_history"),
            "relative_error_rms": _selected("relative_error_rms_history"),
            "boundary_error": _selected("boundary_error_history"),
            "fixed_total_residual": _selected("fixed_total_residual_history"),
            "relative_fixed_l2_residual": _selected(
                "relative_fixed_l2_residual_history"
            ),
            "fixed_boundary_residual": _selected("fixed_boundary_residual_history"),
            "fixed_rms_residual": _selected("fixed_rms_residual_history"),
            "relative_fixed_rms_residual": _selected(
                "relative_fixed_rms_residual_history"
            ),
            "validation_score": _selected("validation_score_history"),
            "validation_data_loss": _selected("validation_data_loss_history"),
            "validation_residual_loss": _selected(
                "validation_residual_loss_history"
            ),
            "point_count": _json_safe(
                len(model.mesh_x) if hasattr(model, "mesh_x") else None
            ),
            "cumulative_runtime_sec": _selected("cumulative_runtime_history"),
        },
        "history_lengths": {
            "total_error": len(getattr(model, "total_error_history", []) or []),
            "relative_l2_error": len(
                getattr(model, "relative_l2_error_history", []) or []
            ),
            "total_error_rms": len(
                getattr(model, "total_error_rms_history", []) or []
            ),
            "relative_error_rms": len(
                getattr(model, "relative_error_rms_history", []) or []
            ),
            "fixed_total_residual": len(
                getattr(model, "fixed_total_residual_history", []) or []
            ),
            "relative_fixed_l2_residual": len(
                getattr(model, "relative_fixed_l2_residual_history", []) or []
            ),
            "relative_fixed_rms_residual": len(
                getattr(model, "relative_fixed_rms_residual_history", []) or []
            ),
            "point_count": len(getattr(model, "mesh_point_count_history", []) or []),
            "iteration_runtime": len(
                getattr(model, "iteration_runtime_history", []) or []
            ),
            "iteration_diagnostics": len(
                getattr(model, "method_iteration_logs", []) or []
            ),
        },
        "latest_iteration_diagnostics": (
            (getattr(model, "method_iteration_logs", []) or [])[-1]
            if (getattr(model, "method_iteration_logs", []) or [])
            else None
        ),
        "artifacts": {
            "history_csv": os.path.join(method_reports_dir(method_name), "history.csv"),
            "iteration_diagnostics_csv": os.path.join(
                method_reports_dir(method_name), "iteration_diagnostics.csv"
            ),
            "diagnostics_json": os.path.join(
                method_reports_dir(method_name), "diagnostics.json"
            ),
        },
    }


def write_method_report_bundle(trained_models: dict):
    """Write per-method histories and diagnostics under reports/methods/<name>/."""
    for method_name, model in sorted(trained_models.items()):
        method_dir = method_reports_dir(method_name)
        history_path = os.path.join(method_dir, "history.csv")
        iteration_log_path = os.path.join(method_dir, "iteration_diagnostics.csv")
        diagnostics_path = os.path.join(method_dir, "diagnostics.json")
        rows = _collect_method_history_rows(method_name, model)
        _write_history_csv(rows, history_path)
        _write_generic_csv(getattr(model, "method_iteration_logs", []) or [], iteration_log_path)
        with open(diagnostics_path, "w") as f:
            json.dump(_method_diagnostics_payload(method_name, model), f, indent=2)
        print(f"Per-method report bundle written to: {method_dir}")


def _write_generic_csv(rows: list[dict], output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if not rows:
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["method", "iteration"])
        return

    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_hyperparameter_study(
    grid: dict | None = None, export_images: bool = False
) -> dict:
    """Run a flexible hyperparameter study by overriding config values.

    Grid keys use dotted paths into config dicts, e.g.:
      - "MODEL_CONFIG.hidden_size": [32, 64]
      - "TRAINING_CONFIG.lr": [1e-3, 3e-4]
      - "MODEL_CONFIG.w_interior": [1.0, 2.0]
      - "MESH_CONFIG.maxh": [0.5, 0.7]

    For each combination, we:
      - Apply overrides in-place to config dicts
      - Seed RNG (fixed if provided in TRAINING_CONFIG, else per-run random)
      - Create a run with metadata and histories
      - Restore original configs after completion

    Returns a dict mapping a short config tag to run details.
    """

    # Default grid if none provided
    if grid is None:
        grid = {
            "TRAINING_CONFIG.lr": [1e-3, 5e-4, 1e-4],
        }

    # Helpers to resolve and set values in config dicts
    def _get_target(parts):
        top, key = parts
        if top == "MODEL_CONFIG":
            return MODEL_CONFIG, key
        if top == "TRAINING_CONFIG":
            return TRAINING_CONFIG, key
        if top == "MESH_CONFIG":
            return MESH_CONFIG, key
        if top == "RANDOM_CONFIG":
            return RANDOM_CONFIG, key
        if top == "VIZ_CONFIG":
            return VIZ_CONFIG, key
        raise KeyError(f"Unsupported config root: {top}")

    def _fmt_val(v):
        if isinstance(v, float):
            return f"{v:g}" if (abs(v) >= 1e-3 and abs(v) < 1e3) else f"{v:.0e}"
        return str(v)

    def _abbr(path):
        mapping = {
            "MODEL_CONFIG.hidden_size": "hs",
            "MODEL_CONFIG.w_interior": "win",
            "MODEL_CONFIG.w_data": "wd",
            "MODEL_CONFIG.w_bc": "wbc",
            "TRAINING_CONFIG.lr": "lr",
            "TRAINING_CONFIG.epochs": "e",
            "TRAINING_CONFIG.iterations": "it",
            "TRAINING_CONFIG.optimizer": "opt",
            "MESH_CONFIG.maxh": "m",
            "MESH_CONFIG.refinement_threshold": "thr",
            "RANDOM_CONFIG.default_point_count": "rpc",
        }
        return mapping.get(path, path.split(".")[-1])

    # Snapshot originals to restore later
    originals = {
        "MODEL_CONFIG": MODEL_CONFIG.copy(),
        "TRAINING_CONFIG": TRAINING_CONFIG.copy(),
        "MESH_CONFIG": MESH_CONFIG.copy(),
        "RANDOM_CONFIG": RANDOM_CONFIG.copy(),
        "VIZ_CONFIG": VIZ_CONFIG.copy(),
    }

    # Build cartesian product of grid values
    items = list(grid.items())
    keys = [k for k, _ in items]
    values_lists = [v for _, v in items]
    combos = list(itertools.product(*values_lists))

    results: dict = {}

    print(f"\n{'='*80}")
    print("STARTING HYPERPARAMETER STUDY")
    print(f"{'='*80}")
    print("Grid:")
    for k, v in grid.items():
        print(f"  - {k}: {v}")

    for combo in combos:
        # Apply overrides
        overrides = {}
        for path, val in zip(keys, combo):
            parts = path.split(".")
            if len(parts) != 2:
                raise ValueError(f"Invalid grid key '{path}', expected 'CONFIG.key'")
            dct, key = _get_target(parts)
            overrides[path] = val
            dct[key] = val

        # Seed: fixed if present, else random per run
        seed = TRAINING_CONFIG.get("seed")
        if seed is None:
            import os as _os

            seed = int.from_bytes(_os.urandom(4), "little")
        set_global_seed(seed)

        # Build a short tag for the run id
        tag_parts = [f"{_abbr(k)}{_fmt_val(v)}" for k, v in overrides.items()]
        tag = "hps-" + "-".join(tag_parts) + f"-seed{seed}"

        # Create run and write metadata
        run_id = generate_run_id(tag)
        run_paths = set_active_run(run_id)
        write_run_metadata(
            {
                "phase": "start",
                "study": "hparams",
                "overrides": overrides,
                "seed": seed,
            }
        )

        try:
            # Pass current mesh/epochs if they were overridden; otherwise defaults inside runner apply
            models = run_complete_experiment(
                mesh_size=MESH_CONFIG.get("maxh"),
                num_adaptations=TRAINING_CONFIG.get("iterations"),
                epochs=TRAINING_CONFIG.get("epochs"),
                export_images=export_images,
                create_gifs=False,
                generate_report=False,
                seed=seed,
            )

            adaptive_model = (
                models.get("adaptive") if isinstance(models, dict) else None
            )
            random_model = models.get("random") if isinstance(models, dict) else None

            if isinstance(models, dict):
                try:
                    write_multi_method_histories_csv(models)
                except Exception as e:
                    print(
                        "Warning: Failed to write multi-method histories CSV "
                        f"for {run_id}: {e}"
                    )

            status = "ok"

        except Exception as e:
            print(f"Error in hparams combo {overrides}: {e}")
            status = f"error: {e}"

        finally:
            write_run_metadata(
                {
                    "phase": "end",
                    "study": "hparams",
                    "overrides": overrides,
                    "seed": seed,
                    "status": status,
                }
            )

        results[run_id] = {
            "run_id": run_id,
            "root": run_paths["root"],
            "overrides": overrides,
            "seed": seed,
            "status": status,
        }

        # Restore configs before next combo
    MODEL_CONFIG.clear()
    MODEL_CONFIG.update(originals["MODEL_CONFIG"])  # type: ignore
    TRAINING_CONFIG.clear()
    TRAINING_CONFIG.update(originals["TRAINING_CONFIG"])  # type: ignore
    MESH_CONFIG.clear()
    MESH_CONFIG.update(originals["MESH_CONFIG"])  # type: ignore
    RANDOM_CONFIG.clear()
    RANDOM_CONFIG.update(originals["RANDOM_CONFIG"])  # type: ignore
    VIZ_CONFIG.clear()
    VIZ_CONFIG.update(originals["VIZ_CONFIG"])  # type: ignore

    print(f"\n{'='*80}")
    print("HYPERPARAMETER STUDY COMPLETED")
    print(f"{'='*80}")

    return results
