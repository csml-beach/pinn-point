"""
High-level experiment runners for PINN adaptive mesh training.
Contains the main experiment orchestration functions.
"""

import copy
import itertools
import os
import time

import torch
from ngsolve import H1, Mesh

from config import (
    DEVICE,
    GEOMETRY_CONFIG,
    MESH_CONFIG,
    MODEL_CONFIG,
    RANDOM_CONFIG,
    TRAINING_CONFIG,
    VIZ_CONFIG,
)
from fem_solver import create_dataset, export_fem_solution, solve_FEM
from geometry import export_vertex_coordinates
from methods import get_method, list_methods
from paths import generate_run_id, images_dir, reports_dir, set_active_run, write_run_metadata
from pinn_model import FeedForward
from problems import get_problem
from training import train_model
from utils import print_model_summary, set_global_seed
from visualization import create_multi_method_visualizations


def _build_problem(problem_name: str = "poisson", problem_kwargs: dict | None = None):
    kwargs = dict(problem_kwargs or {})
    if problem_name == "poisson":
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
    del prototype
    return state


def _record_iteration_runtime(model, runtime_sec: float):
    runtime_sec = float(runtime_sec)
    cumulative = runtime_sec
    if getattr(model, "cumulative_runtime_history", None):
        cumulative += model.cumulative_runtime_history[-1]
    model.iteration_runtime_history.append(runtime_sec)
    model.cumulative_runtime_history.append(cumulative)


def _log_comparison_budget_policy(epochs: int, methods_to_run: list[str]):
    point_budget_source = (
        "adaptive iteration-start point counts"
        if "adaptive" in methods_to_run
        else "fixed per-method collocation counts"
    )
    print("\nComparison budget policy:")
    print(f"  Exact optimizer budget per iteration: {epochs} epochs for every method")
    print("  Adaptive-only extra fine-tuning: disabled")
    print(f"  Point-budget source: {point_budget_source}")
    print(
        "  Runtime metric scope: point selection/refinement, training, and evaluation"
    )


def run_adaptive_training_fair(
    problem,
    initial_mesh,
    shared_dataset,
    reference_mesh,
    reference_solution,
    num_adaptations,
    epochs,
    export_images,
    initial_state_dict=None,
    method_seed=None,
):
    """
    Run adaptive mesh training using shared components for fair comparison.

    Args:
        initial_mesh: Initial mesh object
        shared_dataset: Shared training dataset (from initial mesh FEM solution)
        reference_mesh: High-fidelity reference mesh for error assessment
        reference_solution: High-fidelity reference solution for error assessment
        num_adaptations: Number of adaptation iterations
        epochs: Number of training epochs per iteration
        export_images: Whether to export visualization images

    Returns:
        FeedForward: Trained adaptive PINN model
    """
    from mesh_refinement import adapt_mesh_and_train

    print("Starting adaptive training on device:", DEVICE)
    print(f"Configuration: {num_adaptations} iterations, {epochs} epochs per iteration")
    if method_seed is not None:
        set_global_seed(method_seed)

    # Get initial mesh coordinates
    vertex_array = export_vertex_coordinates(initial_mesh)
    mesh_x, mesh_y = vertex_array.T

    print(f"Initial mesh: {len(mesh_x):,} points")

    # Initialize model with shared training data coordinates
    model = FeedForward(mesh_x=mesh_x, mesh_y=mesh_y, problem=problem).to(DEVICE)
    if initial_state_dict is not None:
        model.load_state_dict(copy.deepcopy(initial_state_dict))

    # Store reference solution in model for consistent error assessment
    model.reference_mesh = reference_mesh
    model.reference_solution = reference_solution

    # Initialize tracking
    model.mesh_point_history = [vertex_array.clone()]
    model.mesh_point_count_history = [len(mesh_x)]

    # Create working mesh copy (will be refined during training)
    current_mesh = initial_mesh  # This gets modified during adaptation

    # Adaptation iterations
    for iteration in range(num_adaptations):
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration + 1}/{num_adaptations}")
        print(f"{'='*60}")

        print(f"\n--- Adaptation Iteration {iteration + 1} ---")
        model.iteration_point_count_history.append(len(model.mesh_x))
        iter_start = time.perf_counter()

        # Train and adapt mesh (uses shared training dataset + current mesh for residuals)
        adapt_mesh_and_train(
            model,
            current_mesh,
            shared_dataset,
            reference_mesh,
            reference_solution,
            epochs,
            export_images=export_images,
            iteration=iteration,
        )
        _record_iteration_runtime(model, time.perf_counter() - iter_start)
        print(
            f"Iteration {iteration + 1} completed. Current mesh: {len(model.mesh_x):,} points"
        )

    print(f"\n{'='*60}")
    print("ADAPTIVE TRAINING COMPLETED")
    print(f"{'='*60}")

    return model


def run_method_training_fair(
    method_name: str,
    problem,
    initial_mesh,
    shared_dataset,
    reference_mesh,
    reference_solution,
    num_adaptations: int,
    epochs: int,
    export_images: bool,
    reference_point_counts: list = None,
    initial_state_dict=None,
    method_seed=None,
):
    """
    Run training with any registered sampling method using shared components for fair comparison.

    This is the generalized training function that works with all methods from the
    methods registry (halton, sobol, random_r, rad, adaptive, random).

    Args:
        method_name: Name of method from methods registry
        initial_mesh: Initial mesh object
        shared_dataset: Shared training dataset (from initial mesh FEM solution)
        reference_mesh: High-fidelity reference mesh for error assessment
        reference_solution: High-fidelity reference solution for error assessment
        num_adaptations: Number of adaptation iterations
        epochs: Number of training epochs per iteration
        export_images: Whether to export visualization images
        reference_point_counts: Optional list of point counts to match (from adaptive method)

    Returns:
        FeedForward: Trained PINN model
    """
    from mesh_refinement import (
        compute_model_error,
        compute_random_model_error,
        compute_random_residuals,
    )
    from config import DEVICE, QUASI_RANDOM_CONFIG, RAD_CONFIG, RANDOM_R_CONFIG

    print(f"\n{'='*60}")
    print(f"METHOD TRAINING: {method_name.upper()}")
    print(f"{'='*60}")
    if method_seed is not None:
        set_global_seed(method_seed)

    # Get method instance with appropriate config
    domain_bounds = problem.get_sampling_bounds()

    if method_name == "rad":
        method = get_method(
            method_name,
            domain_bounds=domain_bounds,
            k=RAD_CONFIG["k"],
            c=RAD_CONFIG["c"],
            num_candidates=RAD_CONFIG["num_candidates"],
            resample_period=RAD_CONFIG["resample_period"],
            seed=method_seed,
        )
    elif method_name == "random_r":
        method = get_method(
            method_name,
            domain_bounds=domain_bounds,
            resample_period=RANDOM_R_CONFIG["resample_period"],
            seed=method_seed,
        )
    elif method_name in ("halton", "sobol"):
        method = get_method(
            method_name,
            domain_bounds=domain_bounds,
            seed=method_seed if method_seed is not None else QUASI_RANDOM_CONFIG["seed"],
        )
    else:
        # Fallback for other methods
        try:
            method = get_method(method_name, domain_bounds=domain_bounds)
        except TypeError:
            method = get_method(method_name)
    if hasattr(method, "set_problem"):
        method.set_problem(problem)

    print(f"Method: {method.name} - {method.description}")

    # Get initial mesh coordinates
    vertex_array = export_vertex_coordinates(initial_mesh)
    mesh_x, mesh_y = vertex_array.T
    initial_point_count = len(mesh_x)

    print(f"Initial mesh: {initial_point_count:,} points")

    # Initialize model
    model = FeedForward(mesh_x=mesh_x, mesh_y=mesh_y, problem=problem).to(DEVICE)
    if initial_state_dict is not None:
        model.load_state_dict(copy.deepcopy(initial_state_dict))
    model.reference_mesh = reference_mesh
    model.reference_solution = reference_solution

    # Initialize tracking histories
    model.mesh_point_history = [vertex_array.clone()]
    model.mesh_point_count_history = [initial_point_count]
    model.total_error_history = []
    model.boundary_error_history = []
    random_fe_space = None
    if method_name == "random":
        random_fe_space = H1(initial_mesh, order=1, dirichlet=".*")

    # Training iterations
    for iteration in range(num_adaptations):
        print(f"\n--- {method_name} Iteration {iteration + 1}/{num_adaptations} ---")
        iter_start = time.perf_counter()

        # Determine target point count
        if reference_point_counts and iteration < len(reference_point_counts):
            target_point_count = reference_point_counts[iteration]
            print(f"Matching reference point count: {target_point_count:,}")
        else:
            # Use initial count (no mesh growth for non-adaptive methods)
            target_point_count = initial_point_count

        # Get collocation points using the method
        x, y = method.get_collocation_points(
            initial_mesh,
            model=model,
            iteration=iteration,
            num_points=target_point_count,
        )

        # Update model's residual computation points
        model.set_mesh_points(x, y)

        actual_count = len(model.mesh_x)
        print(f"Using {actual_count:,} {method_name} points for residual computation")
        model.iteration_point_count_history.append(actual_count)

        # Train model
        train_model(
            model,
            shared_dataset,
            epochs,
            optimizer=TRAINING_CONFIG["optimizer"],
            lr=TRAINING_CONFIG["lr"],
        )

        # Compute and record error
        if method_name == "random":
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

        _record_iteration_runtime(model, time.perf_counter() - iter_start)

        # Record point count
        model.mesh_point_count_history.append(actual_count)

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
    reference_mesh_factor: float | None = None,
    seed: int | None = None,
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
    if methods_to_run is None:
        methods_to_run = ["adaptive", "random"]  # Default to current methods
    if epochs is None:
        epochs = TRAINING_CONFIG["epochs"]
    problem = _build_problem(problem_name, problem_kwargs)
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
    _log_comparison_budget_policy(epochs, methods_to_run)

    # Create shared components for fair comparison
    print("\n" + "=" * 60)
    print("CREATING SHARED COMPONENTS FOR FAIR COMPARISON")
    print("=" * 60)

    # 1. Create initial mesh and training dataset (shared by both methods)
    print("Creating initial mesh and training dataset...")
    initial_mesh = problem.create_mesh(maxh=mesh_size)
    gfu, fes = solve_FEM(initial_mesh, problem=problem)
    vertex_array = export_vertex_coordinates(initial_mesh)
    solution_array = export_fem_solution(initial_mesh, gfu, problem=problem)
    shared_training_dataset = create_dataset(vertex_array, solution_array)
    mesh_x, mesh_y = vertex_array.T
    print(f"Shared training dataset: {len(mesh_x):,} points")
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
    if "adaptive" in execution_order:
        execution_order = ["adaptive"] + [
            method for method in execution_order if method != "adaptive"
        ]

    # Run each requested method with shared components
    for method in execution_order:
        print(f"\n{'='*60}")
        print(f"RUNNING METHOD: {method.upper()}")
        print(f"{'='*60}")
        method_seed = _method_seed(seed, method)
        method_mesh = _clone_mesh(initial_mesh)

        if method == "adaptive":
            print("Starting adaptive mesh training...")
            model = run_adaptive_training_fair(
                problem,
                method_mesh,
                shared_training_dataset,
                reference_mesh,
                reference_solution,
                num_adaptations,
                epochs,
                export_images,
                initial_state_dict=initial_model_state,
                method_seed=method_seed,
            )
            trained_models["adaptive"] = model

        elif method in list_methods():
            # Use generalized method training for all registered methods
            reference_counts = None
            if "adaptive" in trained_models:
                reference_counts = trained_models[
                    "adaptive"
                ].iteration_point_count_history

            print(f"Starting {method} training...")
            model = run_method_training_fair(
                method_name=method,
                problem=problem,
                initial_mesh=method_mesh,
                shared_dataset=shared_training_dataset,
                reference_mesh=reference_mesh,
                reference_solution=reference_solution,
                num_adaptations=num_adaptations,
                epochs=epochs,
                export_images=export_images,
                reference_point_counts=reference_counts,
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
                dataset_size=len(shared_training_dataset),
                output_dir=images_dir(),
                include_gifs=create_gifs and export_images,
                cleanup_pngs=True,
            )
        except Exception as e:
            print(f"Warning: Failed to create multi-method visualizations: {e}")

    print(f"\n{'='*80}")
    print("EXPERIMENT COMPLETED SUCCESSFULLY")
    print(f"{'='*80}")

    return trained_models


def write_multi_method_histories_csv(trained_models: dict):
    """Write training histories for all methods to a single CSV file.

    Args:
        trained_models: Dictionary mapping method names to trained models
    """
    import csv

    output_path = os.path.join(reports_dir(), "all_methods_histories.csv")

    # Collect all data
    rows = []
    for method_name, model in trained_models.items():
        # Get error history
        total_errors = getattr(model, "total_error_history", [])
        total_error_rms = getattr(model, "total_error_rms_history", [])
        boundary_errors = getattr(model, "boundary_error_history", [])
        total_residuals = getattr(model, "total_residual_history", [])
        fixed_total_residuals = getattr(model, "fixed_total_residual_history", [])
        fixed_boundary_residuals = getattr(
            model, "fixed_boundary_residual_history", []
        )
        fixed_rms_residuals = getattr(model, "fixed_rms_residual_history", [])
        iteration_point_counts = getattr(model, "iteration_point_count_history", [])
        point_counts = getattr(model, "mesh_point_count_history", [])
        iteration_runtime = getattr(model, "iteration_runtime_history", [])
        cumulative_runtime = getattr(model, "cumulative_runtime_history", [])

        num_rows = max(
            len(total_errors),
            len(total_error_rms),
            len(boundary_errors),
            len(total_residuals),
            len(fixed_total_residuals),
            len(fixed_boundary_residuals),
            len(fixed_rms_residuals),
            len(iteration_point_counts),
            len(iteration_runtime),
            len(cumulative_runtime),
            max(len(point_counts) - 1, 0),
        )

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
                    "total_error_rms": (
                        total_error_rms[i] if i < len(total_error_rms) else None
                    ),
                    "boundary_error": (
                        boundary_errors[i] if i < len(boundary_errors) else None
                    ),
                    "total_residual": (
                        total_residuals[i] if i < len(total_residuals) else None
                    ),
                    "fixed_total_residual": (
                        fixed_total_residuals[i]
                        if i < len(fixed_total_residuals)
                        else None
                    ),
                    "fixed_boundary_residual": (
                        fixed_boundary_residuals[i]
                        if i < len(fixed_boundary_residuals)
                        else None
                    ),
                    "fixed_rms_residual": (
                        fixed_rms_residuals[i]
                        if i < len(fixed_rms_residuals)
                        else None
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

    # Write CSV
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "iteration",
                "total_error",
                "total_error_rms",
                "boundary_error",
                "total_residual",
                "fixed_total_residual",
                "fixed_boundary_residual",
                "fixed_rms_residual",
                "point_count",
                "iteration_runtime_sec",
                "cumulative_runtime_sec",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Multi-method histories written to: {output_path}")


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
