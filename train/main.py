"""
Main execution script for PINN adaptive mesh training.
This is the entry point for running the complete experiment.
"""

import json
import os
import sys


def _bootstrap_runtime_from_cli(argv):
    device = None
    remaining = [argv[0]]
    i = 1

    while i < len(argv):
        if argv[i] == "--device":
            if i + 1 >= len(argv):
                raise SystemExit("error: --device requires a value")
            device = argv[i + 1]
            i += 2
            continue
        remaining.append(argv[i])
        i += 1

    mode = remaining[1].lower() if len(remaining) > 1 else None
    if device is None and "PINN_DEVICE" not in os.environ and mode == "smoke":
        device = "cpu"

    if device is not None:
        os.environ["PINN_DEVICE"] = device

    return remaining


if __name__ == "__main__":
    sys.argv = _bootstrap_runtime_from_cli(sys.argv)


from experiments import run_complete_experiment, run_hyperparameter_study
from config import DEVICE, TRAINING_CONFIG, MESH_CONFIG
from utils import get_system_info, log_experiment_info, set_global_seed
from paths import generate_run_id, set_active_run, write_run_metadata, OUTPUTS_ROOT
from visualization import plot_ablation_error_shaded


SMOKE_DEFAULT_METHODS = ["adaptive", "random"]
SMOKE_DEFAULT_MESH_SIZE = 0.7
SMOKE_DEFAULT_NUM_ADAPTATIONS = 1
SMOKE_DEFAULT_EPOCHS = 1
SMOKE_DEFAULT_REFERENCE_MESH_FACTOR = 0.05

DEV_DEFAULT_METHODS = ["adaptive", "random", "halton"]
DEV_DEFAULT_MESH_SIZE = 0.7
DEV_DEFAULT_NUM_ADAPTATIONS = 4
DEV_DEFAULT_EPOCHS = 100
DEV_DEFAULT_REFERENCE_MESH_FACTOR = 0.05

SCREEN_DEFAULT_METHODS = [
    "adaptive",
    "random",
    "halton",
    "rad",
]
SCREEN_DEFAULT_MESH_SIZE = 0.7
SCREEN_DEFAULT_NUM_ADAPTATIONS = 6
SCREEN_DEFAULT_EPOCHS = 200
SCREEN_DEFAULT_REFERENCE_MESH_FACTOR = 0.05


def _resolve_seed():
    seed = TRAINING_CONFIG.get("seed")
    if seed is None:
        seed = int.from_bytes(os.urandom(4), "little")
    return int(seed)


def _parse_methods_arg(raw_value, fallback_methods=None):
    methods = [method.strip() for method in raw_value.split(",") if method.strip()]
    fallback = fallback_methods or SMOKE_DEFAULT_METHODS
    return methods or list(fallback)


def _parse_validation_options_arg(raw_value):
    try:
        value = json.loads(raw_value)
    except Exception as e:
        print(f"Warning: could not parse validation options JSON: {e}")
        return None
    if not isinstance(value, dict):
        print("Warning: validation options must decode to a JSON object")
        return None
    return value


def _parse_dev_args(args):
    options = {
        "seed": None,
        "methods_to_run": list(DEV_DEFAULT_METHODS),
        "mesh_size": DEV_DEFAULT_MESH_SIZE,
        "num_adaptations": DEV_DEFAULT_NUM_ADAPTATIONS,
        "epochs": DEV_DEFAULT_EPOCHS,
        "reference_mesh_factor": DEV_DEFAULT_REFERENCE_MESH_FACTOR,
        "problem_name": "poisson",
        "problem_kwargs": None,
        "validation_options": None,
        "export_images": False,
        "create_gifs": False,
        "generate_report": True,
    }

    i = 0
    ignored = []
    while i < len(args):
        arg = args[i]
        if arg == "--seed" and i + 1 < len(args):
            try:
                options["seed"] = int(args[i + 1])
            except ValueError:
                print(f"Warning: invalid seed value '{args[i + 1]}', using random")
            i += 2
            continue
        if arg == "--methods" and i + 1 < len(args):
            options["methods_to_run"] = _parse_methods_arg(
                args[i + 1], fallback_methods=DEV_DEFAULT_METHODS
            )
            i += 2
            continue
        if arg == "--mesh-size" and i + 1 < len(args):
            try:
                options["mesh_size"] = float(args[i + 1])
            except ValueError:
                print(
                    f"Warning: invalid mesh size '{args[i + 1]}', using default dev value"
                )
            i += 2
            continue
        if arg == "--iterations" and i + 1 < len(args):
            try:
                options["num_adaptations"] = int(args[i + 1])
            except ValueError:
                print(
                    f"Warning: invalid iterations value '{args[i + 1]}', using default dev value"
                )
            i += 2
            continue
        if arg == "--epochs" and i + 1 < len(args):
            try:
                options["epochs"] = int(args[i + 1])
            except ValueError:
                print(
                    f"Warning: invalid epochs value '{args[i + 1]}', using default dev value"
                )
            i += 2
            continue
        if arg == "--reference-mesh-factor" and i + 1 < len(args):
            try:
                options["reference_mesh_factor"] = float(args[i + 1])
            except ValueError:
                print(
                    "Warning: invalid reference mesh factor "
                    f"'{args[i + 1]}', using default dev value"
                )
            i += 2
            continue
        if arg == "--problem" and i + 1 < len(args):
            options["problem_name"] = args[i + 1]
            i += 2
            continue
        if arg == "--problem-kwargs" and i + 1 < len(args):
            try:
                options["problem_kwargs"] = json.loads(args[i + 1])
            except Exception as e:
                print(f"Warning: could not parse problem kwargs JSON: {e}")
            i += 2
            continue
        if arg == "--validation-options" and i + 1 < len(args):
            options["validation_options"] = _parse_validation_options_arg(args[i + 1])
            i += 2
            continue
        if arg == "--images":
            options["export_images"] = True
            i += 1
            continue
        if arg == "--gifs":
            options["export_images"] = True
            options["create_gifs"] = True
            i += 1
            continue
        if arg == "--no-report":
            options["generate_report"] = False
            i += 1
            continue
        ignored.append(arg)
        i += 1

    if ignored:
        print(f"Warning: ignoring unsupported dev arguments: {' '.join(ignored)}")
    return options


def _parse_screen_args(args):
    options = {
        "seed": None,
        "methods_to_run": list(SCREEN_DEFAULT_METHODS),
        "mesh_size": SCREEN_DEFAULT_MESH_SIZE,
        "num_adaptations": SCREEN_DEFAULT_NUM_ADAPTATIONS,
        "epochs": SCREEN_DEFAULT_EPOCHS,
        "reference_mesh_factor": SCREEN_DEFAULT_REFERENCE_MESH_FACTOR,
        "problem_name": "poisson",
        "problem_kwargs": None,
        "validation_options": None,
        "export_images": False,
        "create_gifs": False,
        "generate_report": True,
    }

    i = 0
    ignored = []
    while i < len(args):
        arg = args[i]
        if arg == "--seed" and i + 1 < len(args):
            try:
                options["seed"] = int(args[i + 1])
            except ValueError:
                print(f"Warning: invalid seed value '{args[i + 1]}', using random")
            i += 2
            continue
        if arg == "--methods" and i + 1 < len(args):
            options["methods_to_run"] = _parse_methods_arg(
                args[i + 1], fallback_methods=SCREEN_DEFAULT_METHODS
            )
            i += 2
            continue
        if arg == "--mesh-size" and i + 1 < len(args):
            try:
                options["mesh_size"] = float(args[i + 1])
            except ValueError:
                print(
                    "Warning: invalid mesh size "
                    f"'{args[i + 1]}', using default screen value"
                )
            i += 2
            continue
        if arg == "--iterations" and i + 1 < len(args):
            try:
                options["num_adaptations"] = int(args[i + 1])
            except ValueError:
                print(
                    "Warning: invalid iterations value "
                    f"'{args[i + 1]}', using default screen value"
                )
            i += 2
            continue
        if arg == "--epochs" and i + 1 < len(args):
            try:
                options["epochs"] = int(args[i + 1])
            except ValueError:
                print(
                    "Warning: invalid epochs value "
                    f"'{args[i + 1]}', using default screen value"
                )
            i += 2
            continue
        if arg == "--reference-mesh-factor" and i + 1 < len(args):
            try:
                options["reference_mesh_factor"] = float(args[i + 1])
            except ValueError:
                print(
                    "Warning: invalid reference mesh factor "
                    f"'{args[i + 1]}', using default screen value"
                )
            i += 2
            continue
        if arg == "--problem" and i + 1 < len(args):
            options["problem_name"] = args[i + 1]
            i += 2
            continue
        if arg == "--problem-kwargs" and i + 1 < len(args):
            try:
                options["problem_kwargs"] = json.loads(args[i + 1])
            except Exception as e:
                print(f"Warning: could not parse problem kwargs JSON: {e}")
            i += 2
            continue
        if arg == "--validation-options" and i + 1 < len(args):
            options["validation_options"] = _parse_validation_options_arg(args[i + 1])
            i += 2
            continue
        if arg == "--images":
            options["export_images"] = True
            i += 1
            continue
        if arg == "--gifs":
            options["export_images"] = True
            options["create_gifs"] = True
            i += 1
            continue
        if arg == "--no-report":
            options["generate_report"] = False
            i += 1
            continue
        ignored.append(arg)
        i += 1

    if ignored:
        print(f"Warning: ignoring unsupported screen arguments: {' '.join(ignored)}")
    return options


def _parse_smoke_args(args):
    options = {
        "seed": None,
        "methods_to_run": list(SMOKE_DEFAULT_METHODS),
        "mesh_size": SMOKE_DEFAULT_MESH_SIZE,
        "num_adaptations": SMOKE_DEFAULT_NUM_ADAPTATIONS,
        "epochs": SMOKE_DEFAULT_EPOCHS,
        "reference_mesh_factor": SMOKE_DEFAULT_REFERENCE_MESH_FACTOR,
        "problem_name": "poisson",
        "problem_kwargs": None,
        "validation_options": None,
    }

    i = 0
    ignored = []
    while i < len(args):
        arg = args[i]
        if arg == "--seed" and i + 1 < len(args):
            try:
                options["seed"] = int(args[i + 1])
            except ValueError:
                print(f"Warning: invalid seed value '{args[i + 1]}', using random")
            i += 2
            continue
        if arg == "--methods" and i + 1 < len(args):
            options["methods_to_run"] = _parse_methods_arg(
                args[i + 1], fallback_methods=SMOKE_DEFAULT_METHODS
            )
            i += 2
            continue
        if arg == "--mesh-size" and i + 1 < len(args):
            try:
                options["mesh_size"] = float(args[i + 1])
            except ValueError:
                print(
                    f"Warning: invalid mesh size '{args[i + 1]}', using default smoke value"
                )
            i += 2
            continue
        if arg == "--iterations" and i + 1 < len(args):
            try:
                options["num_adaptations"] = int(args[i + 1])
            except ValueError:
                print(
                    f"Warning: invalid iterations value '{args[i + 1]}', using default smoke value"
                )
            i += 2
            continue
        if arg == "--epochs" and i + 1 < len(args):
            try:
                options["epochs"] = int(args[i + 1])
            except ValueError:
                print(
                    f"Warning: invalid epochs value '{args[i + 1]}', using default smoke value"
                )
            i += 2
            continue
        if arg == "--reference-mesh-factor" and i + 1 < len(args):
            try:
                options["reference_mesh_factor"] = float(args[i + 1])
            except ValueError:
                print(
                    "Warning: invalid reference mesh factor "
                    f"'{args[i + 1]}', using default smoke value"
                )
            i += 2
            continue
        if arg == "--problem" and i + 1 < len(args):
            options["problem_name"] = args[i + 1]
            i += 2
            continue
        if arg == "--problem-kwargs" and i + 1 < len(args):
            try:
                options["problem_kwargs"] = json.loads(args[i + 1])
            except Exception as e:
                print(f"Warning: could not parse problem kwargs JSON: {e}")
            i += 2
            continue
        if arg == "--validation-options" and i + 1 < len(args):
            options["validation_options"] = _parse_validation_options_arg(args[i + 1])
            i += 2
            continue
        ignored.append(arg)
        i += 1

    if ignored:
        print(f"Warning: ignoring unsupported smoke arguments: {' '.join(ignored)}")
    return options


def run_smoke_test(
    seed=None,
    methods_to_run=None,
    mesh_size=SMOKE_DEFAULT_MESH_SIZE,
    num_adaptations=SMOKE_DEFAULT_NUM_ADAPTATIONS,
    epochs=SMOKE_DEFAULT_EPOCHS,
    reference_mesh_factor=SMOKE_DEFAULT_REFERENCE_MESH_FACTOR,
    problem_name="poisson",
    problem_kwargs=None,
    validation_options=None,
):
    """Run a minimal end-to-end smoke test through the real training stack."""
    print("Running smoke test...")

    if seed is None:
        seed = _resolve_seed()
    seed = int(seed)
    methods_to_run = methods_to_run or list(SMOKE_DEFAULT_METHODS)
    set_global_seed(seed)

    run_id = generate_run_id(f"smoke-seed{seed}")
    run_paths = set_active_run(run_id)
    print(f"Smoke run ID: {run_id}")
    print(f"Outputs root: {run_paths['root']}")
    print(
        "Smoke configuration: "
        f"mesh_size={mesh_size}, iterations={num_adaptations}, epochs={epochs}, "
        f"reference_mesh_factor={reference_mesh_factor}, methods={methods_to_run}, "
        f"device={DEVICE}"
    )

    try:
        write_run_metadata(
            extra={
                "phase": "smoke_before_run",
                "seed": seed,
                "methods": methods_to_run,
                "problem": problem_name,
            }
        )

        result = run_complete_experiment(
            mesh_size=mesh_size,
            num_adaptations=num_adaptations,
            epochs=epochs,
            export_images=False,
            create_gifs=False,
            generate_report=False,
            methods_to_run=methods_to_run,
            problem_name=problem_name,
            problem_kwargs=problem_kwargs,
            validation_options=validation_options,
            reference_mesh_factor=reference_mesh_factor,
            seed=seed,
        )

        write_run_metadata(
            extra={
                "phase": "smoke_after_run",
                "seed": seed,
                "methods": methods_to_run,
                "problem": problem_name,
            }
        )

        print("\nSmoke test summary:")
        for method_name, model in result.items():
            total_errors = getattr(model, "total_error_history", [])
            relative_l2_errors = getattr(model, "relative_l2_error_history", [])
            point_counts = getattr(model, "mesh_point_count_history", [])
            final_error = total_errors[-1] if total_errors else None
            final_relative_l2 = relative_l2_errors[-1] if relative_l2_errors else None
            final_points = point_counts[-1] if point_counts else None
            print(
                f"  {method_name}: "
                f"points={final_points if final_points is not None else 'n/a'}, "
                f"relative_l2={final_relative_l2 if final_relative_l2 is not None else 'n/a'}, "
                f"error_integral={final_error if final_error is not None else 'n/a'}"
            )

        print(f"Smoke test completed successfully. Results: {run_paths['root']}")
        return True
    except Exception as e:
        print(f"Smoke test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def run_dev_experiment(
    seed=None,
    methods_to_run=None,
    mesh_size=DEV_DEFAULT_MESH_SIZE,
    num_adaptations=DEV_DEFAULT_NUM_ADAPTATIONS,
    epochs=DEV_DEFAULT_EPOCHS,
    reference_mesh_factor=DEV_DEFAULT_REFERENCE_MESH_FACTOR,
    problem_name="poisson",
    problem_kwargs=None,
    validation_options=None,
    export_images=False,
    create_gifs=False,
    generate_report=True,
):
    """Run the lean baseline development experiment profile."""
    print("Running dev experiment...")

    if seed is None:
        seed = _resolve_seed()
    seed = int(seed)
    methods_to_run = methods_to_run or list(DEV_DEFAULT_METHODS)
    set_global_seed(seed)

    run_id = generate_run_id(f"dev-seed{seed}")
    run_paths = set_active_run(run_id)
    print(f"Dev run ID: {run_id}")
    print(f"Outputs root: {run_paths['root']}")
    print(
        "Dev configuration: "
        f"mesh_size={mesh_size}, iterations={num_adaptations}, epochs={epochs}, "
        f"reference_mesh_factor={reference_mesh_factor}, methods={methods_to_run}, "
        f"export_images={export_images}, create_gifs={create_gifs}, device={DEVICE}"
    )

    try:
        write_run_metadata(
            extra={
                "phase": "dev_before_run",
                "seed": seed,
                "methods": methods_to_run,
                "problem": problem_name,
                "profile": "lean_dev",
            }
        )

        result = run_complete_experiment(
            mesh_size=mesh_size,
            num_adaptations=num_adaptations,
            epochs=epochs,
            export_images=export_images,
            create_gifs=create_gifs,
            generate_report=generate_report,
            methods_to_run=methods_to_run,
            problem_name=problem_name,
            problem_kwargs=problem_kwargs,
            validation_options=validation_options,
            reference_mesh_factor=reference_mesh_factor,
            seed=seed,
        )

        write_run_metadata(
            extra={
                "phase": "dev_after_run",
                "seed": seed,
                "methods": methods_to_run,
                "problem": problem_name,
                "profile": "lean_dev",
            }
        )

        print("\nDev experiment summary:")
        for method_name, model in result.items():
            total_errors = getattr(model, "total_error_history", [])
            relative_l2_errors = getattr(model, "relative_l2_error_history", [])
            point_counts = getattr(model, "mesh_point_count_history", [])
            runtimes = getattr(model, "cumulative_runtime_history", [])
            final_error = total_errors[-1] if total_errors else None
            final_relative_l2 = relative_l2_errors[-1] if relative_l2_errors else None
            final_points = point_counts[-1] if point_counts else None
            final_runtime = runtimes[-1] if runtimes else None
            print(
                f"  {method_name}: "
                f"points={final_points if final_points is not None else 'n/a'}, "
                f"relative_l2={final_relative_l2 if final_relative_l2 is not None else 'n/a'}, "
                f"error_integral={final_error if final_error is not None else 'n/a'}, "
                f"runtime_sec={final_runtime if final_runtime is not None else 'n/a'}"
            )

        print(f"Dev experiment completed successfully. Results: {run_paths['root']}")
        return True
    except Exception as e:
        print(f"Dev experiment failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def run_screen_experiment(
    seed=None,
    methods_to_run=None,
    mesh_size=SCREEN_DEFAULT_MESH_SIZE,
    num_adaptations=SCREEN_DEFAULT_NUM_ADAPTATIONS,
    epochs=SCREEN_DEFAULT_EPOCHS,
    reference_mesh_factor=SCREEN_DEFAULT_REFERENCE_MESH_FACTOR,
    problem_name="poisson",
    problem_kwargs=None,
    validation_options=None,
    export_images=False,
    create_gifs=False,
    generate_report=True,
):
    """Run the stronger screening experiment profile."""
    print("Running screen experiment...")

    if seed is None:
        seed = _resolve_seed()
    seed = int(seed)
    methods_to_run = methods_to_run or list(SCREEN_DEFAULT_METHODS)
    set_global_seed(seed)

    run_id = generate_run_id(f"screen-seed{seed}")
    run_paths = set_active_run(run_id)
    print(f"Screen run ID: {run_id}")
    print(f"Outputs root: {run_paths['root']}")
    print(
        "Screen configuration: "
        f"mesh_size={mesh_size}, iterations={num_adaptations}, epochs={epochs}, "
        f"reference_mesh_factor={reference_mesh_factor}, methods={methods_to_run}, "
        f"export_images={export_images}, create_gifs={create_gifs}, device={DEVICE}"
    )

    try:
        write_run_metadata(
            extra={
                "phase": "screen_before_run",
                "seed": seed,
                "methods": methods_to_run,
                "problem": problem_name,
                "profile": "screen",
            }
        )

        result = run_complete_experiment(
            mesh_size=mesh_size,
            num_adaptations=num_adaptations,
            epochs=epochs,
            export_images=export_images,
            create_gifs=create_gifs,
            generate_report=generate_report,
            methods_to_run=methods_to_run,
            problem_name=problem_name,
            problem_kwargs=problem_kwargs,
            validation_options=validation_options,
            reference_mesh_factor=reference_mesh_factor,
            seed=seed,
        )

        write_run_metadata(
            extra={
                "phase": "screen_after_run",
                "seed": seed,
                "methods": methods_to_run,
                "problem": problem_name,
                "profile": "screen",
            }
        )

        print("\nScreen experiment summary:")
        for method_name, model in result.items():
            total_errors = getattr(model, "total_error_history", [])
            relative_l2_errors = getattr(model, "relative_l2_error_history", [])
            relative_fixed_l2_residuals = getattr(
                model, "relative_fixed_l2_residual_history", []
            )
            point_counts = getattr(model, "mesh_point_count_history", [])
            runtimes = getattr(model, "cumulative_runtime_history", [])
            final_error = total_errors[-1] if total_errors else None
            final_relative_l2 = relative_l2_errors[-1] if relative_l2_errors else None
            final_relative_residual = (
                relative_fixed_l2_residuals[-1]
                if relative_fixed_l2_residuals
                else None
            )
            final_points = point_counts[-1] if point_counts else None
            final_runtime = runtimes[-1] if runtimes else None
            print(
                f"  {method_name}: "
                f"points={final_points if final_points is not None else 'n/a'}, "
                f"relative_l2={final_relative_l2 if final_relative_l2 is not None else 'n/a'}, "
                "relative_fixed_l2_residual="
                f"{final_relative_residual if final_relative_residual is not None else 'n/a'}, "
                f"error_integral={final_error if final_error is not None else 'n/a'}, "
                f"runtime_sec={final_runtime if final_runtime is not None else 'n/a'}"
            )

        print(f"Screen experiment completed successfully. Results: {run_paths['root']}")
        return True
    except Exception as e:
        print(f"Screen experiment failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Main function to run the complete PINN adaptive mesh training experiment."""

    print("PINN Adaptive Mesh Experiment")
    print("===========================")

    # Seed handling: respect config seed if provided; else generate one per run
    seed = _resolve_seed()
    set_global_seed(seed)
    print(f"Using random seed: {seed}")

    # Print system information
    system_info = get_system_info()
    print("\nSystem Information:")
    for key, value in system_info.items():
        print(f"  {key}: {value}")

    # Create a per-run output folder (outputs/<run-id>/...)
    run_id = generate_run_id(f"adapt-vs-rand-seed{seed}")
    run_paths = set_active_run(run_id)
    print(f"\nRun ID: {run_id}")
    print(f"Outputs root: {run_paths['root']}")

    # Configuration
    mesh_size = MESH_CONFIG["maxh"]
    num_adaptations = TRAINING_CONFIG["iterations"]
    epochs = TRAINING_CONFIG["epochs"]
    export_images = (
        True  # Set to True to save images during training (including error fields)
    )

    print("\nExperiment Configuration:")
    print(f"  Initial mesh size: {mesh_size}")
    print(f"  Adaptation iterations: {num_adaptations}")
    print(f"  Training epochs per iteration: {epochs}")
    print(f"  Export images: {export_images}")

    try:
        # Write run metadata (configs + system + git) before run starts
        write_run_metadata(extra={"phase": "before_run", "seed": seed})

        # Run the complete experiment
        result = run_complete_experiment(
            mesh_size=mesh_size,
            num_adaptations=num_adaptations,
            epochs=epochs,
            export_images=export_images,
            create_gifs=export_images,  # Only create GIFs if images are exported
            generate_report=True,
            methods_to_run=["adaptive", "random"],  # Default methods
            seed=seed,
        )

        # Get models from the returned dictionary
        adaptive_model = result.get("adaptive")
        random_model = result.get("random")

        # Log experiment information
        config_info = {
            "mesh_size": mesh_size,
            "iterations": num_adaptations,
            "epochs": epochs,
            "export_images": export_images,
        }
        log_experiment_info(adaptive_model, config_info)
        # Update run metadata post-run
        write_run_metadata(extra={"phase": "after_run", "seed": seed})

        print("\n" + "=" * 60)
        print("EXPERIMENT SUMMARY")
        print("=" * 60)

        # Adaptive model results
        if adaptive_model and adaptive_model.mesh_point_count_history:
            initial_points = adaptive_model.mesh_point_count_history[0]
            final_points = adaptive_model.mesh_point_count_history[-1]
            refinement_factor = final_points / initial_points
            print("Adaptive Model:")
            print(
                f"  Mesh refinement: {initial_points:,} → {final_points:,} points (×{refinement_factor:.2f})"
            )

        if adaptive_model and getattr(adaptive_model, "relative_l2_error_history", None):
            initial_error = adaptive_model.relative_l2_error_history[0]
            final_error = adaptive_model.relative_l2_error_history[-1]
            error_reduction = (
                initial_error / final_error if final_error > 0 else float("inf")
            )
            print(
                "  Relative L2 reduction: "
                f"{initial_error:.2e} → {final_error:.2e} (×{error_reduction:.2f})"
            )

        # Random model results
        if random_model:
            print("Random Model:")
            if getattr(random_model, "relative_l2_error_history", None):
                final_random_error = random_model.relative_l2_error_history[-1]
                print(f"  Final relative L2 error: {final_random_error:.2e}")

                if adaptive_model and getattr(
                    adaptive_model, "relative_l2_error_history", None
                ):
                    final_adaptive_error = adaptive_model.relative_l2_error_history[-1]
                    improvement = (
                        final_random_error / final_adaptive_error
                        if final_adaptive_error > 0
                        else float("inf")
                    )
                    print(f"  Adaptive vs Random improvement: ×{improvement:.2f}")

        print(f"\nResults saved to: {run_paths['root']}")
        print("=" * 60)

    except Exception as e:
        print(f"\nError during experiment: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


def run_cleanup(run_id=None):
    """Clean up temporary files from a specific run or all runs.

    Args:
        run_id: Specific run ID to clean, or None for all runs
    """
    import glob
    import shutil

    if run_id:
        # Clean specific run
        run_path = os.path.join(OUTPUTS_ROOT, run_id)
        if not os.path.exists(run_path):
            print(f"Run not found: {run_id}")
            return False
        targets = [run_path]
    else:
        # Clean all runs
        targets = glob.glob(os.path.join(OUTPUTS_ROOT, "*"))

    patterns = ["*.vtu", "*.vtk", "vtk_export*", "*_frame_*.png"]
    total_cleaned = 0

    for target in targets:
        if not os.path.isdir(target):
            continue
        for pattern in patterns:
            for match in glob.glob(os.path.join(target, "**", pattern), recursive=True):
                try:
                    if os.path.isdir(match):
                        shutil.rmtree(match)
                    else:
                        os.remove(match)
                    total_cleaned += 1
                except Exception as e:
                    print(f"Error removing {match}: {e}")

    print(f"Cleaned up {total_cleaned} temporary files")
    return True


def run_full_cleanup():
    """Clean up temporary files from all run directories."""
    print("Cleaning up all temporary files from outputs/...")
    return run_cleanup()


def run_ablation_summary_plot(run_ids):
    """Create shaded mean±std ablation plot from a list of run IDs.

    Expects each run to have reports/all_methods_histories.csv.
    """
    if not run_ids:
        print("No run IDs provided for ablation plot")
        return False
    roots = [os.path.join(OUTPUTS_ROOT, rid) for rid in run_ids]
    out_dir = os.path.join(OUTPUTS_ROOT, "ablation_summary")
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, "ablation_error_shaded.png")
    plot_ablation_error_shaded(roots, save_path)
    print(f"Ablation summary plot saved to: {save_path}")
    return True


if __name__ == "__main__":
    def parse_seed_arg(args):
        """Extract --seed <value> from args list."""
        seed = None
        remaining = []
        i = 0
        while i < len(args):
            if args[i] == "--seed" and i + 1 < len(args):
                try:
                    seed = int(args[i + 1])
                except ValueError:
                    print(f"Warning: invalid seed value '{args[i + 1]}', using random")
                i += 2
            else:
                remaining.append(args[i])
                i += 1
        return seed, remaining

    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()

        if mode == "smoke":
            seed_override, remaining = parse_seed_arg(sys.argv[2:])
            smoke_options = _parse_smoke_args(remaining)
            if seed_override is not None:
                smoke_options["seed"] = seed_override
            success = run_smoke_test(**smoke_options)
        elif mode == "hparams":
            # Optional: allow a JSON grid file or inline JSON after the mode, and --images flag
            import json

            export_images = False
            grid = None
            args = sys.argv[2:]
            # Parse flags first
            if "--images" in args:
                export_images = True
                args = [a for a in args if a != "--images"]
            # Remaining arg can be a path or inline JSON
            if args:
                candidate = args[0]
                try:
                    # Inline JSON
                    if candidate.strip().startswith("{"):
                        grid = json.loads(candidate)
                    else:
                        # Assume file path
                        if os.path.exists(candidate):
                            with open(candidate) as f:
                                grid = json.load(f)
                        else:
                            print(
                                f"Grid file not found: {candidate} (using default grid)"
                            )
                except Exception as e:
                    print(f"Warning: could not parse grid, using default. Error: {e}")
            results = run_hyperparameter_study(grid=grid, export_images=export_images)
            # Consider success if at least one run ok
            success = (
                any(r.get("status") == "ok" for r in results.values())
                if results
                else False
            )
        elif mode == "dev":
            seed_override, remaining = parse_seed_arg(sys.argv[2:])
            dev_options = _parse_dev_args(remaining)
            if seed_override is not None:
                dev_options["seed"] = seed_override
            success = run_dev_experiment(**dev_options)
        elif mode == "screen":
            seed_override, remaining = parse_seed_arg(sys.argv[2:])
            screen_options = _parse_screen_args(remaining)
            if seed_override is not None:
                screen_options["seed"] = seed_override
            success = run_screen_experiment(**screen_options)
        elif mode == "main":
            # Parse --seed flag
            seed_override, _ = parse_seed_arg(sys.argv[2:])
            if seed_override is not None:
                TRAINING_CONFIG["seed"] = seed_override
            success = main()
        elif mode == "cleanup":
            # Optional: cleanup <run-id>
            run_id = sys.argv[2] if len(sys.argv) > 2 else None
            success = run_cleanup(run_id)
        elif mode == "cleanup-all":
            success = run_full_cleanup()
        elif mode == "ablate-plot":
            run_ids = sys.argv[2:]
            if not run_ids:
                print("Usage: python main.py ablate-plot <run-id> [<run-id> ...]")
                success = False
            else:
                success = run_ablation_summary_plot(run_ids)
        else:
            print(
                "Usage: python main.py "
                "[main|dev|screen|smoke|hparams|cleanup|cleanup-all|ablate-plot]"
            )
            print("")
            print("Commands:")
            print("  Global options:")
            print("               --device <name> Select runtime device")
            print("                              Examples: cpu, cuda, cuda:0, auto")
            print("  main         Run full experiment (default)")
            print("               --seed <int>   Override random seed")
            print("  dev          Run the lean development experiment profile")
            print("               --seed <int>   Override dev seed")
            print("               --methods <m1,m2>  Comma-separated method list")
            print("               --mesh-size <f>     Initial mesh size")
            print("               --iterations <n>    Adaptation steps")
            print("               --epochs <n>        Training epochs per step")
            print(
                "               --reference-mesh-factor <f>  Reference mesh size factor"
            )
            print("               --problem <name>    Problem name")
            print("               --problem-kwargs '{...}'  Inline JSON kwargs")
            print("               --images            Export PNG visualizations")
            print("               --gifs              Export PNGs and GIFs")
            print("               --no-report         Skip report/plot bundle generation")
            print("  screen       Run the stronger screening experiment profile")
            print("               --seed <int>   Override screen seed")
            print("               --methods <m1,m2>  Comma-separated method list")
            print("               --mesh-size <f>     Initial mesh size")
            print("               --iterations <n>    Adaptation steps")
            print("               --epochs <n>        Training epochs per step")
            print(
                "               --reference-mesh-factor <f>  Reference mesh size factor"
            )
            print("               --problem <name>    Problem name")
            print("               --problem-kwargs '{...}'  Inline JSON kwargs")
            print("               --images            Export PNG visualizations")
            print("               --gifs              Export PNGs and GIFs")
            print("               --no-report         Skip report/plot bundle generation")
            print("  smoke        Run a minimal end-to-end smoke test")
            print("               --seed <int>   Override smoke-test seed")
            print("               --methods <m1,m2>  Comma-separated method list")
            print("               --mesh-size <f>     Initial mesh size")
            print("               --iterations <n>    Adaptation steps")
            print("               --epochs <n>        Training epochs per step")
            print(
                "               --reference-mesh-factor <f>  Reference mesh size factor"
            )
            print("               --problem <name>    Problem name")
            print("               --problem-kwargs '{...}'  Inline JSON kwargs")
            print("  hparams      Run hyperparameter study")
            print("               [grid.json]    Path to JSON grid file")
            print("               ['{...}']      Inline JSON grid")
            print("               --images       Export images for each run")
            print("  cleanup      Clean up temporary files from outputs/")
            print("               [run-id]       Clean specific run only")
            print("  cleanup-all  Clean up all temp files from outputs/")
            print("  ablate-plot  Generate shaded ablation plot")
            print("               <run-id> ...   One or more run IDs to aggregate")
            success = False
    else:
        # Default: run main experiment
        success = main()

    sys.exit(0 if success else 1)
