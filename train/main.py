"""
Main execution script for PINN adaptive mesh training.
This is the entry point for running the complete experiment.
"""

from experiments import run_complete_experiment, run_hyperparameter_study
from config import TRAINING_CONFIG, MESH_CONFIG, PROJECT_ROOT
from utils import get_system_info, log_experiment_info, set_global_seed
from paths import generate_run_id, set_active_run, write_run_metadata, OUTPUTS_ROOT
from visualization import plot_ablation_error_shaded
import os


def main():
    """Main function to run the complete PINN adaptive mesh training experiment."""

    print("PINN Adaptive Mesh Experiment")
    print("===========================")

    # Seed handling: respect config seed if provided; else generate one per run
    seed = TRAINING_CONFIG.get("seed")
    if seed is None:
        seed = int.from_bytes(os.urandom(4), "little")
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

        if adaptive_model and adaptive_model.total_error_history:
            initial_error = adaptive_model.total_error_history[0]
            final_error = adaptive_model.total_error_history[-1]
            error_reduction = (
                initial_error / final_error if final_error > 0 else float("inf")
            )
            print(
                f"  Error reduction: {initial_error:.2e} → {final_error:.2e} (×{error_reduction:.2f})"
            )

        # Random model results
        if random_model:
            print("Random Model:")
            if random_model.total_error_history:
                final_random_error = random_model.total_error_history[-1]
                print(f"  Final error: {final_random_error:.2e}")

                if adaptive_model and adaptive_model.total_error_history:
                    final_adaptive_error = adaptive_model.total_error_history[-1]
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


def run_quick_test():
    """Run a quick test with reduced parameters for debugging."""
    print("Running quick test...")

    # Set up a proper run for test mode
    seed = TRAINING_CONFIG.get("seed")
    if seed is None:
        seed = int.from_bytes(os.urandom(4), "little")
    set_global_seed(seed)

    run_id = generate_run_id(f"test-seed{seed}")
    run_paths = set_active_run(run_id)
    print(f"Test run ID: {run_id}")

    try:
        write_run_metadata(extra={"phase": "test", "seed": seed})

        run_complete_experiment(
            mesh_size=0.7,  # Coarser mesh for speed
            num_adaptations=2,  # Fewer iterations for speed
            epochs=100,  # Fewer epochs
            export_images=False,
            create_gifs=False,
            generate_report=True,  # Enable to test visualizations
        )
        print(f"Quick test completed successfully! Results: {run_paths['root']}")
        return True

    except Exception as e:
        print(f"Quick test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


## Removed run_parameter_study_example in favor of flexible hparams study


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
    """Clean up all temporary files from outputs/ and legacy directories."""
    import shutil

    print("Cleaning up all temporary files...")
    total_cleaned = 0

    # Clean outputs/
    run_cleanup()

    # Clean legacy directories if they exist
    legacy_dirs = [
        os.path.join(PROJECT_ROOT, "images"),
        os.path.join(PROJECT_ROOT, "reports"),
    ]

    for legacy_dir in legacy_dirs:
        if os.path.exists(legacy_dir):
            try:
                shutil.rmtree(legacy_dir)
                print(f"Removed legacy directory: {legacy_dir}")
                total_cleaned += 1
            except Exception as e:
                print(f"Error removing {legacy_dir}: {e}")

    print("Full cleanup completed")
    return True


def run_ablation_summary_plot(run_ids):
    """Create shaded mean±std ablation plot from a list of run IDs.

    Expects each run to have reports/histories.csv.
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
    import sys

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

        if mode == "test":
            success = run_quick_test()
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
                "Usage: python main.py [main|test|hparams|cleanup|cleanup-all|ablate-plot]"
            )
            print("")
            print("Commands:")
            print("  main         Run full experiment (default)")
            print("               --seed <int>   Override random seed")
            print("  test         Run quick test with reduced parameters")
            print("  hparams      Run hyperparameter study")
            print("               [grid.json]    Path to JSON grid file")
            print("               ['{...}']      Inline JSON grid")
            print("               --images       Export images for each run")
            print("  cleanup      Clean up temporary files from outputs/")
            print("               [run-id]       Clean specific run only")
            print("  cleanup-all  Clean up all temp files including legacy dirs")
            print("  ablate-plot  Generate shaded ablation plot")
            print("               <run-id> ...   One or more run IDs to aggregate")
            success = False
    else:
        # Default: run main experiment
        success = main()

    sys.exit(0 if success else 1)
