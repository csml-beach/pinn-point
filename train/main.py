"""
Main execution script for PINN adaptive mesh training.
This is the entry point for running the complete experiment.
"""

from experiments import run_complete_experiment, run_parameter_study
from config import TRAINING_CONFIG, MESH_CONFIG
from utils import get_system_info, log_experiment_info, cleanup_gif_png_files, cleanup_all_temp_files
from paths import generate_run_id, set_active_run, write_run_metadata


def main():
    """Main function to run the complete PINN adaptive mesh training experiment."""

    print("PINN Adaptive Mesh Experiment")
    print("===========================")
    
    # Print system information
    system_info = get_system_info()
    print("\nSystem Information:")
    for key, value in system_info.items():
        print(f"  {key}: {value}")
    
    # Create a per-run output folder (outputs/<run-id>/...)
    run_id = generate_run_id("adapt-vs-rand")
    run_paths = set_active_run(run_id)
    print(f"\nRun ID: {run_id}")
    print(f"Outputs root: {run_paths['root']}")

    # Configuration
    mesh_size = MESH_CONFIG["maxh"]
    num_adaptations = TRAINING_CONFIG["iterations"]
    epochs = TRAINING_CONFIG["epochs"]
    export_images = True  # Set to True to save images during training (including error fields)
    
    print("\nExperiment Configuration:")
    print(f"  Initial mesh size: {mesh_size}")
    print(f"  Adaptation iterations: {num_adaptations}")
    print(f"  Training epochs per iteration: {epochs}")
    print(f"  Export images: {export_images}")
    
    try:
        # Write run metadata (configs + system + git) before run starts
        write_run_metadata(extra={"phase": "before_run"})

        # Run the complete experiment
        result = run_complete_experiment(
            mesh_size=mesh_size,
            num_adaptations=num_adaptations,
            epochs=epochs,
            export_images=export_images,
            create_gifs=export_images,  # Only create GIFs if images are exported
            generate_report=True,
            methods_to_run=['adaptive', 'random']  # Default methods
        )
        
        # Get models from the returned dictionary
        adaptive_model = result.get('adaptive')
        random_model = result.get('random')
        
        # Log experiment information
        config_info = {
            "mesh_size": mesh_size,
            "iterations": num_adaptations,
            "epochs": epochs,
            "export_images": export_images,
        }
        log_experiment_info(adaptive_model, config_info)
        # Update run metadata post-run
        write_run_metadata(extra={"phase": "after_run"})
        
        print("\n" + "="*60)
        print("EXPERIMENT SUMMARY")
        print("="*60)
        
        # Adaptive model results
        if adaptive_model and adaptive_model.mesh_point_count_history:
            initial_points = adaptive_model.mesh_point_count_history[0]
            final_points = adaptive_model.mesh_point_count_history[-1]
            refinement_factor = final_points / initial_points
            print("Adaptive Model:")
            print(f"  Mesh refinement: {initial_points:,} → {final_points:,} points (×{refinement_factor:.2f})")
        
        if adaptive_model and adaptive_model.total_error_history:
            initial_error = adaptive_model.total_error_history[0]
            final_error = adaptive_model.total_error_history[-1]
            error_reduction = initial_error / final_error if final_error > 0 else float('inf')
            print(f"  Error reduction: {initial_error:.2e} → {final_error:.2e} (×{error_reduction:.2f})")
        
        # Random model results
        if random_model:
            print("Random Model:")
            if random_model.total_error_history:
                final_random_error = random_model.total_error_history[-1]
                print(f"  Final error: {final_random_error:.2e}")
                
                if adaptive_model and adaptive_model.total_error_history:
                    final_adaptive_error = adaptive_model.total_error_history[-1]
                    improvement = final_random_error / final_adaptive_error if final_adaptive_error > 0 else float('inf')
                    print(f"  Adaptive vs Random improvement: ×{improvement:.2f}")
        
        print(f"\nResults saved to: {run_paths['root']}")
        print("="*60)

    except Exception as e:
        print(f"\nError during experiment: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def run_quick_test():
    """Run a quick test with reduced parameters for debugging."""
    print("Running quick test...")
    
    try:
        run_complete_experiment(
            mesh_size=0.7,      # Coarser mesh for speed
            num_adaptations=2,  # Fewer iterations for speed
            epochs=100,         # Fewer epochs
            export_images=False,
            create_gifs=False,
            generate_report=True  # Enable to test visualizations
        )
        print("Quick test completed successfully!")
        return True
        
    except Exception as e:
        print(f"Quick test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_parameter_study_example():
    """Run an example parameter study."""
    print("Running parameter study example...")
    
    # Define parameter ranges
    mesh_sizes = [0.5, 0.7]
    num_adaptations_list = [3, 5]
    epochs_list = [1000, 2000]
    
    try:
        results = run_parameter_study(
            mesh_sizes=mesh_sizes,
            num_adaptations_list=num_adaptations_list,
            epochs_list=epochs_list
        )
        
        print("\nParameter Study Results:")
        for config_name, result in results.items():
            if "error" in result:
                print(f"  {config_name}: FAILED - {result['error']}")
            else:
                adaptive_model = result["adaptive_model"]
                if adaptive_model.total_error_history:
                    final_error = adaptive_model.total_error_history[-1]
                    print(f"  {config_name}: Final error = {final_error:.2e}")
        
        return True
        
    except Exception as e:
        print(f"Parameter study failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_simple_cleanup():
    """Clean up GIF-related PNG files."""
    print("Cleaning up GIF-related PNG files...")
    try:
        results = cleanup_gif_png_files()
        print(f"✓ Cleaned up {len(results['files_deleted'])} PNG files")
        return True
    except Exception as e:
        print(f"Cleanup failed: {e}")
        return False


def run_full_cleanup():
    """Clean up all temporary files."""
    print("Cleaning up all temporary files...")
    try:
        cleanup_all_temp_files()
        print("✓ Full cleanup completed")
        return True
    except Exception as e:
        print(f"Full cleanup failed: {e}")
        return False


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
        if mode == "test":
            success = run_quick_test()
        elif mode == "study":
            success = run_parameter_study_example()
        elif mode == "main":
            success = main()
        elif mode == "cleanup":
            success = run_simple_cleanup()
        elif mode == "cleanup-all":
            success = run_full_cleanup()
        else:
            print("Usage: python main.py [main|test|study|cleanup|cleanup-all]")
            print("  main       - Run full experiment (default)")
            print("  test       - Run quick test with reduced parameters")
            print("  study      - Run parameter study example")
            print("  cleanup    - Clean up GIF-related PNG files")
            print("  cleanup-all - Clean up all temporary files")
            success = False
    else:
        # Default: run main experiment
        success = main()
    
    sys.exit(0 if success else 1)
