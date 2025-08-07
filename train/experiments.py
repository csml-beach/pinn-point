"""
High-level experiment runners for PINN adaptive mesh training.
Contains the main experiment orchestration functions.
"""

import torch
from geometry import create_initial_mesh, get_initial_mesh_data, export_vertex_coordinates, get_random_points
from fem_solver import solve_FEM, export_fem_solution, create_dataset, export_vertex_coordinates
from pinn_model import FeedForward
from training import train_model
from utils import fix_random_model_error, print_model_summary, create_directory_structure
from visualization import create_essential_visualizations
from config import DEVICE, DIRECTORY, TRAINING_CONFIG, MODEL_CONFIG


def run_adaptive_training(mesh, num_adaptations=None, epochs=None, export_images=False):
    """Run the main adaptive mesh training loop.
    
    Training Strategy:
    - PINN trains on initial mesh data throughout (fixed dataset)
    - Mesh refinement based on PINN residuals on current mesh
    - Error assessment against high-fidelity reference solution
    
    Args:
        mesh: Initial NGSolve mesh
        num_adaptations: Number of adaptation iterations
        epochs: Number of training epochs per iteration
        export_images: Whether to export visualization images
        
    Returns:
        FeedForward: Trained PINN model
    """
    if num_adaptations is None:
        num_adaptations = TRAINING_CONFIG["iterations"]
    if epochs is None:
        epochs = TRAINING_CONFIG["epochs"]
    
    print(f"Starting adaptive training on device: {DEVICE}")
    print(f"Configuration: {num_adaptations} iterations, {epochs} epochs per iteration")
    
    # Create directory structure
    create_directory_structure()
    
    # Create high-fidelity reference solution
    from mesh_refinement import create_reference_solution
    from config import MESH_CONFIG
    reference_mesh, reference_solution = create_reference_solution(
        mesh_size_factor=MESH_CONFIG["reference_mesh_factor"]
    )
    
    # Initial setup
    mesh_point_count_0, vertex_coordinates_0 = get_initial_mesh_data(mesh)
    gfu, fes = solve_FEM(mesh)
    vertex_array = export_vertex_coordinates(mesh)
    solution_array = export_fem_solution(mesh, gfu)
    mesh_x, mesh_y = vertex_array.T
    
    print(f"Initial mesh: {len(mesh_x):,} points")
    
    # Create dataset
    dataset = create_dataset(vertex_array, solution_array)
    
    # Initialize model
    model = FeedForward(mesh_x=mesh_x, mesh_y=mesh_y).to(DEVICE)
    model.mesh_point_history = [vertex_coordinates_0]
    model.mesh_point_count_history = [mesh_point_count_0]
    
    print(f"Model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} parameters")

    # Main training loop
    for iteration in range(num_adaptations):
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration + 1}/{num_adaptations}")
        print(f"{'='*60}")
        
        # Import here to avoid circular imports
        from mesh_refinement import adapt_mesh_and_train
        
        adapt_mesh_and_train(
            model, mesh, dataset, reference_mesh, reference_solution,
            epochs, export_images, iteration
        )
        
        # No dataset update needed - PINN trains on initial mesh data throughout
        # Refined mesh is used only for residual computation and further refinement
        
        print(f"Iteration {iteration + 1} completed. Current mesh: {len(model.mesh_x):,} points")

    print(f"\n{'='*60}")
    print("ADAPTIVE TRAINING COMPLETED")
    print(f"{'='*60}")
    
    return model


def run_adaptive_training_fair(initial_mesh, shared_dataset, reference_mesh, reference_solution,
                              num_adaptations, epochs, export_images):
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
    from pinn_model import FeedForward
    from training import train_model
    from mesh_refinement import adapt_mesh_and_train
    from config import DEVICE
    
    print("Starting adaptive training on device:", DEVICE)
    print(f"Configuration: {num_adaptations} iterations, {epochs} epochs per iteration")
    
    # Get initial mesh coordinates
    vertex_array = export_vertex_coordinates(initial_mesh)
    mesh_x, mesh_y = vertex_array.T
    
    print(f"Initial mesh: {len(mesh_x):,} points")
    
    # Initialize model with shared training data coordinates
    model = FeedForward(mesh_x=mesh_x, mesh_y=mesh_y).to(DEVICE)
    
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
        
        # Train and adapt mesh (uses shared training dataset + current mesh for residuals)
        # Enable image export for residual visualization (GIF creation)
        adapt_mesh_and_train(
            model, current_mesh, shared_dataset, reference_mesh, reference_solution,
            epochs, export_images=True, iteration=iteration  # Always export for GIF
        )
        
        # Update tracking
        vertex_array = export_vertex_coordinates(current_mesh)
        mesh_x, mesh_y = vertex_array.T
        print(f"Iteration {iteration + 1} completed. Current mesh: {len(mesh_x):,} points")
        
        model.mesh_point_history.append(vertex_array.clone())
        model.mesh_point_count_history.append(len(mesh_x))
    
    print(f"\n{'='*60}")
    print("ADAPTIVE TRAINING COMPLETED")
    print(f"{'='*60}")
    
    return model


def run_random_training_fair(initial_mesh, shared_dataset, reference_mesh, reference_solution,
                            num_adaptations, epochs, export_images, adaptive_model=None):
    """
    Run random point training using shared components for fair comparison.
    
    Args:
        initial_mesh: Initial mesh object (for domain bounds)
        shared_dataset: Shared training dataset (same as adaptive method)
        reference_mesh: High-fidelity reference mesh for error assessment
        reference_solution: High-fidelity reference solution for error assessment
        num_adaptations: Number of training iterations to match adaptive method
        epochs: Number of training epochs per iteration
        export_images: Whether to export visualization images
        adaptive_model: Trained adaptive model to match point progression (optional)
        
    Returns:
        FeedForward: Trained random point PINN model
    """
    from pinn_model import FeedForward
    from training import train_model
    from mesh_refinement import compute_model_error
    from geometry import get_random_points, create_initial_mesh
    from config import DEVICE
    
    print(f"\n{'='*60}")
    print("RANDOM POINT TRAINING COMPARISON")
    print(f"{'='*60}")
    print("Training strategy: Shared initial data + random residual points")
    print("Error assessment: Same reference solution as adaptive method")
    
    # Get TRUE initial mesh size (75 points) - create fresh mesh to avoid modified mesh
    fresh_initial_mesh = create_initial_mesh(maxh=0.7)  # Use same size as experiment
    vertex_array = export_vertex_coordinates(fresh_initial_mesh)
    mesh_x, mesh_y = vertex_array.T
    initial_point_count = len(mesh_x)
    
    print(f"Random model starting with TRUE initial count: {initial_point_count:,} points")
    
    # Initialize model with true initial coordinates (for fair comparison)
    model = FeedForward(mesh_x=mesh_x, mesh_y=mesh_y).to(DEVICE)
    
    # Store reference solution for consistent error assessment
    model.reference_mesh = reference_mesh
    model.reference_solution = reference_solution
    
    # Initialize tracking with true initial count
    model.mesh_point_count_history = [initial_point_count]
    model.total_error_history = []
    model.boundary_error_history = []
    
    # Training iterations matching adaptive method's progression
    for iteration in range(num_adaptations):
        print(f"\nRandom training iteration: {iteration + 1}/{num_adaptations}")
        
        # Use adaptive model's actual point count progression if available
        # Skip duplicates: indices 1, 3, 5, ... contain unique post-refinement counts  
        if adaptive_model:
            target_index = 1 + (iteration * 2)  # Maps iteration 0->1, 1->3, 2->5, etc.
            if target_index < len(adaptive_model.mesh_point_count_history):
                target_point_count = adaptive_model.mesh_point_count_history[target_index]
                print(f"Matching adaptive post-refinement: using {target_point_count:,} points (index {target_index})")
            else:
                # Fallback: use the last available count
                target_point_count = adaptive_model.mesh_point_count_history[-1]
                print(f"Using final adaptive point count: {target_point_count} points")
        else:
            # Fallback: use fixed progression
            if iteration == 0:
                target_point_count = int(initial_point_count * 1.2)  # First refinement
            else:
                target_point_count = int(initial_point_count * (1.2 ** (iteration + 1)))
            print(f"Using estimated progression: {target_point_count:,} points")
        
        # Update model's residual computation points to random locations
        random_x, random_y = get_random_points(mesh=initial_mesh, random_point_count=target_point_count)
        model.mesh_x = torch.tensor(random_x).to(DEVICE)
        model.mesh_y = torch.tensor(random_y).to(DEVICE)
        
        print(f"Using {target_point_count:,} random points for residual computation")
        
        # Train model (uses shared_dataset for data loss, random points for residual loss)
        train_model(model, shared_dataset, epochs, lr=TRAINING_CONFIG["lr"])
        
        # Error assessment using same reference solution as adaptive method
        compute_model_error(model, reference_mesh, reference_solution, 
                          export_images=False, iteration=iteration)
        
        model.mesh_point_count_history.append(target_point_count)
    
    print("Random training comparison completed")
    return model


def run_random_comparison(model, mesh, num_adaptations=None, epochs=None):
    """Run comparison with random point training.
    
    Args:
        model: Trained adaptive model for comparison
        mesh: NGSolve mesh
        num_adaptations: Number of training iterations
        epochs: Number of training epochs per iteration
        
    Returns:
        FeedForward: Random training model
    """
    if num_adaptations is None:
        num_adaptations = TRAINING_CONFIG["iterations"]
    if epochs is None:
        epochs = TRAINING_CONFIG["epochs"]
    
    print(f"\n{'='*60}")
    print("RANDOM POINT TRAINING COMPARISON")
    print(f"{'='*60}")
    
    # Get the final FEM solution for comparison
    gfu, fes = solve_FEM(mesh)
    
    # Initialize random model with same number of points as initial adaptive model
    initial_point_count = model.mesh_point_count_history[0] if model.mesh_point_count_history else 1000
    rand_x, rand_y = get_random_points(mesh=mesh, random_point_count=initial_point_count)
    rand_model = FeedForward(torch.tensor(rand_x), torch.tensor(rand_y))
    
    print(f"Random model initialized with {initial_point_count:,} random points")
    
    # Create dummy dataset for random model (using zeros since we don't have FEM data at random points)
    dummy_vertex_array = torch.stack([torch.tensor(rand_x), torch.tensor(rand_y)], dim=1)
    dummy_solution_array = torch.zeros(len(rand_x), 1)
    dummy_dataset = create_dataset(dummy_vertex_array, dummy_solution_array)

    for iteration in range(num_adaptations):
        print(f"\nRandom training iteration: {iteration + 1}/{num_adaptations}")
        
        # Get random points matching the adaptive model's mesh size at this iteration
        if iteration < len(model.mesh_point_count_history):
            target_point_count = model.mesh_point_count_history[iteration]
        else:
            target_point_count = model.mesh_point_count_history[-1]
        
        train_x, train_y = get_random_points(mesh=mesh, random_point_count=target_point_count)
        rand_model.mesh_x = torch.tensor(train_x)
        rand_model.mesh_y = torch.tensor(train_y)

        # Train random model
        train_model(rand_model, dummy_dataset, epochs, lr=TRAINING_CONFIG["lr"])

        # Evaluate the random model on its own mesh (for consistent error calculation)
        mesh_x, mesh_y = export_vertex_coordinates(mesh).unbind(1)
        eval_x = torch.tensor(mesh_x)
        eval_y = torch.tensor(mesh_y)

        rand_model.mesh_point_count_history.append(len(rand_model.mesh_x))
        fix_random_model_error(rand_model, mesh, gfu, eval_x, eval_y, fes)
    
    print(f"Random training comparison completed")
    return rand_model


def run_complete_experiment(mesh_size=None, num_adaptations=None, epochs=None, export_images=False, 
                          create_gifs=True, generate_report=True, methods_to_run=None):
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
        from config import MESH_CONFIG
        mesh_size = MESH_CONFIG["maxh"]
    if num_adaptations is None:
        num_adaptations = TRAINING_CONFIG["iterations"]
    if epochs is None:
        epochs = TRAINING_CONFIG["epochs"]
    if methods_to_run is None:
        methods_to_run = ['adaptive', 'random']  # Default to current methods
    if epochs is None:
        epochs = TRAINING_CONFIG["epochs"]
    
    print(f"\n{'='*80}")
    print("STARTING COMPLETE PINN ADAPTIVE MESH EXPERIMENT")
    print(f"{'='*80}")
    print(f"Mesh size: {mesh_size}")
    print(f"Iterations: {num_adaptations}")
    print(f"Epochs per iteration: {epochs}")
    print(f"Export images: {export_images}")
    print(f"Methods to test: {', '.join(methods_to_run)}")
    print(f"Device: {DEVICE}")
    
    # Create shared components for fair comparison
    print("\n" + "="*60)
    print("CREATING SHARED COMPONENTS FOR FAIR COMPARISON")
    print("="*60)
    
    # 1. Create initial mesh and training dataset (shared by both methods)
    print("Creating initial mesh and training dataset...")
    initial_mesh = create_initial_mesh(maxh=mesh_size)
    gfu, fes = solve_FEM(initial_mesh)
    vertex_array = export_vertex_coordinates(initial_mesh)
    solution_array = export_fem_solution(initial_mesh, gfu)
    shared_training_dataset = create_dataset(vertex_array, solution_array)
    mesh_x, mesh_y = vertex_array.T
    print(f"Shared training dataset: {len(mesh_x):,} points")
    
    # 2. Create high-fidelity reference solution (shared by both methods)
    print("Creating high-fidelity reference solution...")
    from mesh_refinement import create_reference_solution
    from config import MESH_CONFIG
    reference_mesh, reference_solution = create_reference_solution(
        mesh_size_factor=MESH_CONFIG["reference_mesh_factor"]
    )
    ref_vertex_count = len(export_vertex_coordinates(reference_mesh))
    print(f"Reference solution: {ref_vertex_count:,} points")
    print("This reference solution will be used for all error computations")
    
    # Dictionary to store all trained models
    trained_models = {}
    
    # Run each requested method with shared components
    for method in methods_to_run:
        print(f"\n{'='*60}")
        print(f"RUNNING METHOD: {method.upper()}")
        print(f"{'='*60}")
        
        if method == 'adaptive':
            print("Starting adaptive mesh training...")
            model = run_adaptive_training_fair(
                initial_mesh, shared_training_dataset, reference_mesh, reference_solution,
                num_adaptations, epochs, export_images
            )
            trained_models['adaptive'] = model
            
        elif method == 'random':
            if 'adaptive' in trained_models:
                print("Starting random point training...")
                model = run_random_training_fair(
                    initial_mesh, shared_training_dataset, reference_mesh, reference_solution,
                    num_adaptations, epochs, export_images, adaptive_model=trained_models['adaptive']
                )
                trained_models['random'] = model
            else:
                print("Starting random point training (without adaptive reference)...")
                model = run_random_training_fair(
                    initial_mesh, shared_training_dataset, reference_mesh, reference_solution,
                    num_adaptations, epochs, export_images
                )
                trained_models['random'] = model
            
        else:
            print(f"Warning: Unknown method '{method}' - skipping")
    
    # Print summaries for all models
    for method_name, model in trained_models.items():
        print(f"\n{method_name.title()} Model Summary:")
        print_model_summary(model)
    
    # Create essential visualizations with residual GIF
    if generate_report and 'adaptive' in trained_models and 'random' in trained_models:
        print("\nGenerating essential visualizations...")
        create_essential_visualizations(
            trained_models['adaptive'], 
            trained_models['random'], 
            output_dir=DIRECTORY,
            include_residual_gif=True  # Always include the residual GIF
        )

    print(f"\n{'='*80}")
    print("EXPERIMENT COMPLETED SUCCESSFULLY")
    print(f"{'='*80}")
    
    return trained_models


def run_parameter_study(mesh_sizes=None, num_adaptations_list=None, epochs_list=None):
    """Run a parameter study with different configurations.
    
    Args:
        mesh_sizes: List of mesh sizes to test
        num_adaptations_list: List of adaptation counts to test
        epochs_list: List of epoch counts to test
        
    Returns:
        dict: Results from parameter study
    """
    if mesh_sizes is None:
        mesh_sizes = [0.3, 0.5, 0.7]
    if num_adaptations_list is None:
        num_adaptations_list = [5, 10, 15]
    if epochs_list is None:
        epochs_list = [1000, 2000, 3000]
    
    results = {}
    
    print(f"\n{'='*80}")
    print("STARTING PARAMETER STUDY")
    print(f"{'='*80}")
    
    for mesh_size in mesh_sizes:
        for num_adaptations in num_adaptations_list:
            for epochs in epochs_list:
                config_name = f"mesh_{mesh_size}_iter_{num_adaptations}_epochs_{epochs}"
                print(f"\nRunning configuration: {config_name}")
                
                try:
                    adaptive_model, random_model = run_complete_experiment(
                        mesh_size=mesh_size,
                        num_adaptations=num_adaptations,
                        epochs=epochs,
                        export_images=False,
                        create_gifs=False,
                        generate_report=False
                    )
                    
                    results[config_name] = {
                        "adaptive_model": adaptive_model,
                        "random_model": random_model,
                        "config": {
                            "mesh_size": mesh_size,
                            "num_adaptations": num_adaptations,
                            "epochs": epochs
                        }
                    }
                    
                except Exception as e:
                    print(f"Error in configuration {config_name}: {e}")
                    results[config_name] = {"error": str(e)}
    
    print(f"\n{'='*80}")
    print("PARAMETER STUDY COMPLETED")
    print(f"{'='*80}")
    
    return results
