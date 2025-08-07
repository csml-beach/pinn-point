"""
Simplified visualization module for PINN experiments.
Focuses on the most scientifically valuable plots with clean, publication-ready output.
"""

import os
from PIL import Image

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend - MUST be before pyplot import
import matplotlib.pyplot as plt
plt.ioff()  # Turn off interactive mode

# PyVista for 3D mesh visualization (imported on demand to handle missing dependency)
try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False
    print("Warning: PyVista not available. 3D mesh visualizations will be disabled.")

from config import DIRECTORY, VIZ_CONFIG


def export_to_png(mesh, gfu, fieldname, filename, size=None):
    """Export mesh solution to PNG image using PyVista.
    
    Args:
        mesh: NGSolve mesh
        gfu: GridFunction or solution field
        fieldname: Name of the field for visualization
        filename: Output filename
        size: Image size (uses config default if None)
        
    Returns:
        None
    """
    if not PYVISTA_AVAILABLE:
        print(f"Warning: Cannot export {filename} - PyVista not available")
        return
        
    if size is None:
        size = VIZ_CONFIG["image_size"]
    
    # Import NGSolve functions on demand
    try:
        from ngsolve import VTKOutput
    except ImportError:
        print("Warning: NGSolve not available for mesh export")
        return
    
    # Export to VTK format
    vtk = VTKOutput(
        mesh, coefs=[gfu], names=[fieldname], filename="./vtk_export", subdivision=0
    )
    vtk.Do()

    # Read the VTU file with PyVista
    try:
        meshpv = pv.read("./vtk_export.vtu")
    except Exception as e:
        print(f"Warning: Could not read VTK file for visualization: {e}")
        return

    # Create visualization if the field exists
    if fieldname in meshpv.point_data:
        try:
            # Try off-screen rendering first (no interactive windows)
            plotter = pv.Plotter(window_size=[size, size], off_screen=True)
            
            # Add mesh with same colormap for both errors and residuals
            if "error" in fieldname.lower():
                # Use same 'bwr' colormap as residuals for consistency
                plotter.add_mesh(meshpv, scalars=fieldname, show_scalar_bar=True, cmap="bwr", opacity=0.8)
            else:
                # Use blue-white-red for other fields (residuals, etc.)
                plotter.add_mesh(meshpv, scalars=fieldname, show_scalar_bar=True, cmap="bwr", opacity=0.8)
            
            # Add wireframe in black for better visibility
            plotter.add_mesh(
                meshpv, color="black", style="wireframe", show_scalar_bar=False, line_width=1
            )
            plotter.view_xy()
            
            # Configure scalar bar
            plotter.scalar_bar.SetPosition(0.85, 0.15)
            plotter.scalar_bar.SetOrientationToVertical()
            plotter.scalar_bar.SetWidth(0.05)
            plotter.scalar_bar.SetHeight(0.7)
            plotter.scalar_bar.SetLabelFormat("%2.2e")

            # Save screenshot using off-screen rendering
            output_path = os.path.join(DIRECTORY, filename)
            plotter.screenshot(output_path)
            plotter.close()  # Explicitly close the plotter
            print(f"Exported visualization to {output_path}")
            
        except Exception as e:
            print(f"Off-screen rendering failed: {e}")
            try:
                # Fallback: use show with screenshot but print warning
                print("Falling back to interactive mode - you may need to close the plot window")
                plotter = pv.Plotter(window_size=[size, size])
                
                # Add mesh with same colormap for both errors and residuals
                if "error" in fieldname.lower():
                    # Use same 'bwr' colormap as residuals for consistency
                    plotter.add_mesh(meshpv, scalars=fieldname, show_scalar_bar=True, cmap="bwr", opacity=0.8)
                else:
                    # Use blue-white-red for other fields (residuals, etc.)
                    plotter.add_mesh(meshpv, scalars=fieldname, show_scalar_bar=True, cmap="bwr", opacity=0.8)
                    plotter.add_mesh(meshpv, scalars=fieldname, show_scalar_bar=True, cmap="bwr", opacity=0.8)
                
                # Add wireframe in black for better visibility
                plotter.add_mesh(
                    meshpv, color="black", style="wireframe", show_scalar_bar=False, line_width=1
                )
                plotter.view_xy()
                
                # Configure scalar bar
                plotter.scalar_bar.SetPosition(0.85, 0.15)
                plotter.scalar_bar.SetOrientationToVertical()
                plotter.scalar_bar.SetWidth(0.05)
                plotter.scalar_bar.SetHeight(0.7)
                plotter.scalar_bar.SetLabelFormat("%2.2e")

                output_path = os.path.join(DIRECTORY, filename)
                plotter.show(screenshot=output_path)
                print(f"Exported visualization to {output_path}")
            except Exception as e2:
                print(f"Visualization export failed completely: {e2}")
    else:
        print(f"Warning: Field '{fieldname}' not found in mesh data")


def ensure_figure_closed(func):
    """Decorator to ensure matplotlib figures are always closed."""
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            plt.close('all')  # Close all figures to prevent hanging
    return wrapper


@ensure_figure_closed
def plot_method_comparison(adaptive_model, random_model, save_path=None):
    """
    Create a clean comparison plot showing the key performance metrics.
    
    Args:
        adaptive_model: Trained adaptive PINN model
        random_model: Trained random PINN model  
        save_path: Optional path to save the plot
    """
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Error comparison
        ax1 = axes[0]
        if adaptive_model.total_error_history and random_model.total_error_history:
            iterations = range(len(adaptive_model.total_error_history))
            ax1.semilogy(iterations, adaptive_model.total_error_history, 
                        'b-o', label='Adaptive Mesh', linewidth=2, markersize=6)
            ax1.semilogy(iterations, random_model.total_error_history, 
                        'r--s', label='Random Points', linewidth=2, markersize=6)
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('Total Error')
            ax1.set_title('Error Reduction Comparison')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # Point count progression
        ax2 = axes[1] 
        if adaptive_model.mesh_point_count_history:
            # Remove duplicates for cleaner visualization
            unique_counts = []
            prev_count = None
            for count in adaptive_model.mesh_point_count_history:
                if count != prev_count:
                    unique_counts.append(count)
                    prev_count = count
            
            iterations = range(len(unique_counts))
            ax2.plot(iterations, unique_counts, 'b-o', linewidth=2, markersize=8)
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('Number of Points')
            ax2.set_title('Mesh Refinement Progression')
            ax2.grid(True, alpha=0.3)
            
            # Add refinement factor annotation
            if len(unique_counts) > 1:
                factor = unique_counts[-1] / unique_counts[0]
                ax2.text(0.02, 0.98, f'Refinement: ×{factor:.2f}', 
                        transform=ax2.transAxes, va='top', 
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Method comparison saved to {save_path}")
        
        plt.close()  # Always close to prevent interactive display
            
    except Exception as e:
        print(f"Error creating method comparison plot: {e}")
        import traceback
        traceback.print_exc()
        plt.close()  # Close on error too


def create_performance_summary(adaptive_model, random_model, save_path=None):
    """
    Create a text summary of key performance metrics.
    
    Args:
        adaptive_model: Trained adaptive PINN model
        random_model: Trained random PINN model
        save_path: Optional path to save the summary
    """
    summary_lines = []
    summary_lines.append("PINN ADAPTIVE MESH EXPERIMENT SUMMARY")
    summary_lines.append("=" * 50)
    summary_lines.append("")
    
    # Adaptive model metrics
    summary_lines.append("ADAPTIVE MESH METHOD:")
    if adaptive_model.mesh_point_count_history:
        initial = adaptive_model.mesh_point_count_history[0]
        final = adaptive_model.mesh_point_count_history[-1]
        factor = final / initial
        summary_lines.append(f"  Mesh progression: {initial:,} → {final:,} points (×{factor:.2f})")
    
    if adaptive_model.total_error_history:
        initial_error = adaptive_model.total_error_history[0]
        final_error = adaptive_model.total_error_history[-1]
        reduction = initial_error / final_error
        summary_lines.append(f"  Error reduction: {initial_error:.2e} → {final_error:.2e} (×{reduction:.2f})")
    
    summary_lines.append("")
    
    # Random model metrics
    summary_lines.append("RANDOM POINTS METHOD:")
    if random_model.total_error_history:
        final_random_error = random_model.total_error_history[-1]
        summary_lines.append(f"  Final error: {final_random_error:.2e}")
    
    summary_lines.append("")
    
    # Comparison
    if (adaptive_model.total_error_history and random_model.total_error_history):
        adaptive_final = adaptive_model.total_error_history[-1]
        random_final = random_model.total_error_history[-1]
        improvement = ((random_final - adaptive_final) / random_final) * 100
        summary_lines.append("COMPARISON:")
        summary_lines.append(f"  Adaptive advantage: {improvement:.1f}% better error rate")
    
    summary_text = "\n".join(summary_lines)
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(summary_text)
        print(f"Performance summary saved to {save_path}")
    else:
        print(summary_text)
    
    return summary_text


@ensure_figure_closed
def plot_training_convergence_simple(model, model_name, save_path=None):
    """
    Create a simple training convergence plot.
    
    Args:
        model: PINN model with training history
        model_name: Name for the model (for title)
        save_path: Optional path to save the plot
    """
    if not model.train_loss_history:
        print(f"No training history available for {model_name}")
        return
    
    plt.figure(figsize=(8, 5))
    
    # Convert nested loss history to flat array if needed
    losses = []
    for loss_data in model.train_loss_history:
        if isinstance(loss_data, list):
            losses.extend(loss_data)
        else:
            losses.append(loss_data)
    
    plt.semilogy(losses, 'b-', linewidth=1.5)
    plt.xlabel('Training Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Convergence - {model_name}')
    plt.grid(True, alpha=0.3)
    
    # Add final loss annotation
    if losses:
        final_loss = losses[-1]
        plt.text(0.98, 0.95, f'Final Loss: {final_loss:.2e}', 
                transform=plt.gca().transAxes, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training convergence plot saved to {save_path}")
    
    plt.close()  # Always close to prevent interactive display


def create_residual_gif(folder_path, output_filename="adaptive_residual_evolution.gif", duration=None, loop=None, cleanup_pngs=True):
    """
    Create an animated GIF showing residual field evolution (adaptive method only).
    This visualization shows how adaptive mesh refinement zooms in on high-residual regions.
    
    Args:
        folder_path: Path to folder containing residual images
        output_filename: Name of output GIF file
        duration: Duration between frames in milliseconds
        loop: Number of loops (0 = infinite)
        cleanup_pngs: Whether to delete PNG files after GIF creation
        
    Returns:
        None
    """
    if duration is None:
        duration = VIZ_CONFIG["gif_duration"]
    if loop is None:
        loop = VIZ_CONFIG["gif_loop"]
    
    images = []
    
    # Find all residual images
    try:
        residual_files = []
        for file_name in os.listdir(folder_path):
            if file_name.lower().startswith("residuals") and file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                residual_files.append(file_name)
        
        # Sort by iteration number (extract from filename like "residuals_iter_1.png")
        residual_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]) if '_' in x and x.split('_')[-1].split('.')[0].isdigit() else 0)
        
        for file_name in residual_files:
            image_path = os.path.join(folder_path, file_name)
            try:
                img = Image.open(image_path)
                images.append(img)
            except Exception as e:
                print(f"Warning: Could not load residual image {image_path}: {e}")
                
    except FileNotFoundError:
        print(f"Warning: Folder {folder_path} not found")
        return

    if not images:
        print("No residual images found - GIF creation skipped")
        print("Note: Enable 'export_images=True' in experiment to generate residual fields")
        return

    # Create animated GIF
    try:
        output_path = os.path.join(folder_path, output_filename)
        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=loop,
        )
        print(f"✓ Residual evolution GIF created: {output_path}")
        print(f"  Shows adaptive refinement focusing on {len(images)} iterations")
        
        # Clean up PNG files if requested
        if cleanup_pngs:
            for file_name in residual_files:
                png_path = os.path.join(folder_path, file_name)
                try:
                    os.remove(png_path)
                    print(f"  Cleaned up: {file_name}")
                except Exception as e:
                    print(f"  Warning: Could not remove {file_name}: {e}")
            print(f"  Cleaned up {len(residual_files)} residual PNG files")
        
    except Exception as e:
        print(f"Error creating residual GIF: {e}")


def create_error_gif(folder_path, output_filename="adaptive_error_evolution.gif", duration=None, loop=None, cleanup_pngs=True):
    """
    Create an animated GIF showing error field evolution (adaptive method only).
    This visualization shows how the error between PINN predictions and reference solution changes.
    
    Args:
        folder_path: Path to folder containing error images
        output_filename: Name of output GIF file
        duration: Duration between frames in milliseconds
        loop: Number of loops (0 = infinite)
        cleanup_pngs: Whether to delete PNG files after GIF creation
        
    Returns:
        None
    """
    if duration is None:
        duration = VIZ_CONFIG["gif_duration"]
    if loop is None:
        loop = VIZ_CONFIG["gif_loop"]
    
    images = []
    
    # Find all error images
    try:
        error_files = []
        for file_name in os.listdir(folder_path):
            if file_name.lower().startswith("errors") and file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                error_files.append(file_name)
        
        # Sort by iteration number (extract from filename like "errors_1.png")
        error_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]) if '_' in x and x.split('_')[-1].split('.')[0].isdigit() else 0)
        
        for file_name in error_files:
            image_path = os.path.join(folder_path, file_name)
            try:
                img = Image.open(image_path)
                images.append(img)
            except Exception as e:
                print(f"Warning: Could not load error image {image_path}: {e}")
                
    except FileNotFoundError:
        print(f"Warning: Folder {folder_path} not found")
        return

    if not images:
        print("No error images found - GIF creation skipped")
        print("Note: Enable 'export_images=True' in experiment to generate error fields")
        return

    # Create animated GIF
    try:
        output_path = os.path.join(folder_path, output_filename)
        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=loop,
        )
        print(f"✓ Error evolution GIF created: {output_path}")
        print(f"  Shows error field evolution over {len(images)} iterations")
        
        # Clean up PNG files if requested
        if cleanup_pngs:
            for file_name in error_files:
                png_path = os.path.join(folder_path, file_name)
                try:
                    os.remove(png_path)
                    print(f"  Cleaned up: {file_name}")
                except Exception as e:
                    print(f"  Warning: Could not remove {file_name}: {e}")
            print(f"  Cleaned up {len(error_files)} error PNG files")
        
    except Exception as e:
        print(f"Error creating error GIF: {e}")


def create_random_residual_gif(folder_path, output_filename="random_residual_evolution.gif", duration=None, loop=None, cleanup_pngs=True):
    """
    Create an animated GIF showing random residual field evolution.
    This visualization shows how residuals evolve with random point training.
    
    Args:
        folder_path: Path to folder containing random residual images
        output_filename: Name of output GIF file
        duration: Duration between frames in milliseconds
        loop: Number of loops (0 = infinite)
        cleanup_pngs: Whether to delete PNG files after GIF creation
        
    Returns:
        None
    """
    if duration is None:
        duration = VIZ_CONFIG["gif_duration"]
    if loop is None:
        loop = VIZ_CONFIG["gif_loop"]
    
    images = []
    
    # Find all random residual images
    try:
        residual_files = []
        for file_name in os.listdir(folder_path):
            if file_name.lower().startswith("random_residuals") and file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                residual_files.append(file_name)
        
        # Sort by iteration number (extract from filename like "random_residuals_1.png")
        residual_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]) if '_' in x and x.split('_')[-1].split('.')[0].isdigit() else 0)
        
        for file_name in residual_files:
            image_path = os.path.join(folder_path, file_name)
            try:
                img = Image.open(image_path)
                images.append(img)
            except Exception as e:
                print(f"Warning: Could not load random residual image {image_path}: {e}")
                
    except FileNotFoundError:
        print(f"Warning: Folder {folder_path} not found")
        return

    if not images:
        print("No random residual images found - GIF creation skipped")
        print("Note: Enable 'export_images=True' in experiment to generate random residual fields")
        return

    # Create animated GIF
    try:
        output_path = os.path.join(folder_path, output_filename)
        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=loop,
        )
        print(f"✓ Random residual evolution GIF created: {output_path}")
        print(f"  Shows random point residual evolution over {len(images)} iterations")
        
        # Clean up PNG files if requested
        if cleanup_pngs:
            for file_name in residual_files:
                png_path = os.path.join(folder_path, file_name)
                try:
                    os.remove(png_path)
                    print(f"  Cleaned up: {file_name}")
                except Exception as e:
                    print(f"  Warning: Could not remove {file_name}: {e}")
            print(f"  Cleaned up {len(residual_files)} random residual PNG files")
        
    except Exception as e:
        print(f"Error creating random residual GIF: {e}")


def create_random_error_gif(folder_path, output_filename="random_error_evolution.gif", duration=None, loop=None, cleanup_pngs=True):
    """
    Create an animated GIF showing random error field evolution.
    This visualization shows how the error between random PINN predictions and reference solution changes.
    
    Args:
        folder_path: Path to folder containing random error images
        output_filename: Name of output GIF file
        duration: Duration between frames in milliseconds
        loop: Number of loops (0 = infinite)
        cleanup_pngs: Whether to delete PNG files after GIF creation
        
    Returns:
        None
    """
    if duration is None:
        duration = VIZ_CONFIG["gif_duration"]
    if loop is None:
        loop = VIZ_CONFIG["gif_loop"]
    
    images = []
    
    # Find all random error images
    try:
        error_files = []
        for file_name in os.listdir(folder_path):
            if file_name.lower().startswith("random_errors") and file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                error_files.append(file_name)
        
        # Sort by iteration number (extract from filename like "random_errors_1.png")
        error_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]) if '_' in x and x.split('_')[-1].split('.')[0].isdigit() else 0)
        
        for file_name in error_files:
            image_path = os.path.join(folder_path, file_name)
            try:
                img = Image.open(image_path)
                images.append(img)
            except Exception as e:
                print(f"Warning: Could not load random error image {image_path}: {e}")
                
    except FileNotFoundError:
        print(f"Warning: Folder {folder_path} not found")
        return

    if not images:
        print("No random error images found - GIF creation skipped")
        print("Note: Enable 'export_images=True' in experiment to generate random error fields")
        return

    # Create animated GIF
    try:
        output_path = os.path.join(folder_path, output_filename)
        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=loop,
        )
        print(f"✓ Random error evolution GIF created: {output_path}")
        print(f"  Shows random point error field evolution over {len(images)} iterations")
        
        # Clean up PNG files if requested
        if cleanup_pngs:
            for file_name in error_files:
                png_path = os.path.join(folder_path, file_name)
                try:
                    os.remove(png_path)
                    print(f"  Cleaned up: {file_name}")
                except Exception as e:
                    print(f"  Warning: Could not remove {file_name}: {e}")
            print(f"  Cleaned up {len(error_files)} random error PNG files")
        
    except Exception as e:
        print(f"Error creating random error GIF: {e}")


def create_essential_visualizations(adaptive_model, random_model, output_dir=None, include_gifs=True, cleanup_pngs=True):
    """
    Create the essential set of visualizations for the experiment.
    
    Args:
        adaptive_model: Trained adaptive PINN model
        random_model: Trained random PINN model
        output_dir: Directory to save plots (uses config default if None)
        include_gifs: Whether to create evolution GIFs for residuals and errors
        cleanup_pngs: Whether to clean up PNG files after GIF creation
    """
    if output_dir is None:
        output_dir = DIRECTORY
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print("Creating essential visualizations...")
    
    # 1. Method comparison (most important)
    comparison_path = os.path.join(output_dir, "method_comparison.png")
    plot_method_comparison(adaptive_model, random_model, comparison_path)
    
    # 2. Performance summary
    summary_path = os.path.join(output_dir, "performance_summary.txt")
    create_performance_summary(adaptive_model, random_model, summary_path)
    
    # 3. Training convergence plots
    adaptive_training_path = os.path.join(output_dir, "adaptive_training_convergence.png")
    plot_training_convergence_simple(adaptive_model, "Adaptive Mesh", adaptive_training_path)
    
    random_training_path = os.path.join(output_dir, "random_training_convergence.png")
    plot_training_convergence_simple(random_model, "Random Points", random_training_path)
    
    # 4. Evolution GIFs (shows adaptive refinement strategy and error evolution)
    if include_gifs:
        print("Creating adaptive residual evolution GIF...")
        gif_path = "adaptive_residual_evolution.gif"
        create_residual_gif(output_dir, gif_path, cleanup_pngs=cleanup_pngs)
        
        print("Creating adaptive error evolution GIF...")
        gif_path = "adaptive_error_evolution.gif"
        create_error_gif(output_dir, gif_path, cleanup_pngs=cleanup_pngs)
        
        print("Creating random residual evolution GIF...")
        gif_path = "random_residual_evolution.gif"
        create_random_residual_gif(output_dir, gif_path, cleanup_pngs=cleanup_pngs)
        
        print("Creating random error evolution GIF...")
        gif_path = "random_error_evolution.gif"
        create_random_error_gif(output_dir, gif_path, cleanup_pngs=cleanup_pngs)
    
    print(f"Essential visualizations saved to {output_dir}")
    if include_gifs:
        print("Key files: method_comparison.png, performance_summary.txt")
        print("  Adaptive GIFs: adaptive_residual_evolution.gif, adaptive_error_evolution.gif")
        print("  Random GIFs: random_residual_evolution.gif, random_error_evolution.gif")
    else:
        print("Key files: method_comparison.png, performance_summary.txt")


def create_detailed_visualizations(adaptive_model, random_model, reference_solution=None, output_dir=None):
    """
    Create detailed visualizations for in-depth analysis (optional).
    
    Args:
        adaptive_model: Trained adaptive PINN model
        random_model: Trained random PINN model  
        reference_solution: Reference solution for error visualization
        output_dir: Directory to save plots
    """
    if output_dir is None:
        output_dir = DIRECTORY
    
    print("Creating detailed visualizations...")
    
    # Only create if reference solution is available
    if reference_solution is not None:
        print("Note: Detailed error field visualizations require mesh export functionality")
        print("This feature can be added if needed for specific analysis")
    
    # Multi-metric comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Detailed Method Comparison', fontsize=16)
    
    # Total error
    if adaptive_model.total_error_history and random_model.total_error_history:
        axes[0,0].semilogy(adaptive_model.total_error_history, 'b-o', label='Adaptive')
        axes[0,0].semilogy(random_model.total_error_history, 'r--s', label='Random')
        axes[0,0].set_title('Total Error')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
    
    # Boundary error
    if adaptive_model.boundary_error_history and random_model.boundary_error_history:
        axes[0,1].semilogy(adaptive_model.boundary_error_history, 'b-o', label='Adaptive')
        axes[0,1].semilogy(random_model.boundary_error_history, 'r--s', label='Random')
        axes[0,1].set_title('Boundary Error')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
    
    # Residual history
    if adaptive_model.total_residual_history and random_model.total_residual_history:
        axes[1,0].semilogy(adaptive_model.total_residual_history, 'b-o', label='Adaptive')
        axes[1,0].semilogy(random_model.total_residual_history, 'r--s', label='Random')
        axes[1,0].set_title('Total Residual')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
    
    # Point count evolution (for adaptive only)
    if adaptive_model.mesh_point_count_history:
        unique_counts = []
        prev_count = None
        for count in adaptive_model.mesh_point_count_history:
            if count != prev_count:
                unique_counts.append(count)
                prev_count = count
        axes[1,1].plot(range(len(unique_counts)), unique_counts, 'b-o')
        axes[1,1].set_title('Mesh Point Evolution')
        axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    detailed_path = os.path.join(output_dir, "detailed_comparison.png")
    plt.savefig(detailed_path, dpi=300, bbox_inches='tight')
    plt.close()  # Always close to prevent interactive display
    
    print(f"Detailed comparison saved to {detailed_path}")
