"""
Simplified visualization module for PINN experiments.
Focuses on the most scientifically valuable plots with clean, publication-ready output.
"""

import os
import csv
import tempfile
from typing import List
from PIL import Image
import numpy as np
from config import VIZ_CONFIG
import matplotlib
import matplotlib.pyplot as plt
from paths import (
    comparison_images_dir,
    images_dir,
    method_images_dir,
    reports_dir,
)


# Configure matplotlib backend before using pyplot
matplotlib.use("Agg")  # Use non-interactive backend - MUST be before pyplot import
plt.ioff()  # Turn off interactive mode

# PyVista for 3D mesh visualization (imported on demand to handle missing dependency)
try:
    import pyvista as pv

    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False
    print("Warning: PyVista not available. 3D mesh visualizations will be disabled.")


def export_to_png(mesh, gfu, fieldname, filename, size=None, output_dir=None):
    """Export mesh solution to PNG image using PyVista.

    Args:
        mesh: NGSolve mesh
        gfu: GridFunction or solution field
        fieldname: Name of the field for visualization
        filename: Output filename
        size: Image size (uses config default if None)
        output_dir: Output directory (uses root images dir if None)

    Returns:
        None
    """
    if not PYVISTA_AVAILABLE:
        print(f"Warning: Cannot export {filename} - PyVista not available")
        return

    if size is None:
        size = VIZ_CONFIG["image_size"]
    if output_dir is None:
        output_dir = images_dir()
    os.makedirs(output_dir, exist_ok=True)

    # Import NGSolve functions on demand
    try:
        from ngsolve import VTKOutput
    except ImportError:
        print("Warning: NGSolve not available for mesh export")
        return

    # Export to VTK format using temp directory to avoid polluting project root
    with tempfile.TemporaryDirectory() as tmpdir:
        vtk_path = os.path.join(tmpdir, "vtk_export")
        vtk = VTKOutput(
            mesh, coefs=[gfu], names=[fieldname], filename=vtk_path, subdivision=0
        )
        vtk.Do()

        # Read the VTU file with PyVista
        try:
            meshpv = pv.read(f"{vtk_path}.vtu")
        except Exception as e:
            print(f"Warning: Could not read VTK file for visualization: {e}")
            return

        # Determine fixed color limits if configured
        clim = None
        try:
            if "error" in fieldname.lower():
                clim = VIZ_CONFIG.get("error_clim")
            elif "residual" in fieldname.lower():
                clim = VIZ_CONFIG.get("residual_clim")
        except Exception:
            clim = None

        # Create visualization if the field exists
        if fieldname in meshpv.point_data:
            try:
                # Try off-screen rendering first (no interactive windows)
                plotter = pv.Plotter(window_size=[size, size], off_screen=True)

                # Add mesh with same colormap for both errors and residuals
                if "error" in fieldname.lower():
                    # Use same 'bwr' colormap as residuals for consistency
                    plotter.add_mesh(
                        meshpv,
                        scalars=fieldname,
                        show_scalar_bar=True,
                        cmap="bwr",
                        opacity=0.8,
                        clim=clim,
                    )
                else:
                    # Use blue-white-red for other fields (residuals, etc.)
                    plotter.add_mesh(
                        meshpv,
                        scalars=fieldname,
                        show_scalar_bar=True,
                        cmap="bwr",
                        opacity=0.8,
                        clim=clim,
                    )

                # Add wireframe in black for better visibility
                plotter.add_mesh(
                    meshpv,
                    color="black",
                    style="wireframe",
                    show_scalar_bar=False,
                    line_width=1,
                )
                plotter.view_xy()

                # Configure scalar bar
                plotter.scalar_bar.SetPosition(0.85, 0.15)
                plotter.scalar_bar.SetOrientationToVertical()
                plotter.scalar_bar.SetWidth(0.05)
                plotter.scalar_bar.SetHeight(0.7)
                plotter.scalar_bar.SetLabelFormat("%2.2e")

                # Save screenshot using off-screen rendering
                output_path = os.path.join(output_dir, filename)
                plotter.screenshot(output_path)
                plotter.close()  # Explicitly close the plotter
                print(f"Exported visualization to {output_path}")

            except Exception as e:
                print(f"Off-screen rendering failed: {e}")
                try:
                    # Fallback: use show with screenshot but print warning
                    print(
                        "Falling back to interactive mode - you may need to close the plot window"
                    )
                    plotter = pv.Plotter(window_size=[size, size])

                    # Add mesh with same colormap for both errors and residuals
                    if "error" in fieldname.lower():
                        # Use same 'bwr' colormap as residuals for consistency
                        plotter.add_mesh(
                            meshpv,
                            scalars=fieldname,
                            show_scalar_bar=True,
                            cmap="bwr",
                            opacity=0.8,
                            clim=clim,
                        )
                    else:
                        # Use blue-white-red for other fields (residuals, etc.)
                        plotter.add_mesh(
                            meshpv,
                            scalars=fieldname,
                            show_scalar_bar=True,
                            cmap="bwr",
                            opacity=0.8,
                            clim=clim,
                        )

                    # Add wireframe in black for better visibility
                    plotter.add_mesh(
                        meshpv,
                        color="black",
                        style="wireframe",
                        show_scalar_bar=False,
                        line_width=1,
                    )
                    plotter.view_xy()

                    # Configure scalar bar
                    plotter.scalar_bar.SetPosition(0.85, 0.15)
                    plotter.scalar_bar.SetOrientationToVertical()
                    plotter.scalar_bar.SetWidth(0.05)
                    plotter.scalar_bar.SetHeight(0.7)
                    plotter.scalar_bar.SetLabelFormat("%2.2e")

                    output_path = os.path.join(output_dir, filename)
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
            plt.close("all")  # Close all figures to prevent hanging

    return wrapper


def _load_multi_method_series(run_roots: List[str], field: str) -> dict[str, List[np.ndarray]]:
    """Collect a numeric field from all_methods_histories.csv across runs."""
    series_by_method: dict[str, List[np.ndarray]] = {}

    for root in run_roots:
        csv_path = os.path.join(root, "reports", "all_methods_histories.csv")
        if not os.path.exists(csv_path):
            continue

        per_method: dict[str, List[float]] = {}
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                method = (row.get("method") or "").strip()
                if not method:
                    continue
                raw_val = row.get(field, "")
                try:
                    value = float(raw_val) if raw_val not in ("", None) else np.nan
                except Exception:
                    value = np.nan
                per_method.setdefault(method, []).append(value)

        for method, values in per_method.items():
            if values:
                series_by_method.setdefault(method, []).append(
                    np.array(values, dtype=float)
                )

    return series_by_method


def _pad_stack(series: List[np.ndarray]) -> np.ndarray:
    if not series:
        return np.empty((0, 0))
    length = max(len(s) for s in series)
    padded = []
    for s in series:
        if len(s) < length:
            s = np.concatenate([s, np.full(length - len(s), np.nan)])
        padded.append(s)
    return np.vstack(padded)


@ensure_figure_closed
def plot_ablation_error_shaded(
    run_roots: List[str], save_path: str, title: str = "Error vs Iteration (mean ± std)"
):
    """Aggregate total_error from all_methods_histories.csv and plot mean±std shading."""
    series_by_method = _load_multi_method_series(run_roots, "total_error")
    if not series_by_method:
        print("No histories found; skipping ablation plot")
        return

    plt.figure(figsize=(7, 5))
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Total Error")
    plt.yscale("log")
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])

    for idx, method in enumerate(sorted(series_by_method)):
        stacked = _pad_stack(series_by_method[method])
        if stacked.size == 0:
            continue
        mean = np.nanmean(stacked, axis=0)
        std = np.nanstd(stacked, axis=0)
        x = np.arange(len(mean))
        color = color_cycle[idx % len(color_cycle)] if color_cycle else None
        plt.plot(x, mean, color=color, label=f"{method} (mean ± std)", linewidth=2)
        plt.fill_between(
            x,
            np.maximum(mean - std, 1e-32),
            mean + std,
            color=color,
            alpha=0.2,
        )

    plt.legend()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Ablation plot saved to {save_path}")


@ensure_figure_closed
def plot_ablation_fixed_residual_shaded(
    run_roots: List[str],
    save_path: str,
    title: str = "Fixed residual ∫Ω r^2 (mean ± std)",
):
    """Aggregate fixed_total_residual from all_methods_histories.csv and plot mean±std shading."""
    series_by_method = _load_multi_method_series(run_roots, "fixed_total_residual")
    if not series_by_method:
        print(
            "No fixed residual histories found; skipping ablation fixed residual plot"
        )
        return

    plt.figure(figsize=(7, 5))
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("∫Ω r(x)^2 dΩ")
    plt.yscale("log")
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])

    for idx, method in enumerate(sorted(series_by_method)):
        stacked = _pad_stack(series_by_method[method])
        if stacked.size == 0:
            continue
        mean = np.nanmean(stacked, axis=0)
        std = np.nanstd(stacked, axis=0)
        x = np.arange(len(mean))
        color = color_cycle[idx % len(color_cycle)] if color_cycle else None
        plt.plot(x, mean, color=color, label=f"{method} (mean ± std)", linewidth=2)
        plt.fill_between(
            x,
            np.maximum(mean - std, 1e-32),
            mean + std,
            color=color,
            alpha=0.2,
        )

    plt.legend()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Ablation fixed residual plot saved to {save_path}")


def _method_label(method_name: str) -> str:
    return method_name.replace("_", " ").title()


def _history_as_array(model, attr: str) -> np.ndarray:
    values = getattr(model, attr, []) or []
    if not values:
        return np.array([], dtype=float)
    cleaned = []
    for value in values:
        try:
            cleaned.append(float(value))
        except Exception:
            cleaned.append(np.nan)
    return np.asarray(cleaned, dtype=float)


def _aligned_history(model, y_attr: str, x_attr: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    y_values = _history_as_array(model, y_attr)
    if y_values.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    if x_attr is None:
        x_values = np.arange(1, len(y_values) + 1, dtype=float)
    else:
        x_values = _history_as_array(model, x_attr)
        if x_values.size == 0:
            x_values = np.arange(1, len(y_values) + 1, dtype=float)

    length = min(len(x_values), len(y_values))
    return x_values[:length], y_values[:length]


@ensure_figure_closed
def plot_multi_method_comparison(
    trained_models: dict,
    history_attr: str,
    save_path: str,
    *,
    title: str,
    ylabel: str,
    xlabel: str = "Iteration",
    logy: bool = True,
    x_attr: str | None = None,
):
    """Plot one tracked metric for every trained method."""
    plt.figure(figsize=(7, 5))
    plotted = False

    for method_name, model in sorted(trained_models.items()):
        x_values, y_values = _aligned_history(model, history_attr, x_attr=x_attr)
        if y_values.size == 0:
            continue

        if logy:
            positive = np.where(y_values > 0, y_values, np.nan)
            if np.all(np.isnan(positive)):
                continue
            plt.semilogy(
                x_values,
                positive,
                marker="o",
                linewidth=2,
                markersize=5,
                label=_method_label(method_name),
            )
        else:
            plt.plot(
                x_values,
                y_values,
                marker="o",
                linewidth=2,
                markersize=5,
                label=_method_label(method_name),
            )
        plotted = True

    if not plotted:
        print(f"No {history_attr} histories; skipping plot")
        return

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Comparison plot saved to {save_path}")


def create_multi_method_performance_summary(trained_models: dict, save_path: str | None = None):
    """Create a concise text summary for all trained methods."""
    lines = ["PINN EXPERIMENT SUMMARY", "=" * 50, ""]
    ranking: list[tuple[float, str]] = []
    ranking_metric = "error integral"

    for method_name, model in sorted(trained_models.items()):
        label = _method_label(method_name)
        lines.append(f"{label}:")

        point_counts = _history_as_array(model, "mesh_point_count_history")
        if point_counts.size:
            initial = int(point_counts[0])
            final = int(point_counts[-1])
            growth = final / initial if initial else float("nan")
            lines.append(f"  Points: {initial:,} -> {final:,} (x{growth:.2f})")

        total_error = _history_as_array(model, "total_error_history")
        relative_l2 = _history_as_array(model, "relative_l2_error_history")
        if relative_l2.size:
            lines.append(f"  Final relative L2 error: {relative_l2[-1]:.6e}")
            ranking.append((relative_l2[-1], method_name))
            ranking_metric = "relative L2 error"
        elif total_error.size:
            ranking.append((total_error[-1], method_name))
        if total_error.size:
            lines.append(f"  Final error integral: {total_error[-1]:.6e}")

        total_error_rms = _history_as_array(model, "total_error_rms_history")
        relative_error_rms = _history_as_array(model, "relative_error_rms_history")
        if total_error_rms.size:
            lines.append(f"  Final RMS error: {total_error_rms[-1]:.6e}")
        if relative_error_rms.size:
            lines.append(f"  Final relative RMS error: {relative_error_rms[-1]:.6e}")

        fixed_residual = _history_as_array(model, "fixed_total_residual_history")
        relative_fixed_l2_residual = _history_as_array(
            model, "relative_fixed_l2_residual_history"
        )
        fixed_rms_residual = _history_as_array(model, "fixed_rms_residual_history")
        relative_fixed_rms_residual = _history_as_array(
            model, "relative_fixed_rms_residual_history"
        )
        if fixed_residual.size:
            lines.append(f"  Final fixed residual integral: {fixed_residual[-1]:.6e}")
        if relative_fixed_l2_residual.size:
            lines.append(
                "  Final relative fixed L2 residual: "
                f"{relative_fixed_l2_residual[-1]:.6e}"
            )
        if fixed_rms_residual.size:
            lines.append(
                f"  Final fixed reference-mesh RMS residual: {fixed_rms_residual[-1]:.6e}"
            )
        if relative_fixed_rms_residual.size:
            lines.append(
                "  Final relative fixed RMS residual: "
                f"{relative_fixed_rms_residual[-1]:.6e}"
            )

        runtime = _history_as_array(model, "cumulative_runtime_history")
        if runtime.size:
            lines.append(f"  Cumulative runtime: {runtime[-1]:.2f}s")

        lines.append("")

    if ranking:
        lines.append(f"FINAL ERROR RANKING ({ranking_metric}):")
        for idx, (_, method_name) in enumerate(sorted(ranking), start=1):
            lines.append(f"  {idx}. {_method_label(method_name)}")

    summary_text = "\n".join(lines)
    if save_path:
        with open(save_path, "w") as file_obj:
            file_obj.write(summary_text)
        print(f"Performance summary saved to {save_path}")
    else:
        print(summary_text)
    return summary_text


def create_multi_method_point_usage_table(
    trained_models: dict, dataset_size: int, save_path: str | None = None
):
    """Create a per-iteration residual-point table for every method."""
    if save_path is None:
        os.makedirs(reports_dir(), exist_ok=True)
        save_path = os.path.join(reports_dir(), "point_usage_table.txt")

    method_names = sorted(trained_models)
    histories = {
        method_name: list(
            getattr(trained_models[method_name], "iteration_point_count_history", []) or []
        )
        for method_name in method_names
    }
    num_iterations = max((len(values) for values in histories.values()), default=0)
    labeled_batch = next(
        (
            getattr(model, "num_data", None)
            for model in trained_models.values()
            if getattr(model, "num_data", None) is not None
        ),
        "N/A",
    )

    header_cells = ["Iter", "LabeledDatasetSize", "LabeledBatchSize"] + [
        f"{_method_label(method_name)}Interior" for method_name in method_names
    ]
    widths = [6, 18, 18] + [max(18, len(cell) + 2) for cell in header_cells[3:]]

    def _format_row(values: list[object]) -> str:
        return " | ".join(str(value).rjust(width) for value, width in zip(values, widths))

    lines = ["PER-ITERATION POINT USAGE TABLE", "=" * 50, "", _format_row(header_cells)]
    lines.append("-" * len(lines[-1]))

    for iteration in range(num_iterations):
        row = [iteration + 1, dataset_size, labeled_batch]
        for method_name in method_names:
            history = histories[method_name]
            row.append(history[iteration] if iteration < len(history) else "N/A")
        lines.append(_format_row(row))

    lines.extend(
        [
            "",
            "Notes:",
            "- LabeledDatasetSize is the shared FEM supervision set size.",
            "- LabeledBatchSize is the shared supervised batch size used in loss_data.",
            "- Method columns report PDE residual collocation counts per iteration.",
        ]
    )

    with open(save_path, "w") as file_obj:
        file_obj.write("\n".join(lines))
    print(f"Point usage table saved to {save_path}")


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

    plt.semilogy(losses, "b-", linewidth=1.5)
    plt.xlabel("Training Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training Convergence - {model_name}")
    plt.grid(True, alpha=0.3)

    # Add final loss annotation
    if losses:
        final_loss = losses[-1]
        plt.text(
            0.98,
            0.95,
            f"Final Loss: {final_loss:.2e}",
            transform=plt.gca().transAxes,
            ha="right",
            va="top",
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.7),
        )

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Training convergence plot saved to {save_path}")

    plt.close()  # Always close to prevent interactive display


def create_residual_gif(
    folder_path,
    output_filename="adaptive_residual_evolution.gif",
    duration=None,
    loop=None,
    cleanup_pngs=True,
):
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
            if file_name.lower().startswith("residuals") and file_name.lower().endswith(
                (".png", ".jpg", ".jpeg")
            ):
                residual_files.append(file_name)

        # Sort by iteration number (extract from filename like "residuals_iter_1.png")
        residual_files.sort(
            key=lambda x: (
                int(x.split("_")[-1].split(".")[0])
                if "_" in x and x.split("_")[-1].split(".")[0].isdigit()
                else 0
            )
        )

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
        print(
            "Note: Enable 'export_images=True' in experiment to generate residual fields"
        )
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


def create_error_gif(
    folder_path,
    output_filename="adaptive_error_evolution.gif",
    duration=None,
    loop=None,
    cleanup_pngs=True,
):
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
            if file_name.lower().startswith("errors") and file_name.lower().endswith(
                (".png", ".jpg", ".jpeg")
            ):
                error_files.append(file_name)

        # Sort by iteration number (extract from filename like "errors_1.png")
        error_files.sort(
            key=lambda x: (
                int(x.split("_")[-1].split(".")[0])
                if "_" in x and x.split("_")[-1].split(".")[0].isdigit()
                else 0
            )
        )

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
        print(
            "Note: Enable 'export_images=True' in experiment to generate error fields"
        )
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


def create_random_residual_gif(
    folder_path,
    output_filename="random_residual_evolution.gif",
    duration=None,
    loop=None,
    cleanup_pngs=True,
):
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
            if file_name.lower().startswith(
                "random_residuals"
            ) and file_name.lower().endswith((".png", ".jpg", ".jpeg")):
                residual_files.append(file_name)

        # Sort by iteration number (extract from filename like "random_residuals_1.png")
        residual_files.sort(
            key=lambda x: (
                int(x.split("_")[-1].split(".")[0])
                if "_" in x and x.split("_")[-1].split(".")[0].isdigit()
                else 0
            )
        )

        for file_name in residual_files:
            image_path = os.path.join(folder_path, file_name)
            try:
                img = Image.open(image_path)
                images.append(img)
            except Exception as e:
                print(
                    f"Warning: Could not load random residual image {image_path}: {e}"
                )

    except FileNotFoundError:
        print(f"Warning: Folder {folder_path} not found")
        return

    if not images:
        print("No random residual images found - GIF creation skipped")
        print(
            "Note: Enable 'export_images=True' in experiment to generate random residual fields"
        )
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


def create_random_error_gif(
    folder_path,
    output_filename="random_error_evolution.gif",
    duration=None,
    loop=None,
    cleanup_pngs=True,
):
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
            if file_name.lower().startswith(
                "random_errors"
            ) and file_name.lower().endswith((".png", ".jpg", ".jpeg")):
                error_files.append(file_name)

        # Sort by iteration number (extract from filename like "random_errors_1.png")
        error_files.sort(
            key=lambda x: (
                int(x.split("_")[-1].split(".")[0])
                if "_" in x and x.split("_")[-1].split(".")[0].isdigit()
                else 0
            )
        )

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
        print(
            "Note: Enable 'export_images=True' in experiment to generate random error fields"
        )
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
        print(
            f"  Shows random point error field evolution over {len(images)} iterations"
        )

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


def create_essential_visualizations(
    adaptive_model, random_model, output_dir=None, include_gifs=True, cleanup_pngs=True
):
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
        output_dir = comparison_images_dir()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Creating essential visualizations...")

    # 1. Error comparisons: integrated and RMS
    err_int_path = os.path.join(output_dir, "error_integral_comparison.png")
    plot_error_integral_comparison(adaptive_model, random_model, err_int_path)
    err_rms_path = os.path.join(output_dir, "fixed_error_rms_comparison.png")
    plot_fixed_error_rms_comparison(adaptive_model, random_model, err_rms_path)

    # 1b. Fixed-grid residual integral comparison for fair evaluation
    fixed_residual_path = os.path.join(output_dir, "fixed_residual_comparison.png")
    try:
        plot_fixed_residual_comparison(
            adaptive_model, random_model, fixed_residual_path
        )
    except Exception as e:
        print(f"Warning: Failed to create fixed residual comparison plot: {e}")

    # 1b2. Fixed-grid residual integral comparison (∫ r^2)
    fixed_residual_int_path = os.path.join(
        output_dir, "fixed_residual_integral_comparison.png"
    )
    try:
        plot_fixed_residual_integral_comparison(
            adaptive_model, random_model, fixed_residual_int_path
        )
    except Exception as e:
        print(f"Warning: Failed to create fixed residual integral comparison plot: {e}")

    # 1c. Fixed-grid RMS error comparison
    fixed_error_rms_path = os.path.join(output_dir, "fixed_error_rms_comparison.png")
    try:
        plot_fixed_error_rms_comparison(
            adaptive_model, random_model, fixed_error_rms_path
        )
    except Exception as e:
        print(f"Warning: Failed to create fixed RMS error comparison plot: {e}")

    # 2. Performance summary
    os.makedirs(reports_dir(), exist_ok=True)
    summary_path = os.path.join(reports_dir(), "performance_summary.txt")
    create_performance_summary(adaptive_model, random_model, summary_path)

    # 2b. Interior points vs iteration (adaptive)
    points_path = os.path.join(output_dir, "points_vs_iteration.png")
    plot_points_vs_iteration(adaptive_model, points_path)

    # 3. Training convergence plots
    adaptive_dir = method_images_dir("adaptive")
    random_dir = method_images_dir("random")
    adaptive_training_path = os.path.join(adaptive_dir, "training_convergence.png")
    plot_training_convergence_simple(
        adaptive_model, "Adaptive Mesh", adaptive_training_path
    )

    random_training_path = os.path.join(random_dir, "training_convergence.png")
    plot_training_convergence_simple(
        random_model, "Random Points", random_training_path
    )

    # 4. Evolution GIFs (shows adaptive refinement strategy and error evolution)
    if include_gifs:
        print("Creating adaptive residual evolution GIF...")
        create_image_sequence_gif(
            adaptive_dir,
            "residuals_",
            "adaptive_residual_evolution.gif",
            description="Adaptive residual evolution",
            cleanup_pngs=cleanup_pngs,
        )

        print("Creating adaptive error evolution GIF...")
        create_image_sequence_gif(
            adaptive_dir,
            "errors_",
            "adaptive_error_evolution.gif",
            description="Adaptive error evolution",
            cleanup_pngs=cleanup_pngs,
        )

        print("Creating random residual evolution GIF...")
        create_image_sequence_gif(
            random_dir,
            "residuals_",
            "random_residual_evolution.gif",
            description="Random residual evolution",
            cleanup_pngs=cleanup_pngs,
        )

        print("Creating random error evolution GIF...")
        create_image_sequence_gif(
            random_dir,
            "errors_",
            "random_error_evolution.gif",
            description="Random error evolution",
            cleanup_pngs=cleanup_pngs,
        )

    print(f"Essential visualizations saved to {output_dir}")
    if include_gifs:
        print(
            "Key files: error_integral_comparison.png, fixed_error_rms_comparison.png, performance_summary.txt"
        )
        print(
            "  Adaptive GIFs: adaptive_residual_evolution.gif, adaptive_error_evolution.gif"
        )
        print(
            "  Random GIFs: random_residual_evolution.gif, random_error_evolution.gif"
        )
    else:
        print(
            "Key files: error_integral_comparison.png, fixed_error_rms_comparison.png, performance_summary.txt"
        )


def create_detailed_visualizations(
    adaptive_model, random_model, reference_solution=None, output_dir=None
):
    """
    Create detailed visualizations for in-depth analysis (optional).

    Args:
        adaptive_model: Trained adaptive PINN model
        random_model: Trained random PINN model
        reference_solution: Reference solution for error visualization
        output_dir: Directory to save plots
    """
    if output_dir is None:
        output_dir = comparison_images_dir()

    print("Creating detailed visualizations...")

    # Only create if reference solution is available
    if reference_solution is not None:
        print(
            "Note: Detailed error field visualizations require mesh export functionality"
        )
        print("This feature can be added if needed for specific analysis")

    # Multi-metric comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Detailed Method Comparison", fontsize=16)

    # Total error
    if adaptive_model.total_error_history and random_model.total_error_history:
        axes[0, 0].semilogy(adaptive_model.total_error_history, "b-o", label="Adaptive")
        axes[0, 0].semilogy(random_model.total_error_history, "r--s", label="Random")
        axes[0, 0].set_title("Total Error")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

    # Boundary error
    if adaptive_model.boundary_error_history and random_model.boundary_error_history:
        axes[0, 1].semilogy(
            adaptive_model.boundary_error_history, "b-o", label="Adaptive"
        )
        axes[0, 1].semilogy(random_model.boundary_error_history, "r--s", label="Random")
        axes[0, 1].set_title("Boundary Error")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

    # Residual history
    if adaptive_model.total_residual_history and random_model.total_residual_history:
        axes[1, 0].semilogy(
            adaptive_model.total_residual_history, "b-o", label="Adaptive"
        )
        axes[1, 0].semilogy(random_model.total_residual_history, "r--s", label="Random")
        axes[1, 0].set_title("Total Residual")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

    # Point count evolution (for adaptive only)
    if adaptive_model.mesh_point_count_history:
        unique_counts = []
        prev_count = None
        for count in adaptive_model.mesh_point_count_history:
            if count != prev_count:
                unique_counts.append(count)
                prev_count = count
        axes[1, 1].plot(range(len(unique_counts)), unique_counts, "b-o")
        axes[1, 1].set_title("Mesh Point Evolution")
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    detailed_path = os.path.join(output_dir, "detailed_comparison.png")
    plt.savefig(detailed_path, dpi=300, bbox_inches="tight")
    plt.close()  # Always close to prevent interactive display

    print(f"Detailed comparison saved to {detailed_path}")


def _iter_image_sequence(folder_path: str, prefix: str) -> list[str]:
    try:
        files = [
            file_name
            for file_name in os.listdir(folder_path)
            if file_name.lower().startswith(prefix.lower())
            and file_name.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
    except FileNotFoundError:
        return []

    files.sort(
        key=lambda value: (
            int(value.split("_")[-1].split(".")[0])
            if "_" in value and value.split("_")[-1].split(".")[0].isdigit()
            else 0
        )
    )
    return files


def create_image_sequence_gif(
    folder_path: str,
    prefix: str,
    output_filename: str,
    *,
    description: str,
    duration: int | None = None,
    loop: int | None = None,
    cleanup_pngs: bool = True,
):
    """Create a GIF from a prefixed image sequence when files are available."""
    if duration is None:
        duration = VIZ_CONFIG["gif_duration"]
    if loop is None:
        loop = VIZ_CONFIG["gif_loop"]

    image_files = _iter_image_sequence(folder_path, prefix)
    if not image_files:
        print(f"No images found for {prefix}; GIF creation skipped")
        return

    images = []
    for file_name in image_files:
        image_path = os.path.join(folder_path, file_name)
        try:
            images.append(Image.open(image_path))
        except Exception as exc:
            print(f"Warning: Could not load image {image_path}: {exc}")

    if not images:
        print(f"No loadable images found for {prefix}; GIF creation skipped")
        return

    output_path = os.path.join(folder_path, output_filename)
    try:
        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=loop,
        )
        print(f"GIF created: {output_path}")
        print(f"  {description} over {len(images)} iterations")
    except Exception as exc:
        print(f"Error creating GIF {output_filename}: {exc}")
        return

    if cleanup_pngs:
        for file_name in image_files:
            png_path = os.path.join(folder_path, file_name)
            try:
                os.remove(png_path)
            except Exception as exc:
                print(f"Warning: Could not remove {file_name}: {exc}")


def create_multi_method_visualizations(
    trained_models: dict,
    dataset_size: int,
    output_dir: str | None = None,
    include_gifs: bool = False,
    cleanup_pngs: bool = True,
):
    """Create the canonical visualization/report bundle for multi-method runs."""
    if output_dir is None:
        output_dir = comparison_images_dir()
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(reports_dir(), exist_ok=True)

    print("Creating multi-method visualizations...")

    plot_multi_method_comparison(
        trained_models,
        "total_error_history",
        os.path.join(output_dir, "error_integral_comparison.png"),
        title="Integrated error vs iteration",
        ylabel="Integrated error ∫Ω (u-u_ref)^2 dΩ",
    )
    plot_multi_method_comparison(
        trained_models,
        "relative_l2_error_history",
        os.path.join(output_dir, "relative_l2_error_comparison.png"),
        title="Relative L2 error vs iteration",
        ylabel="||u-u_ref||_L2 / ||u_ref||_L2",
        logy=False,
    )
    plot_multi_method_comparison(
        trained_models,
        "total_error_rms_history",
        os.path.join(output_dir, "fixed_error_rms_comparison.png"),
        title="Fixed reference-mesh RMS error",
        ylabel="RMS error (sqrt(mean((u-u_ref)^2)))",
    )
    plot_multi_method_comparison(
        trained_models,
        "relative_error_rms_history",
        os.path.join(output_dir, "relative_rms_error_comparison.png"),
        title="Relative RMS error vs iteration",
        ylabel="RMS(u-u_ref) / RMS(u_ref)",
        logy=False,
    )
    plot_multi_method_comparison(
        trained_models,
        "fixed_rms_residual_history",
        os.path.join(output_dir, "fixed_residual_comparison.png"),
        title="Fixed reference-mesh RMS residual",
        ylabel="RMS residual (sqrt(mean(r^2)))",
    )
    plot_multi_method_comparison(
        trained_models,
        "relative_fixed_rms_residual_history",
        os.path.join(output_dir, "relative_fixed_residual_rms_comparison.png"),
        title="Relative fixed reference-mesh RMS residual",
        ylabel="RMS(r) / RMS(f)",
        logy=False,
    )
    plot_multi_method_comparison(
        trained_models,
        "fixed_total_residual_history",
        os.path.join(output_dir, "fixed_residual_integral_comparison.png"),
        title="Fixed reference-mesh residual integral",
        ylabel="∫Ω r(x)^2 dΩ",
    )
    plot_multi_method_comparison(
        trained_models,
        "relative_fixed_l2_residual_history",
        os.path.join(output_dir, "relative_fixed_residual_comparison.png"),
        title="Relative fixed residual L2 vs iteration",
        ylabel="||r||_L2 / ||f||_L2",
        logy=False,
    )
    plot_multi_method_comparison(
        trained_models,
        "iteration_point_count_history",
        os.path.join(output_dir, "point_count_comparison.png"),
        title="Residual points vs iteration",
        ylabel="Number of residual points",
        logy=False,
    )
    plot_multi_method_comparison(
        trained_models,
        "cumulative_runtime_history",
        os.path.join(output_dir, "runtime_comparison.png"),
        title="Cumulative runtime vs iteration",
        ylabel="Runtime (s)",
        logy=False,
    )

    summary_path = os.path.join(reports_dir(), "performance_summary.txt")
    create_multi_method_performance_summary(trained_models, summary_path)

    point_usage_path = os.path.join(reports_dir(), "point_usage_table.txt")
    create_multi_method_point_usage_table(
        trained_models, dataset_size=dataset_size, save_path=point_usage_path
    )

    for method_name, model in sorted(trained_models.items()):
        method_output_dir = method_images_dir(method_name)
        plot_training_convergence_simple(
            model,
            _method_label(method_name),
            os.path.join(method_output_dir, "training_convergence.png"),
        )

    if include_gifs:
        if "adaptive" in trained_models:
            adaptive_dir = method_images_dir("adaptive")
            create_image_sequence_gif(
                adaptive_dir,
                "residuals_",
                "adaptive_residual_evolution.gif",
                description="Adaptive residual evolution",
                cleanup_pngs=cleanup_pngs,
            )
            create_image_sequence_gif(
                adaptive_dir,
                "errors_",
                "adaptive_error_evolution.gif",
                description="Adaptive error evolution",
                cleanup_pngs=cleanup_pngs,
            )
        if "random" in trained_models:
            random_dir = method_images_dir("random")
            create_image_sequence_gif(
                random_dir,
                "residuals_",
                "random_residual_evolution.gif",
                description="Random residual evolution",
                cleanup_pngs=cleanup_pngs,
            )
            create_image_sequence_gif(
                random_dir,
                "errors_",
                "random_error_evolution.gif",
                description="Random error evolution",
                cleanup_pngs=cleanup_pngs,
            )
        for method_name in sorted(set(trained_models) - {"adaptive", "random"}):
            method_dir = method_images_dir(method_name)
            create_image_sequence_gif(
                method_dir,
                "errors_",
                f"{method_name}_error_evolution.gif",
                description=f"{_method_label(method_name)} error evolution",
                cleanup_pngs=cleanup_pngs,
            )

    print(f"Multi-method visualizations saved to {output_dir}")
