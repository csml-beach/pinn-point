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
from config import DIRECTORY, REPORTS_DIRECTORY, VIZ_CONFIG
import matplotlib
import matplotlib.pyplot as plt

try:
    from paths import images_dir as _images_dir, reports_dir as _reports_dir

    images_dir = _images_dir
    reports_dir = _reports_dir
except Exception:

    def images_dir():
        return DIRECTORY

    def reports_dir():
        return REPORTS_DIRECTORY


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
                output_path = os.path.join(images_dir(), filename)
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

                    output_path = os.path.join(images_dir(), filename)
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


def write_histories_csv(
    adaptive_model,
    random_model,
    filename: str = "histories.csv",
    output_dir: str | None = None,
) -> str:
    """Write key training histories for both methods into a CSV for postprocessing.

    Columns: iteration, adaptive_total_error, adaptive_total_residual,
             random_total_error, random_total_residual,
             adaptive_fixed_total_residual, adaptive_fixed_boundary_residual,
             random_fixed_total_residual, random_fixed_boundary_residual

    Returns: path to the CSV file.
    """
    if output_dir is None:
        output_dir = reports_dir()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)

    a_err = list(getattr(adaptive_model, "total_error_history", []) or [])
    a_err_rms = list(getattr(adaptive_model, "total_error_rms_history", []) or [])
    a_res = list(getattr(adaptive_model, "total_residual_history", []) or [])
    a_fix_tot = list(getattr(adaptive_model, "fixed_total_residual_history", []) or [])
    a_fix_bnd = list(
        getattr(adaptive_model, "fixed_boundary_residual_history", []) or []
    )
    a_fix_rms = list(getattr(adaptive_model, "fixed_rms_residual_history", []) or [])

    r_err = list(getattr(random_model, "total_error_history", []) or [])
    r_err_rms = list(getattr(random_model, "total_error_rms_history", []) or [])
    r_res = list(getattr(random_model, "total_residual_history", []) or [])
    r_fix_tot = list(getattr(random_model, "fixed_total_residual_history", []) or [])
    r_fix_bnd = list(getattr(random_model, "fixed_boundary_residual_history", []) or [])
    r_fix_rms = list(getattr(random_model, "fixed_rms_residual_history", []) or [])

    n = max(
        len(a_err),
        len(a_err_rms),
        len(a_res),
        len(a_fix_tot),
        len(a_fix_bnd),
        len(a_fix_rms),
        len(r_err),
        len(r_err_rms),
        len(r_res),
        len(r_fix_tot),
        len(r_fix_bnd),
        len(r_fix_rms),
    )

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "iteration",
                "adaptive_total_error",
                "adaptive_total_residual",
                "random_total_error",
                "adaptive_total_error_rms",
                "random_total_error_rms",
                "random_total_residual",
                "adaptive_fixed_total_residual",
                "adaptive_fixed_boundary_residual",
                "random_fixed_total_residual",
                "random_fixed_boundary_residual",
                "adaptive_fixed_rms_residual",
                "random_fixed_rms_residual",
            ]
        )
        for i in range(n):
            row = [
                i,
                a_err[i] if i < len(a_err) else np.nan,
                a_res[i] if i < len(a_res) else np.nan,
                r_err[i] if i < len(r_err) else np.nan,
                a_err_rms[i] if i < len(a_err_rms) else np.nan,
                r_err_rms[i] if i < len(r_err_rms) else np.nan,
                r_res[i] if i < len(r_res) else np.nan,
                a_fix_tot[i] if i < len(a_fix_tot) else np.nan,
                a_fix_bnd[i] if i < len(a_fix_bnd) else np.nan,
                r_fix_tot[i] if i < len(r_fix_tot) else np.nan,
                r_fix_bnd[i] if i < len(r_fix_bnd) else np.nan,
                a_fix_rms[i] if i < len(a_fix_rms) else np.nan,
                r_fix_rms[i] if i < len(r_fix_rms) else np.nan,
            ]
            w.writerow(row)
    print(
        f"Histories CSV saved to {path} | lens: a_err={len(a_err)}, a_res={len(a_res)}, a_fix={len(a_fix_tot)}, "
        f"r_err={len(r_err)}, r_res={len(r_res)}, r_fix={len(r_fix_tot)}"
    )
    return path


@ensure_figure_closed
def plot_ablation_error_shaded(
    run_roots: List[str], save_path: str, title: str = "Error vs Iteration (mean ± std)"
):
    """Aggregate error histories from multiple run roots and plot mean±std shading.

    Expects each root to contain reports/histories.csv as produced by write_histories_csv.
    """
    # Collect series
    A_list, R_list = [], []
    for root in run_roots:
        csv_path = os.path.join(root, "reports", "histories.csv")
        if not os.path.exists(csv_path):
            continue
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            a_vals, r_vals = [], []
            for row in reader:
                a = row.get("adaptive_total_error", "")
                r = row.get("random_total_error", "")
                try:
                    a_vals.append(float(a)) if a != "" else a_vals.append(np.nan)
                except Exception:
                    a_vals.append(np.nan)
                try:
                    r_vals.append(float(r)) if r != "" else r_vals.append(np.nan)
                except Exception:
                    r_vals.append(np.nan)
        if a_vals:
            A_list.append(np.array(a_vals, dtype=float))
        if r_vals:
            R_list.append(np.array(r_vals, dtype=float))

    def pad_stack(series):
        if not series:
            return np.empty((0, 0))
        L = max(len(s) for s in series)
        out = []
        for s in series:
            if len(s) < L:
                s = np.concatenate([s, np.full(L - len(s), np.nan)])
            out.append(s)
        return np.vstack(out)

    A = pad_stack(A_list)
    R = pad_stack(R_list)

    if A.size == 0 and R.size == 0:
        print("No histories found; skipping ablation plot")
        return

    plt.figure(figsize=(7, 5))
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Total Error")
    plt.yscale("log")

    def plot_band(M, color, label):
        if M.size == 0:
            return
        mean = np.nanmean(M, axis=0)
        std = np.nanstd(M, axis=0)
        x = np.arange(len(mean))
        plt.plot(x, mean, color=color, label=label, linewidth=2)
        plt.fill_between(
            x, np.maximum(mean - std, 1e-32), mean + std, color=color, alpha=0.2
        )

    plot_band(A, "tab:blue", "Adaptive (mean ± std)")
    plot_band(R, "tab:orange", "Random (mean ± std)")
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
    """Aggregate fixed residual histories from multiple run roots and plot mean±std shading."""
    import csv
    import numpy as np

    A_list, R_list = [], []
    for root in run_roots:
        csv_path = os.path.join(root, "reports", "histories.csv")
        if not os.path.exists(csv_path):
            continue
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            a_vals, r_vals = [], []
            for row in reader:
                a = row.get("adaptive_fixed_total_residual", "")
                r = row.get("random_fixed_total_residual", "")
                try:
                    a_vals.append(float(a)) if a != "" else a_vals.append(np.nan)
                except Exception:
                    a_vals.append(np.nan)
                try:
                    r_vals.append(float(r)) if r != "" else r_vals.append(np.nan)
                except Exception:
                    r_vals.append(np.nan)
        if a_vals:
            A_list.append(np.array(a_vals, dtype=float))
        if r_vals:
            R_list.append(np.array(r_vals, dtype=float))

    def pad_stack(series):
        if not series:
            return np.empty((0, 0))
        L = max(len(s) for s in series)
        out = []
        for s in series:
            if len(s) < L:
                s = np.concatenate([s, np.full(L - len(s), np.nan)])
            out.append(s)
        return np.vstack(out)

    A = pad_stack(A_list)
    R = pad_stack(R_list)

    if A.size == 0 and R.size == 0:
        print(
            "No fixed residual histories found; skipping ablation fixed residual plot"
        )
        return

    plt.figure(figsize=(7, 5))
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("∫Ω r(x)^2 dΩ")
    plt.yscale("log")

    def plot_band(M, color, label):
        if M.size == 0:
            return
        mean = np.nanmean(M, axis=0)
        std = np.nanstd(M, axis=0)
        x = np.arange(len(mean))
        plt.plot(x, mean, color=color, label=label, linewidth=2)
        plt.fill_between(
            x, np.maximum(mean - std, 1e-32), mean + std, color=color, alpha=0.2
        )

    plot_band(A, "tab:blue", "Adaptive (mean ± std)")
    plot_band(R, "tab:orange", "Random (mean ± std)")
    plt.legend()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Ablation fixed residual plot saved to {save_path}")


@ensure_figure_closed
def plot_fixed_residual_comparison(
    adaptive_model,
    random_model,
    save_path: str | None = None,
    title: str = "Fixed reference-mesh RMS residual",
):
    """Plot fixed-grid RMS residual histories for both methods on a semilog y-axis."""
    a = list(getattr(adaptive_model, "fixed_rms_residual_history", []) or [])
    r = list(getattr(random_model, "fixed_rms_residual_history", []) or [])
    if not a and not r:
        print("No fixed residual histories; skipping plot")
        return

    plt.figure(figsize=(7, 5))
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("RMS residual (sqrt(mean(r^2)))")
    plt.yscale("log")
    if a:
        plt.semilogy(
            range(len(a)), a, "b-o", label="Adaptive", linewidth=2, markersize=6
        )
    if r:
        plt.semilogy(
            range(len(r)), r, "r--s", label="Random", linewidth=2, markersize=6
        )
    plt.grid(True, alpha=0.3)
    plt.legend()
    if save_path is None:
        save_path = os.path.join(images_dir(), "fixed_residual_comparison.png")
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Fixed residual comparison saved to {save_path}")


@ensure_figure_closed
def plot_fixed_error_rms_comparison(
    adaptive_model,
    random_model,
    save_path: str | None = None,
    title: str = "Fixed reference-mesh RMS error",
):
    """Plot fixed-grid RMS error histories for both methods on a semilog y-axis."""
    a = list(getattr(adaptive_model, "total_error_rms_history", []) or [])
    r = list(getattr(random_model, "total_error_rms_history", []) or [])
    if not a and not r:
        print("No RMS error histories; skipping plot")
        return

    plt.figure(figsize=(7, 5))
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("RMS error (sqrt(mean((u-u_ref)^2)))")
    plt.yscale("log")
    if a:
        plt.semilogy(
            range(len(a)), a, "b-o", label="Adaptive", linewidth=2, markersize=6
        )
    if r:
        plt.semilogy(
            range(len(r)), r, "r--s", label="Random", linewidth=2, markersize=6
        )
    plt.grid(True, alpha=0.3)
    plt.legend()
    if save_path is None:
        save_path = os.path.join(images_dir(), "fixed_error_rms_comparison.png")
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Fixed RMS error comparison saved to {save_path}")


@ensure_figure_closed
def plot_error_integral_comparison(
    adaptive_model,
    random_model,
    save_path: str | None = None,
    title: str = "Integrated error ∫Ω (u-u_ref)^2 dΩ",
):
    """Plot integrated error histories for both methods on a semilog y-axis."""
    a = list(getattr(adaptive_model, "total_error_history", []) or [])
    r = list(getattr(random_model, "total_error_history", []) or [])
    if not a and not r:
        print("No integrated error histories; skipping plot")
        return

    plt.figure(figsize=(7, 5))
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Integrated error ∫Ω (u-u_ref)^2 dΩ")
    plt.yscale("log")
    if a:
        plt.semilogy(
            range(len(a)), a, "b-o", label="Adaptive", linewidth=2, markersize=6
        )
    if r:
        plt.semilogy(
            range(len(r)), r, "r--s", label="Random", linewidth=2, markersize=6
        )
    plt.grid(True, alpha=0.3)
    plt.legend()
    if save_path is None:
        save_path = os.path.join(images_dir(), "error_integral_comparison.png")
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Integrated error comparison saved to {save_path}")


@ensure_figure_closed
def plot_fixed_residual_integral_comparison(
    adaptive_model,
    random_model,
    save_path: str | None = None,
    title: str = "Fixed reference-mesh ∫Ω r^2 dΩ",
):
    """Plot fixed-grid integral of residual^2 histories for both methods on a semilog y-axis."""
    a = list(getattr(adaptive_model, "fixed_total_residual_history", []) or [])
    r = list(getattr(random_model, "fixed_total_residual_history", []) or [])
    if not a and not r:
        print("No fixed residual integral histories; skipping plot")
        return

    plt.figure(figsize=(7, 5))
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("∫Ω r(x)^2 dΩ")
    plt.yscale("log")
    if a:
        plt.semilogy(
            range(len(a)), a, "b-o", label="Adaptive", linewidth=2, markersize=6
        )
    if r:
        plt.semilogy(
            range(len(r)), r, "r--s", label="Random", linewidth=2, markersize=6
        )
    plt.grid(True, alpha=0.3)
    plt.legend()
    if save_path is None:
        save_path = os.path.join(images_dir(), "fixed_residual_integral_comparison.png")
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Fixed residual integral comparison saved to {save_path}")


@ensure_figure_closed
def plot_points_vs_iteration(
    adaptive_model,
    save_path=None,
    title: str = "Interior points vs iteration (adaptive)",
):
    """Plot the number of interior residual points per iteration for the adaptive method."""
    try:
        hist = list(getattr(adaptive_model, "mesh_point_count_history", []) or [])
        if not hist:
            print("No point count history; skipping points vs iteration plot")
            return
        # Remove duplicates for cleaner visualization
        unique_counts = []
        prev = None
        for c in hist:
            if c != prev:
                unique_counts.append(c)
                prev = c
        plt.figure(figsize=(7, 5))
        plt.plot(
            range(len(unique_counts)), unique_counts, "b-o", linewidth=2, markersize=6
        )
        plt.xlabel("Iteration")
        plt.ylabel("Number of interior residual points")
        plt.title(title)
        plt.grid(True, alpha=0.3)
        if len(unique_counts) > 1 and unique_counts[0] > 0:
            factor = unique_counts[-1] / unique_counts[0]
            plt.text(
                0.02,
                0.98,
                f"Grow: ×{factor:.2f}",
                transform=plt.gca().transAxes,
                va="top",
                bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.7),
            )
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Points vs iteration saved to {save_path}")
    except Exception as e:
        print(f"Error creating points vs iteration plot: {e}")
        import traceback

        traceback.print_exc()
        plt.close()


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
        summary_lines.append(
            f"  Mesh progression: {initial:,} → {final:,} points (×{factor:.2f})"
        )

    if adaptive_model.total_error_history:
        initial_error = adaptive_model.total_error_history[0]
        final_error = adaptive_model.total_error_history[-1]
        reduction = initial_error / final_error
        summary_lines.append(
            f"  Error reduction: {initial_error:.2e} → {final_error:.2e} (×{reduction:.2f})"
        )

    summary_lines.append("")

    # Random model metrics
    summary_lines.append("RANDOM POINTS METHOD:")
    if random_model.total_error_history:
        final_random_error = random_model.total_error_history[-1]
        summary_lines.append(f"  Final error: {final_random_error:.2e}")

    summary_lines.append("")

    # Comparison
    if adaptive_model.total_error_history and random_model.total_error_history:
        adaptive_final = adaptive_model.total_error_history[-1]
        random_final = random_model.total_error_history[-1]
        improvement = ((random_final - adaptive_final) / random_final) * 100
        summary_lines.append("COMPARISON:")
        summary_lines.append(
            f"  Adaptive advantage: {improvement:.1f}% better error rate"
        )

    summary_text = "\n".join(summary_lines)

    if save_path:
        with open(save_path, "w") as f:
            f.write(summary_text)
        print(f"Performance summary saved to {save_path}")
    else:
        print(summary_text)

    return summary_text


def create_point_usage_table(
    adaptive_model, random_model, dataset_size, save_path=None
):
    """
    Create a text table of per-iteration labeled and interior point counts for both methods.

    Columns:
      Iteration | LabeledDatasetSize | LabeledBatchSize | AdaptiveInterior | RandomInterior | Match

    Args:
        adaptive_model: Trained adaptive PINN model
        random_model: Trained random PINN model
        dataset_size: Size of the shared labeled dataset (FEM vertices)
        save_path: Output path for the table (defaults to reports/point_usage_table.txt)
    """
    from config import MODEL_CONFIG

    if save_path is None:
        os.makedirs(reports_dir(), exist_ok=True)
        save_path = os.path.join(reports_dir(), "point_usage_table.txt")

    # Determine number of iterations from histories (index 0 is initial count)
    adapt_hist = getattr(adaptive_model, "mesh_point_count_history", []) or []
    rand_hist = getattr(random_model, "mesh_point_count_history", []) or []
    # Adaptive often stores duplicates per iteration; use indices 1,3,5,...
    adapt_iters = max(0, (len(adapt_hist) - 1 + 1) // 2)
    rand_iters = max(0, len(rand_hist) - 1)
    n_iters = min(adapt_iters, rand_iters)

    labeled_batch = MODEL_CONFIG.get("num_data", None)

    lines = []
    lines.append("PER-ITERATION POINT USAGE TABLE")
    lines.append("=" * 50)
    lines.append("")
    lines.append(
        f"{'Iter':>4} | {'LabeledDatasetSize':>18} | {'LabeledBatchSize':>16} | {'AdaptiveInterior':>16} | {'RandomInterior':>14} | {'Match':>5}"
    )
    lines.append("-" * 90)

    for i in range(1, n_iters + 1):
        # Map iteration i -> adaptive index 1 + 2*(i-1); random index i
        adapt_idx = 1 + 2 * (i - 1)
        rand_idx = i
        adapt_interior = adapt_hist[adapt_idx]
        rand_interior = rand_hist[rand_idx]
        match = "yes" if adapt_interior == rand_interior else "no"
        lines.append(
            f"{i:>4} | {dataset_size:>18} | {labeled_batch if labeled_batch is not None else 'N/A':>16} | {adapt_interior:>16} | {rand_interior:>14} | {match:>5}"
        )

    lines.append("")
    lines.append("Notes:")
    lines.append(
        "- LabeledDatasetSize is the total FEM data points shared by both methods."
    )
    lines.append(
        "- LabeledBatchSize is the per-step sample size used in loss_data for both methods."
    )
    lines.append(
        "- AdaptiveInterior and RandomInterior are the number of PDE residual points per iteration."
    )
    lines.append(
        "- Adaptive values use indices 1,3,5,... to avoid duplicates stored per iteration."
    )

    with open(save_path, "w") as f:
        f.write("\n".join(lines))
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
        output_dir = images_dir()

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
    adaptive_training_path = os.path.join(
        output_dir, "adaptive_training_convergence.png"
    )
    plot_training_convergence_simple(
        adaptive_model, "Adaptive Mesh", adaptive_training_path
    )

    random_training_path = os.path.join(output_dir, "random_training_convergence.png")
    plot_training_convergence_simple(
        random_model, "Random Points", random_training_path
    )

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
        output_dir = images_dir()

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
