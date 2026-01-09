"""
Geometry creation and mesh handling functions.
Contains functions for creating complex geometries and managing mesh operations.
"""

import numpy as np
import torch
from netgen.occ import *
from ngsolve import *
from config import MESH_CONFIG, GEOMETRY_CONFIG
from paths import reports_dir


def create_base_shape(length, width):
    """Create a cross-shaped base geometry.

    Args:
        length: Length parameter for the cross
        width: Width parameter for the cross

    Returns:
        Cross-shaped geometry object
    """
    wp1 = WorkPlane().RectangleC(length, width).Face()
    wp2 = WorkPlane().RectangleC(width, length).Face()
    cross = wp1 + wp2
    return cross


def create_complex_geometry():
    """Create the complex geometry with holes and patterns.

    Returns:
        Complex geometry object with named boundaries
    """
    base = create_base_shape(
        length=GEOMETRY_CONFIG["base_l"], width=GEOMETRY_CONFIG["base_w"]
    )
    offset = GEOMETRY_CONFIG["offset"]
    figure = base

    for i in range(3):
        for j in range(3):
            if (i + j) % 2 == 0:
                figure += create_base_shape(i, j).Move((i * offset, j * offset, 0))
            else:
                if i + j == 1:
                    figure += (
                        WorkPlane().Circle(1).Face().Move((i * offset, j * offset, 0))
                    )
                else:
                    figure += base.Mirror(Axis((0, 0, 0), (0, 1, 0))).Move(
                        (i * offset, j * offset, 0)
                    )

    # Create the main geometry by subtracting the figure from a rectangle
    domain_size = GEOMETRY_CONFIG["domain_size"]
    geo = WorkPlane().Rectangle(domain_size, domain_size).Face() - figure.Move(
        (-2, -2, 0)
    )

    # Name the boundaries
    geo.edges.Min(Y).name = "bottom"
    geo.edges.Max(Y).name = "top"
    geo.edges.Min(X).name = "left"
    geo.edges.Max(X).name = "right"

    return geo


def create_initial_mesh(maxh=None):
    """Create the initial mesh from geometry.

    Args:
        maxh: Maximum mesh size. If None, uses config default.

    Returns:
        NGSolve mesh object
    """
    if maxh is None:
        maxh = MESH_CONFIG["maxh"]

    geo = create_complex_geometry()
    plate = OCCGeometry(geo, dim=2)
    mesh = Mesh(plate.GenerateMesh(maxh=maxh)).Curve(3)
    return mesh


def get_initial_mesh_data(mesh):
    """Extract initial mesh data for tracking.

    Args:
        mesh: NGSolve mesh object

    Returns:
        tuple: (mesh_point_count, vertex_coordinates_list)
    """
    mesh_point_count = len(mesh.vertices)
    vertex_coordinates = []

    for v in mesh.vertices:
        vertex_coordinates.append(v.point)

    mesh_x, mesh_y = zip(*vertex_coordinates)
    mesh_x = np.array(mesh_x)
    mesh_y = np.array(mesh_y)
    vertex_coordinates_list = [mesh_x, mesh_y]

    return mesh_point_count, vertex_coordinates_list


def export_vertex_coordinates(mesh):
    """Export vertex coordinates from mesh as a tensor.

    Args:
        mesh: NGSolve mesh object

    Returns:
        torch.Tensor: Vertex coordinates as (N, 2) tensor
    """
    vertex_coordinates = []
    for v in mesh.vertices:
        vertex_coordinates.append(v.point)
    vertex_array = torch.tensor(np.array(vertex_coordinates))
    return vertex_array


def get_random_points(mesh, random_point_count=None):
    """Generate random points within the mesh domain.

    Args:
        mesh: NGSolve mesh object
        random_point_count: Number of random points to generate

    Returns:
        tuple: (rand_x, rand_y) arrays of random coordinates
    """
    if random_point_count is None:
        from config import RANDOM_CONFIG

        random_point_count = RANDOM_CONFIG["default_point_count"]

    from config import RANDOM_CONFIG

    domain_bounds = RANDOM_CONFIG.get("domain_bounds", (0, 5))
    # Prefer tight bounds if configured as "auto"
    if domain_bounds == "auto" or domain_bounds is None:
        verts = np.array([v.point for v in mesh.vertices])
        # verts is (N, 2); compute per-axis mins/maxs and use global min/max
        domain_min = float(np.min(verts))
        domain_max = float(np.max(verts))
    else:
        domain_min, domain_max = domain_bounds

    random_points = []
    attempts = 0
    # Scale attempt budget to improve likelihood of reaching target in complex domains
    max_attempts = max(1000, random_point_count * 20)

    while len(random_points) < random_point_count and attempts < max_attempts:
        # Generate random (x,y) coordinates in the domain
        x = np.random.uniform(domain_min, domain_max)
        y = np.random.uniform(domain_min, domain_max)

        # Check if the generated point (x,y) is in the domain
        try:
            if mesh(x, y).nr != -1:
                random_points.append((x, y))
        except Exception:
            # Point is outside domain, skip
            pass
        attempts += 1

    # If oversampled, truncate to requested size
    if len(random_points) > random_point_count:
        random_points = random_points[:random_point_count]

    if len(random_points) == 0:
        raise ValueError("Could not generate any valid random points in the domain")

    # Optional diagnostics
    try:
        if RANDOM_CONFIG.get("log_sampling_stats", False):
            import os

            rdir = reports_dir()
            os.makedirs(rdir, exist_ok=True)
            acc = len(random_points)
            rate = acc / max(attempts, 1)
            with open(os.path.join(rdir, "point_sampling_stats.txt"), "a") as f:
                f.write(
                    f"requested={random_point_count}, accepted={acc}, attempts={attempts}, acceptance_rate={rate:.4f}, bounds=[{domain_min:.3f},{domain_max:.3f}]\n"
                )
    except Exception:
        pass

    if len(random_points) < random_point_count:
        print(
            f"Warning: Requested {random_point_count} random points but generated {len(random_points)} after {attempts} attempts."
        )

    rand_points = np.array(random_points)
    rand_x, rand_y = rand_points.T

    return rand_x, rand_y
