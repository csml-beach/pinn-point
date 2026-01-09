"""
FEM Solver Module - Handles finite element mesh generation and solving.

NOTE: This module contains the default Poisson problem FEM solver.
For new PDEs, use the extensible problems module:
    from problems import get_problem
    problem = get_problem("poisson")  # or your custom problem
    gfu, fes = problem.solve_fem(mesh)
"""

import torch
import numpy as np
from torch.utils.data import TensorDataset
from netgen.geom2d import unit_square
from ngsolve import *
from geometry import export_vertex_coordinates


def solve_FEM(mesh):
    """Solve the Poisson equation on the given mesh.

    Solves: -∇²u = f(x,y) with f(x,y) = x*y
    BC: Dirichlet u=0 on bottom boundary

    NOTE: This is the default Poisson implementation. For new PDEs:
        from problems import get_problem
        problem = get_problem("your_problem")
        gfu, fes = problem.solve_fem(mesh)

    Args:
        mesh: NGSolve mesh object

    Returns:
        tuple: (gfu, fes) - GridFunction solution and finite element space
    """
    # H1-conforming finite element space
    fes = H1(mesh, order=1, dirichlet="bottom", autoupdate=True)

    # Define trial- and test-functions
    u = fes.TrialFunction()
    v = fes.TestFunction()

    # The bilinear-form
    a = BilinearForm(grad(u) * grad(v) * dx)

    # Right-hand side function: f(x,y) = x*y
    funcf = 1 * x * y
    # Alternative: funcf = 50*sin(y)
    f = LinearForm(funcf * v * dx)

    # Assemble system
    a.Assemble()
    f.Assemble()

    # Solve system
    gfu = GridFunction(fes)
    gfu.vec.data = a.mat.Inverse(freedofs=fes.FreeDofs()) * f.vec

    return gfu, fes


def export_fem_solution(mesh, gfu):
    """Export FEM solution values at mesh vertices.

    Args:
        mesh: NGSolve mesh object
        gfu: GridFunction with FEM solution

    Returns:
        torch.Tensor: Solution values at mesh vertices
    """
    vertex_array = export_vertex_coordinates(mesh).to(torch.float32)
    mesh_x, mesh_y = vertex_array.T
    solution_array = torch.tensor([x for x in gfu.vec])
    return solution_array


def create_dataset(vertex_array, solution_array):
    """Create a PyTorch dataset from mesh data.

    Args:
        vertex_array: Tensor of vertex coordinates
        solution_array: Tensor of solution values

    Returns:
        TensorDataset: PyTorch dataset containing the data
    """
    dataset = TensorDataset(vertex_array, solution_array.reshape(-1, 1))
    return dataset
