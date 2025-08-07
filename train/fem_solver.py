"""
FEM Solver Module - Handles finite element mesh generation and solving.
"""

import torch
import numpy as np
from torch.utils.data import TensorDataset
from netgen.geom2d import unit_square
from ngsolve import *


def export_vertex_coordinates(mesh):
    """Export vertex coordinates from mesh as a tensor."""
    vertex_coordinates = []
    for v in mesh.vertices:
        vertex_coordinates.append(v.point)
    vertex_array = torch.tensor(np.array(vertex_coordinates))
    return vertex_array


def solve_FEM(mesh):
    """Solve the finite element problem on the given mesh.
    
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
