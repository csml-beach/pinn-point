"""
Problems module - Defines PDE problems for PINN training.

This module provides an extensible framework for defining different PDEs
that can be solved using Physics-Informed Neural Networks.

Available problems:
    - PoissonProblem: Poisson equation with source term f(x,y) = x*y
    
To add a new problem:
    1. Create a new file in this directory (e.g., heat.py)
    2. Subclass PDEProblem from base.py
    3. Implement required methods: pde_residual, boundary_loss, source_term, solve_fem
    4. Register in this __init__.py

Example:
    from problems import PoissonProblem
    
    problem = PoissonProblem()
    residual = problem.pde_residual(model, x, y)
    bc_loss = problem.boundary_loss(model)
"""

from .base import PDEProblem
from .poisson import PoissonProblem

# Registry of available problems for CLI/config selection
PROBLEM_REGISTRY = {
    "poisson": PoissonProblem,
}


def get_problem(name: str) -> PDEProblem:
    """Get a problem instance by name.

    Args:
        name: Problem name (e.g., 'poisson')

    Returns:
        PDEProblem instance

    Raises:
        ValueError: If problem name not found
    """
    if name not in PROBLEM_REGISTRY:
        available = ", ".join(PROBLEM_REGISTRY.keys())
        raise ValueError(f"Unknown problem '{name}'. Available: {available}")
    return PROBLEM_REGISTRY[name]()


def list_problems() -> list:
    """List available problem names."""
    return list(PROBLEM_REGISTRY.keys())
