"""
Problems module - Defines PDE problems for PINN training.

This module provides an extensible framework for defining different PDEs
that can be solved using Physics-Informed Neural Networks.

Available problems:
    - PoissonProblem: Poisson equation with localized source bumps
    - PoissonRingProblem: eccentric annulus Poisson benchmark
    - PoissonRingHardProblem: harder Poisson ring control benchmark
    - AllenCahnObstacles2DProblem: transient Allen-Cahn with interior obstacles
    - AdvectionDiffusionProblem: steady advection-diffusion-reaction equation
    - NavierStokesChannelObstacleProblem: geometry-first prototype for a future
      transient Navier-Stokes benchmark
    
To add a new problem:
    1. Create a new file in this directory (e.g., heat.py)
    2. Subclass PDEProblem from base.py
    3. Implement required methods: pde_residual, boundary_loss, source_term, solve_fem
    4. Register in this __init__.py

Example:
    from problems import (
        PoissonProblem,
        AllenCahnObstacles2DProblem,
        AdvectionDiffusionProblem,
        NavierStokesChannelObstacleProblem,
    )
    
    problem = PoissonProblem()
    residual = problem.pde_residual(model, x, y)
    bc_loss = problem.boundary_loss(model)
"""

from .base import PDEProblem
from .advection_diffusion import AdvectionDiffusionProblem
from .allen_cahn_obstacles_2d import AllenCahnObstacles2DProblem
from .navier_stokes_channel_obstacle import NavierStokesChannelObstacleProblem
from .poisson import PoissonProblem
from .poisson_ring import PoissonRingProblem
from .poisson_ring_hard import PoissonRingHardProblem

# Registry of available problems for CLI/config selection
PROBLEM_REGISTRY = {
    "poisson": PoissonProblem,
    "allen_cahn_obstacles_2d": AllenCahnObstacles2DProblem,
    "poisson_ring": PoissonRingProblem,
    "poisson_ring_hard": PoissonRingHardProblem,
    "advection_diffusion": AdvectionDiffusionProblem,
    "navier_stokes_channel_obstacle": NavierStokesChannelObstacleProblem,
}


def get_problem(name: str, **kwargs) -> PDEProblem:
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
    return PROBLEM_REGISTRY[name](**kwargs)


def list_problems() -> list:
    """List available problem names."""
    return list(PROBLEM_REGISTRY.keys())
