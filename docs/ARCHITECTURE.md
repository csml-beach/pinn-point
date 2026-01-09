# PINN-Point Architecture Guide

> **Purpose**: Reference document for AI assistants and developers working on this project.

## Project Overview

**PINN-Point** is a research project comparing Physics-Informed Neural Networks (PINNs) trained with:
- **Adaptive mesh refinement** — concentrates collocation points in high-residual regions
- **Random sampling** — uniform random point selection (baseline)

The key innovation is a **fair comparison**: both methods share the same labeled FEM data, isolating the effect of collocation point selection.

## Directory Structure

```
pinn-point/
├── train/                    # Core Python modules
│   ├── main.py              # CLI entrypoint (main, test, hparams, cleanup, ablate-plot)
│   ├── experiments.py       # Experiment orchestration
│   ├── pinn_model.py        # FeedForward neural network class
│   ├── training.py          # Model training, optimizer setup
│   ├── mesh_refinement.py   # Adaptive refinement logic
│   ├── visualization.py     # Plots, GIFs, CSV export
│   ├── geometry.py          # NGSolve geometry, mesh utilities
│   ├── fem_solver.py        # FEM solver, dataset creation
│   ├── config.py            # All configuration dictionaries
│   ├── paths.py             # Per-run output path management
│   ├── utils.py             # System info, checkpoints, cleanup
│   ├── problems/            # Extensible PDE definitions
│   │   ├── __init__.py      # Problem registry, get_problem()
│   │   ├── base.py          # Abstract PDEProblem class
│   │   └── poisson.py       # Poisson equation implementation
│   └── methods/             # Extensible training methods
│       ├── __init__.py      # Method registry, get_method()
│       ├── base.py          # Abstract TrainingMethod class
│       ├── adaptive.py      # Residual-based adaptive refinement
│       └── random.py        # Uniform random sampling
├── notebooks/               # Jupyter notebooks for exploration
├── docs/                    # Documentation (CSV format, parameter study)
├── outputs/                 # Generated experiment outputs (gitignored)
├── backup/                  # Old experiment outputs (gitignored)
└── paper/                   # LaTeX paper files
```

## Running the Project

**Environment**: pyenv virtualenv named `netgen`
```bash
~/.pyenv/versions/netgen/bin/python train/main.py <command>
```

**CLI Commands**:
- `main` — Run full adaptive vs random comparison
- `test` — Quick test run
- `hparams` — Hyperparameter study
- `ablate-plot` — Generate ablation summary plots
- `cleanup` — Remove generated artifacts

## Key Configuration (train/config.py)

| Config Dict | Purpose |
|-------------|---------|
| `MODEL_CONFIG` | Network architecture, loss weights |
| `TRAINING_CONFIG` | Epochs, learning rate, iterations |
| `MESH_CONFIG` | Initial mesh size, refinement threshold |
| `GEOMETRY_CONFIG` | Domain size, shape parameters |
| `VIZ_CONFIG` | Image sizes, colormaps |

## Adding a New PDE Problem

1. Create `train/problems/your_pde.py`:

```python
from .base import PDEProblem
from ngsolve import *

class YourPDEProblem(PDEProblem):
    name = "your_pde"
    description = "Description of your PDE"
    
    def pde_residual(self, model, x, y):
        """Compute PDE residual (should be zero when satisfied)."""
        # Use self.compute_derivative(u, var, order) for derivatives
        pass
    
    def boundary_loss(self, model, num_boundary_points=100):
        """Compute boundary condition loss."""
        pass
    
    def source_term(self, x, y):
        """Evaluate source term f(x,y)."""
        pass
    
    def solve_fem(self, mesh):
        """Solve using NGSolve FEM. Returns (gfu, fes)."""
        pass
    
    def get_domain_bounds(self):
        """Return (x_min, x_max, y_min, y_max)."""
        pass
```

2. Register in `train/problems/__init__.py`:
```python
from .your_pde import YourPDEProblem
PROBLEM_REGISTRY["your_pde"] = YourPDEProblem
```

3. Use it:
```python
from problems import get_problem
problem = get_problem("your_pde")
residual = problem.pde_residual(model, x, y)
```

## Adding a Competitive Method

1. Create `train/methods/your_method.py`:

```python
from .base import TrainingMethod

class YourMethod(TrainingMethod):
    name = "your_method"
    description = "Description of your method"
    
    def get_collocation_points(self, mesh, model=None, iteration=0):
        """Return (x, y) tensors of collocation points."""
        pass
    
    def refine_mesh(self, mesh, model, iteration=0):
        """Return (new_mesh, was_refined) tuple."""
        pass
```

2. Register in `train/methods/__init__.py`:
```python
from .your_method import YourMethod
METHOD_REGISTRY["your_method"] = YourMethod
```

## Current PDE: Poisson Equation

**Equation**: $-\nabla^2 u = f(x,y)$ where $f(x,y) = xy$

**Domain**: Complex geometry with holes (L-shaped region with patterns), size 5×5

**Boundary Conditions**: Dirichlet $u = 0$ on bottom boundary

**Implementation locations** (legacy, will be refactored):
- `train/pinn_model.py`: `PDE_residual()`, `loss_boundary_condition()`
- `train/fem_solver.py`: `solve_FEM()`
- `train/problems/poisson.py`: New modular implementation

## Data Flow

```
1. create_initial_mesh()     → NGSolve mesh
2. solve_FEM(mesh)           → Reference FEM solution (gfu, fes)
3. create_dataset()          → PyTorch dataset of (coords, solution)
4. FeedForward(mesh_x, mesh_y) → PINN model
5. train_model()             → Train on dataset + PDE loss
6. refine_mesh() or get_random_points() → New collocation points
7. Repeat 5-6 for N iterations
8. Export visualizations, CSVs, GIFs
```

## Important Functions

| Function | Location | Purpose |
|----------|----------|---------|
| `export_vertex_coordinates(mesh)` | geometry.py | Extract mesh vertices as tensor |
| `ngsolve_to_pyvista()` | visualization.py | Convert mesh for 3D viz (uses temp dir) |
| `run_adaptive_training_fair()` | experiments.py | Main adaptive experiment |
| `run_random_training_fair()` | experiments.py | Main random experiment |
| `compute_model_error()` | mesh_refinement.py | Error vs reference solution |

## Experiment Outputs

Each run creates `outputs/<timestamp>_<name>/`:
```
├── images/           # PNG plots, GIF animations
├── reports/          # histories.csv, metadata.json
├── checkpoints/      # Model weights (optional)
└── artifacts/        # Additional outputs
```

## Files to Never Commit

Already in `.gitignore`:
- `outputs/`, `backup/` — Experiment results
- `images/`, `train/images/` — Generated plots
- `*.vtu`, `*.vtk`, `vtk_export*` — VTK mesh files
- `reports/` — Generated logs

## Known Issues / Tech Debt

1. **PDE coupling**: `pinn_model.py` and `fem_solver.py` have hardcoded Poisson equation. Use `problems/` module for new PDEs.

2. **Method dispatch**: `experiments.py` uses string-based if/else. Consider using `methods/` registry for cleaner extension.

3. **Geometry coupling**: L-shaped domain is hardcoded in `geometry.py`. Future: pass geometry as parameter to problem class.

## Testing Changes

```bash
cd /Users/arash/Documents/GitHub/pinn-point/train
~/.pyenv/versions/netgen/bin/python -c "
from problems import get_problem, list_problems
from methods import get_method, list_methods
print('Problems:', list_problems())
print('Methods:', list_methods())
"
```

## Dependencies

See `requirements.txt`. Key packages:
- `torch` — Neural network
- `ngsolve`, `netgen` — FEM and mesh generation
- `pyvista` — 3D visualization
- `matplotlib`, `PIL` — 2D plots, GIF creation
