# PINN-Point Architecture Guide

> **Purpose**: Reference document for AI assistants and developers working on this project.

## Project Overview

**PINN-Point** is a research project comparing Physics-Informed Neural Networks (PINNs) trained with:
- **Adaptive mesh refinement** — concentrates collocation points in high-residual regions
- **Baseline and competitive samplers** — including random, low-discrepancy, residual-based fixed-mesh methods, and a hybrid anchor-guided adaptive method

The core experiment idea is a **fair comparison**: all methods share the same labeled FEM data, reference solution, initial weights, and point-budget schedule, isolating the effect of collocation point selection and refinement policy. The active comparison policy also uses the same configured training budget per iteration for every method, disables adaptive-only bonus training, and records runtime with method-specific sampling/refinement overhead included.

## Directory Structure

```
pinn-point/
├── train/                    # Core Python modules
│   ├── main.py              # CLI entrypoint (main, dev, screen, smoke, hparams, cleanup, cleanup-all, ablate-plot)
│   ├── experiments.py       # Experiment orchestration
│   ├── pinn_model.py        # FeedForward neural network class
│   ├── training.py          # Model training, optimizer setup
│   ├── mesh_refinement.py   # Adaptive refinement logic
│   ├── visualization.py     # Plots, summaries, CSV export
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
│       ├── hybrid_anchor.py # Residual + anchor-error adaptive refinement
│       ├── random.py        # Uniform random sampling
│       ├── random_r.py      # Random-R periodic resampling
│       ├── quasi_random.py  # Halton and Sobol low-discrepancy methods
│       ├── rad.py           # Residual-based Adaptive Distribution
│       └── sampling.py      # Shared point-sampling helpers
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
- `main` — Run the default experiment configuration
- `dev` — Run the lean development profile for fast iteration
- `screen` — Run the stronger screening profile before heavier confirmatory runs
- `smoke` — Run a minimal end-to-end smoke test
- `hparams` — Hyperparameter study
- `ablate-plot` — Generate ablation summary plots
- `cleanup` — Remove generated artifacts
- `cleanup-all` — Remove temporary files from `outputs/`

## Key Configuration (train/config.py)

| Config Dict | Purpose |
|-------------|---------|
| `MODEL_CONFIG` | Network architecture, loss weights |
| `TRAINING_CONFIG` | Epochs, learning rate, iterations |
| `MESH_CONFIG` | Initial mesh size, refinement threshold |
| `HYBRID_ADAPTIVE_CONFIG` | Hybrid anchor count, blend weights, normalization quantile, hybrid refinement threshold |
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

**Domain**: Perforated square domain with repeated cross/circle cutout patterns, size 5×5

**Boundary Conditions**: Dirichlet $u = 0$ on bottom boundary

**Active implementation locations**:
- `train/problems/poisson.py`: Poisson residual, boundary loss, FEM solve, and mesh creation
- `train/pinn_model.py`: delegates residual and boundary loss through the active `problem`
- `train/fem_solver.py`: thin helpers that call `problem.solve_fem(...)`

The problem abstraction is now on the active training path, but the Poisson problem is still the only built-in PDE and some backend-sensitive FEM details remain under active cleanup.

## Data Flow

```
1. `problem.make_mesh()`     → NGSolve mesh
2. `problem.solve_fem(mesh)` → Reference FEM solution (gfu, fes)
3. `create_dataset()`        → PyTorch dataset of `(coords, solution)`
4. `FeedForward(...)`        → PINN model with active `problem`
5. `train_model()`           → Train on supervised data + PDE/boundary loss
6. `method.refine_mesh()` or `method.get_collocation_points()` → New collocation set
7. Repeat for `N` iterations with shared evaluation metrics
8. Export canonical reports and comparison plots
```

## Important Functions

| Function | Location | Purpose |
|----------|----------|---------|
| `export_vertex_coordinates(mesh)` | geometry.py | Extract mesh vertices as tensor |
| `create_multi_method_visualizations()` | visualization.py | Generate the canonical report bundle |
| `run_adaptive_training_fair()` | experiments.py | Main adaptive experiment |
| `run_method_training_fair()` | experiments.py | Main non-adaptive method runner |
| `run_complete_experiment()` | experiments.py | Shared multi-method experiment entrypoint |
| `compute_model_error()` | mesh_refinement.py | Error vs reference solution |

## Experiment Outputs

Each run creates `outputs/<timestamp>_<name>/`:
```
├── images/
│   ├── comparison/   # cross-method comparison plots
│   └── methods/      # method-local images grouped by method name
├── reports/
│   ├── methods/      # per-method histories, diagnostics, and sampling stats
│   ├── all_methods_histories.csv
│   ├── run_config.json
│   └── run_manifest.json
├── checkpoints/      # Model weights (optional)
└── artifacts/        # Additional outputs
```

Canonical report artifacts:
- `reports/all_methods_histories.csv`
- `reports/performance_summary.txt`
- `reports/point_usage_table.txt`
- `reports/run_config.json`
- `reports/run_manifest.json`

Per-method report artifacts:
- `reports/methods/<method>/history.csv`
- `reports/methods/<method>/diagnostics.json`
- `reports/methods/<method>/iteration_diagnostics.csv`
- `reports/methods/<method>/sampling_stats.txt`

## Files to Never Commit

Already in `.gitignore`:
- `outputs/`, `backup/` — Experiment results
- `*.vtu`, `*.vtk`, `vtk_export*` — VTK mesh files

## Known Issues / Tech Debt

See [docs/todo.md](/Users/arash/Documents/GitHub/pinn-point/docs/todo.md) for the active follow-up list.

## Testing Changes

```bash
cd /Users/arash/Documents/GitHub/pinn-point
~/.pyenv/versions/netgen/bin/python -m py_compile train/*.py train/problems/*.py train/methods/*.py
scripts/smoke_test.sh --seed 123
```

## Dependencies

See `requirements.txt`. Key packages:
- `torch` — Neural network
- `ngsolve`, `netgen` — FEM and mesh generation
- `pyvista` — 3D visualization
- `matplotlib`, `PIL` — 2D plots, GIF creation
