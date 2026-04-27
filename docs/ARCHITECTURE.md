# PINN-Point Architecture Guide

> **Purpose**: Reference document for AI assistants and developers working on this project.

## Project Overview

**PINN-Point** is a research project comparing Physics-Informed Neural Networks (PINNs) trained with:
- **Adaptive residual-guided sampling** — refines a mesh scaffold, then allocates a fixed interior collocation budget inside high-residual elements
- **Baseline and competitive samplers** — including random, low-discrepancy (Halton/Sobol), residual-based (RAD), and a family of power-tempered and persistent adaptive variants

All methods share the same labeled coarse-mesh FEM data, reference solution, initial weights, and a fixed interior collocation budget per iteration, isolating the effect of collocation point selection. Runtime includes method-specific sampling/refinement overhead.

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
│   │   ├── poisson.py       # Poisson on perforated square
│   │   ├── poisson_ring.py  # Poisson on eccentric annulus
│   │   ├── poisson_ring_hard.py  # Poisson on multi-hole annulus (thin corridors)
│   │   ├── advection_diffusion.py        # Advection-diffusion PDE
│   │   ├── allen_cahn_obstacles_2d.py    # Allen-Cahn with 2D obstacle geometry
│   │   └── navier_stokes_channel_obstacle.py  # Navier-Stokes channel with obstacle
│   └── methods/             # Extensible training methods
│       ├── __init__.py      # Method registry, get_method()
│       ├── base.py          # Abstract TrainingMethod class
│       ├── adaptive.py      # Residual-guided interior sampling on a refined scaffold
│       ├── adaptive_persistent.py        # Adaptive with persistent point retention
│       ├── adaptive_power_tempered.py    # Power-tempered residual weighting (main variant)
│       ├── adaptive_entropy_balanced.py  # Entropy-regularised adaptive sampling
│       ├── adaptive_halton_base.py       # Halton-seeded adaptive scaffold
│       ├── hybrid_anchor.py # Residual + anchor-error adaptive refinement
│       ├── random.py        # Uniform random sampling
│       ├── quasi_random.py  # Halton and Sobol low-discrepancy methods
│       ├── rad.py           # Residual-based Adaptive Distribution (RAD)
│       └── sampling.py      # Shared point-sampling helpers
├── scripts/                 # Submission and analysis scripts
│   ├── submit_m3_cpu_xl_poisson_sweep.sh  # 4-flavor Poisson sweep orchestrator
│   └── compile_elsarticle.sh              # LaTeX build helper
├── notebooks/               # Jupyter notebooks for exploration
├── docs/                    # Documentation and write-ups
├── outputs/                 # Generated experiment outputs (DVC-tracked)
├── paper/                   # LaTeX paper source
└── specs/                   # Experiment spec files (used by remote-ops runner)
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

## Registered Problems

| Name | File | Description |
|------|------|-------------|
| `poisson` | `problems/poisson.py` | Poisson on a perforated square with Gaussian sources |
| `poisson_ring` | `problems/poisson_ring.py` | Poisson on an eccentric annulus (asymmetric sources) |
| `poisson_ring_hard` | `problems/poisson_ring_hard.py` | Poisson on a multi-hole annulus with thin corridors |
| `advection_diffusion` | `problems/advection_diffusion.py` | Advection-diffusion with configurable Péclet number |
| `allen_cahn_obstacles_2d` | `problems/allen_cahn_obstacles_2d.py` | Allen-Cahn phase-field with 2D obstacle geometry |
| `navier_stokes_channel_obstacle` | `problems/navier_stokes_channel_obstacle.py` | Steady Navier-Stokes in a channel with a circular obstacle |

All problems accept a `problem_kwargs` dict in specs / CLI to control geometry and source parameters.

## Registered Methods

| Name | Description |
|------|-------------|
| `random` | Uniform random collocation |
| `halton` | Halton low-discrepancy sequence |
| `sobol` | Sobol low-discrepancy sequence |
| `rad` | Residual-based Adaptive Distribution (RAD) with NaN guards |
| `adaptive` | Residual-guided refinement scaffold |
| `adaptive_persistent` | Adaptive with cross-iteration point retention |
| `adaptive_power_tempered` | Power-tempered residual weighting (recommended adaptive baseline) |
| `adaptive_power_tempered_beta25` | Power-tempered with β=2.5 |
| `adaptive_power_tempered_beta30` | Power-tempered with β=3.0 |
| `adaptive_power_tempered_floor15` | Power-tempered with coverage floor=0.15 |
| `adaptive_power_tempered_floor25` | Power-tempered with coverage floor=0.25 |
| `adaptive_entropy_balanced` | Entropy-regularised adaptive sampling |
| `adaptive_halton_base` | Halton-seeded adaptive scaffold |
| `adaptive_hybrid_anchor` | Residual + anchor-error hybrid refinement |

## Key Configuration (train/config.py)

| Config Dict | Purpose |
|-------------|--------|
| `MODEL_CONFIG` | Network architecture, loss weights |
| `TRAINING_CONFIG` | Epochs, learning rate, iterations |
| `MESH_CONFIG` | Initial mesh size, refinement threshold |
| `HYBRID_ADAPTIVE_CONFIG` | Hybrid anchor count, blend weights, normalization quantile, hybrid refinement threshold |
| `GEOMETRY_CONFIG` | Domain size, shape parameters |
| `VIZ_CONFIG` | Image sizes, colormaps |

Most of these can be overridden per-run via CLI flags (`--mesh-size`, `--iterations`, `--epochs`) or via `problem_kwargs` in the spec JSON.

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

## Submitting a Sweep

The primary sweep workflow uses SSH + tmux on the remote CPU node. The helper scripts live in `../remote-ops/pinn-point/` (private repo, not committed here).

**Typical flow:**
```bash
# 1. Bootstrap the remote once (installs .venv-netgen, clones repo)
bash ../remote-ops/pinn-point/bootstrap_remote.sh

# 2. Submit a 4-flavor Poisson sweep (10 seeds each, 400 epochs)
bash scripts/submit_m3_cpu_xl_poisson_sweep.sh \
  --mesh-size 0.35 \
  --iterations 12 \
  --epochs 400

# 3. Monitor (on remote via SSH)
ssh m3-cpu-xl 'tmux ls'

# 4. Sync results back locally (runs rsync per flavor/seed)
bash ../remote-ops/pinn-point/sync.sh --session <session>
# or use the sweep script's built-in sync by re-running with --sync-only
```

**Required remote env var** (NGSolve shared libs):
```bash
export LD_LIBRARY_PATH=.venv-netgen/lib:${LD_LIBRARY_PATH:-}
```
This is set automatically by `bootstrap_remote.sh` and the sweep scripts.

**Sync root convention**: results land under
`outputs/m3-cpu-xl-poisson-sweep-<epochs>e[-ms<mesh_size>]/<flavor>/<run_id>/`

## Current PDEs

Six PDE problems are registered and production-tested:

| Problem | Geometry | Difficulty driver |
|---------|----------|------------------|
| `poisson` | Perforated square | Multiple cutouts |
| `poisson_ring` | Eccentric annulus | Asymmetric sources + curved boundary |
| `poisson_ring_hard` | Multi-hole annulus | Thin corridors between 3 holes |
| `advection_diffusion` | Rectangle | High Péclet number |
| `allen_cahn_obstacles_2d` | 2D domain with obstacles | Sharp interface + nonlinearity |
| `navier_stokes_channel_obstacle` | Channel with circular obstacle | Coupled velocity-pressure, inf-sup |

The `poisson_ring` / `poisson_ring_hard` benchmark family has the most experimental results (10-seed sweeps at mesh_size 0.7 and 0.35, 4 source/geometry flavors).

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
