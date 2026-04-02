# PINN Adaptive Mesh Experiments

This project compares Physics-Informed Neural Networks (PINNs) trained with:
- **Adaptive mesh refinement** — selects new interior points where residuals are high
- **Baseline and competitive methods** — including random, low-discrepancy, and adaptive distributions

The code runs controlled experiments, saves per-run artifacts to `outputs/<run-id>/`, and supports ablation studies across multiple runs and seeds.

## Available Methods

You can benchmark and compare the following methods:
- `adaptive` — Residual-based adaptive mesh refinement
- `adaptive_hybrid_anchor` — Residual + fixed-anchor supervised-error mesh refinement
- `random` — Uniform random point sampling (baseline)
- `halton` — Halton low-discrepancy sequence sampling
- `sobol` — Sobol low-discrepancy sequence sampling
- `random_r` — Uniform random with periodic resampling (Random-R)
- `rad` — Residual-based Adaptive Distribution (Wu et al. 2022)

Select methods via CLI/config:
```bash
python3 train/main.py main  # Default: adaptive + random
python3 train/main.py main --methods adaptive adaptive_hybrid_anchor random halton sobol rad
```
Or set `methods_to_run` in your config or experiment script.

## Device Selection

The runtime device can be selected explicitly:

```bash
# Auto-select (default): CUDA if available, otherwise CPU
python3 train/main.py main

# Force CUDA
python3 train/main.py --device cuda:0 main

# Force CPU
python3 train/main.py --device cpu main

# Smoke tests default to CPU
scripts/smoke_test.sh --seed 123
```

You can also use the environment variable:

```bash
PINN_DEVICE=cuda:0 python3 train/main.py main
PINN_DEVICE=cpu scripts/smoke_test.sh
```

## Quick Start

```bash
# Run a single experiment
python3 train/main.py main

# Smoke test (CPU by default)
scripts/smoke_test.sh

# Hyperparameter study
python3 train/main.py hparams my_grid.json --images
```

## Requirements

- Python 3.10+ (3.11 recommended)
- PyTorch, NumPy, Matplotlib
- NGSolve/Netgen (FEM), PyVista (mesh viz)

See `notebooks/environment.yml` for a Conda setup.

## Project Structure

```
pinn-point/
├── train/           # Source code
│   ├── main.py      # CLI entrypoint
│   ├── config.py    # All configuration
│   └── ...
├── configs/         # Hyperparameter grid JSON files
├── docs/            # Detailed documentation
├── outputs/         # Per-run results (gitignored)
└── notebooks/       # Jupyter notebooks
```

## Configuration

Edit `train/config.py`:

| Config | Key settings |
|--------|--------------|
| `MODEL_CONFIG` | `hidden_size`, `w_data`, `w_interior`, `w_bc` |
| `TRAINING_CONFIG` | `epochs`, `iterations`, `lr`, `seed` |
| `MESH_CONFIG` | `maxh`, `refinement_threshold` |
| `HYBRID_ADAPTIVE_CONFIG` | `anchor_count`, `alpha`, `beta`, `normalization_quantile` |

## Documentation

- [Architecture Overview](docs/ARCHITECTURE.md) — code structure and design
- [Parameter Studies](docs/parameter_study.md) — hyperparameter sweeps and ablation plots
- [Output Format](docs/histories_csv.md) — CSV columns and metrics
- [Hybrid Adaptive Plan](docs/hybrid_adaptive_plan.md) — anchor-based hybrid refinement design
- [Repo TODO](docs/todo.md) — active follow-ups and completed milestones

## Outputs

Each run creates `outputs/<timestamp>_<tag>/`:
- `images/` — plots and per-method convergence figures
- `reports/all_methods_histories.csv` — canonical per-method metrics
- `reports/performance_summary.txt` — summarized end-of-run metrics
- `reports/point_usage_table.txt` — per-iteration collocation budgets
- `reports/run_config.json` — full config, system info, git state, seed

## Reproducibility

- Set `TRAINING_CONFIG["seed"]` for deterministic runs
- Or leave unset — a random seed is generated, saved, and included in the run ID

## Citation

If this work helps your research, please cite appropriately. See `paper/` for details.
