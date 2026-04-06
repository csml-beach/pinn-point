# PINN Adaptive Mesh Experiments

This project compares Physics-Informed Neural Networks (PINNs) trained with:
- **Adaptive residual-guided sampling** — uses a refined mesh as a scaffold, then samples a fixed interior budget where residuals are high
- **Baseline and competitive methods** — including random, low-discrepancy, and adaptive distributions

The code runs controlled experiments, saves per-run artifacts to `outputs/<run-id>/`, and supports ablation studies across multiple runs and seeds.
The current default config is a lean iteration profile intended for faster method development, while heavier paper-facing runs can still be launched explicitly.

## Available Methods

You can benchmark and compare the following methods:
- `adaptive` — Residual-guided fixed-budget interior sampling on a refined mesh scaffold
- `adaptive_hybrid_anchor` — Residual + fixed-anchor supervised-error mesh refinement
- `random` — Uniform random point sampling (baseline)
- `halton` — Halton low-discrepancy sequence sampling
- `sobol` — Sobol low-discrepancy sequence sampling
- `rad` — Residual-based Adaptive Distribution (Wu et al. 2022)

Select methods via CLI/config:
```bash
python3 train/main.py main  # Main experiment entrypoint
python3 train/main.py dev   # Lean development profile
python3 train/main.py screen  # Stronger screening profile
```
Method selection is currently controlled by CLI mode or by experiment scripts/specs:
- `dev` defaults to `adaptive`, `random`, `halton`
- `screen` defaults to `adaptive`, `random`, `halton`, `rad`
- `main` currently runs `adaptive` and `random`

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
# Lean development run (recommended inner loop)
python3 train/main.py dev

# Stronger screening run
python3 train/main.py screen

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
| `MODEL_CONFIG` | `hidden_size`, `num_data`, `w_data`, `w_interior`, `w_bc` |
| `TRAINING_CONFIG` | `epochs`, `iterations`, `optimizer`, `lr`, `seed` |
| `MESH_CONFIG` | `maxh`, `refinement_threshold` |
| `GEOMETRY_CONFIG` | `domain_size`, `grid_n`, `cell_fill`, `circle_radius` |
| `HYBRID_ADAPTIVE_CONFIG` | `anchor_count`, `alpha`, `beta`, `normalization_quantile`, `refinement_threshold` |
| `RAD_CONFIG` | `k`, `c`, `num_candidates`, `resample_period` |

## Documentation

- [Architecture Overview](docs/ARCHITECTURE.md) — code structure and design
- [Lean Baseline Protocol](docs/lean_baseline_protocol.md) — recommended fast-iteration experiment profile
- [Parameter Studies](docs/parameter_study.md) — hyperparameter sweeps and ablation plots
- [Output Format](docs/histories_csv.md) — CSV columns and metrics
- [Hybrid Adaptive Plan](docs/hybrid_adaptive_plan.md) — anchor-based hybrid refinement design
- [Repo TODO](docs/todo.md) — active follow-ups and completed milestones

## Outputs

Each run creates `outputs/<timestamp>_<tag>/`:
- `images/comparison/` — cross-method comparison plots
- `images/methods/<method>/` — method-local images such as training convergence and field snapshots
- `reports/all_methods_histories.csv` — canonical per-method metrics, including raw and normalized error and fixed-reference residual fields
- `reports/performance_summary.txt` — summarized end-of-run metrics with relative L2/RMS error reporting
- `reports/point_usage_table.txt` — per-iteration collocation budgets
- `reports/run_config.json` — full config, system info, git state, seed
- `reports/run_manifest.json` — index of run directories and artifact locations
- `reports/methods/<method>/` — method-local `history.csv`, `diagnostics.json`, `iteration_diagnostics.csv`, and `sampling_stats.txt`

## Reproducibility

- Set `TRAINING_CONFIG["seed"]` for deterministic runs
- Or leave unset — a random seed is generated, saved, and included in the run ID

## Citation

If this work helps your research, please cite appropriately. See `paper/` for details.
