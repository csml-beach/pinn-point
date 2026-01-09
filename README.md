# PINN Adaptive Mesh Experiments

This project compares Physics-Informed Neural Networks (PINNs) trained with:
- **Adaptive mesh refinement** — selects new interior points where residuals are high
- **Random interior point sampling** — baseline comparison

The code runs controlled experiments, saves per-run artifacts to `outputs/<run-id>/`, and supports ablation studies across multiple runs and seeds.

## Quick Start

```bash
# Run a single experiment
python3 train/main.py main

# Quick test (small/fast)
python3 train/main.py test

# Hyperparameter study
python3 train/main.py hparams configs/param_study_1.json --images

# Show all commands
python3 train/main.py help
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

## Documentation

- [Architecture Overview](docs/ARCHITECTURE.md) — code structure and design
- [Parameter Studies](docs/parameter_study.md) — hyperparameter sweeps and ablation plots
- [Output Format](docs/histories_csv.md) — CSV columns and metrics

## Outputs

Each run creates `outputs/<timestamp>_<tag>/`:
- `images/` — plots, GIFs
- `reports/histories.csv` — metrics for aggregation
- `reports/run_config.json` — full config, system info, git state, seed

## Reproducibility

- Set `TRAINING_CONFIG["seed"]` for deterministic runs
- Or leave unset — a random seed is generated, saved, and included in the run ID

## Citation

If this work helps your research, please cite appropriately. See `paper/` for details.
