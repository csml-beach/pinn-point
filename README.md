# PINN Adaptive Mesh Experiments

This project compares Physics-Informed Neural Networks (PINNs) trained with:
- Adaptive mesh refinement (selects new interior points where residuals are high), versus
- Random interior point sampling (baseline).

The code runs controlled experiments, saves per-run artifacts (plots, logs, metadata) into `outputs/<run-id>/`, and supports ablation studies across multiple runs and seeds.

## Key ideas
- Fairness: both methods share the same labeled FEM data; only interior points differ (adaptive vs random).
- Reproducibility: each run writes a JSON metadata file with configs, system info, and git state; the RNG seed is also recorded.
- Postprocessing: each run emits a `reports/histories.csv`; you can aggregate many runs into a shaded mean±std error plot.

## Repo structure (high-level)
- `train/`
  - `main.py` — CLI entrypoint for single runs, parameter study demo, cleanup, and ablation plotting
  - `experiments.py` — experiment orchestration
  - `visualization.py` — publication-ready plots and GIFs
  - `pinn_model.py`, `training.py`, `mesh_refinement.py`, `geometry.py` — training and sampling logic
  - `config.py` — all configs (model, training, mesh, random, viz)
  - `paths.py` — per-run outputs manager and metadata writer
- `outputs/<run-id>/` — per-run folder created automatically
  - `images/` — plots, GIFs
  - `reports/` — text summaries, histories CSV, run metadata (JSON)
  - `checkpoints/`, `artifacts/` — optional

## Requirements
- Python 3.10+ (3.11 recommended)
- PyTorch
- NumPy, Matplotlib
- PyVista (for mesh visualizations)
- NGSolve/Netgen (for FEM export)

If you use Conda, see `notebooks/environment.yml` as a starting point.

## Configuration
Edit `train/config.py` to control experiments:
- MODEL_CONFIG
  - `hidden_size`, `num_data`, `num_bd`, `w_data`, `w_interior`, `w_bc`
- TRAINING_CONFIG
  - `epochs`, `iterations`, `lr`, `optimizer`
  - Optional: `seed` — fixed RNG seed for fully deterministic runs. If omitted, a random seed is generated per run and saved.
- MESH_CONFIG
  - `maxh` (initial mesh size), `refinement_threshold`, `reference_mesh_factor`
- RANDOM_CONFIG
  - `default_point_count`, `domain_bounds` ("auto" or bounds), `log_sampling_stats`
- VIZ_CONFIG
  - `image_size`, `gif_duration`, `gif_loop`, `residual_clim`, `error_clim` (fixed colorbar ranges)

## CLI usage
Run all commands from the repo root.

- Single run (default experiment)
```bash
python3 train/main.py main
```

- Quick test (small/fast)
```bash
python3 train/main.py test
```

- Parameter study example (small demo grid)
```bash
python3 train/main.py study
```

- Clean up PNGs used for GIFs
```bash
python3 train/main.py cleanup
```

- Full cleanup of temporary files
```bash
python3 train/main.py cleanup-all
```

- Create a shaded ablation error plot from specific run IDs
```bash
python3 train/main.py ablate-plot <run-id-1> <run-id-2> ...
```

## Examples and tips
- Run the main experiment 7 times (ablation set):
```bash
for i in {1..7}; do python3 train/main.py main; done
```

- Build an ablation plot from the 7 most recent runs:
```bash
python3 train/main.py ablate-plot $(ls -1dt outputs/*/ | head -n 7 | xargs -n1 basename)
```

- Use all runs under `outputs/`:
```bash
python3 train/main.py ablate-plot $(find outputs -mindepth 1 -maxdepth 1 -type d -exec basename {} \;)
```

- Filter runs by pattern (e.g., only adapt-vs-rand):
```bash
python3 train/main.py ablate-plot $(find outputs -mindepth 1 -maxdepth 1 -type d -name '*adapt-vs-rand*' -exec basename {} \;)
```

## Outputs per run
Created at `outputs/<timestamp>_<tag>/` where the tag includes the seed, e.g., `adapt-vs-rand-seed12345678`.
- `images/`
  - `method_comparison.png`
  - `adaptive_training_convergence.png`, `random_training_convergence.png`
  - Optional GIFs: `adaptive_residual_evolution.gif`, `adaptive_error_evolution.gif`, `random_residual_evolution.gif`, `random_error_evolution.gif`
- `reports/`
  - `performance_summary.txt`
  - `histories.csv` (for ablation aggregation)
  - `run_config.json` (configs + system + git + extras like seed and phase)

## Reproducibility and seeds
- If `TRAINING_CONFIG["seed"]` is set, that fixed seed is used and saved.
- If not set, a random per-run seed is generated, used, included in the run ID, and saved in `reports/run_config.json`.


## Citation
If this work helps your research, please cite appropriately. A short synopsis lives in `paper/pinn_adaptive_mesh_synopsis.tex`.
