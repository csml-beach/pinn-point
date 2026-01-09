# Parameter and Hyperparameter Study

This document explains how to run parameter sweeps in this project to study how different settings affect performance of adaptive vs random PINN training.

## What it does
- Runs multiple experiments with different configuration values.
- Creates an isolated per-run folder at `outputs/<run-id>/` with:
  - `images/` plots (if enabled)
  - `reports/histories.csv` for postprocessing
  - `reports/run_config.json` with configs, system, git metadata, and seed
- You can aggregate multiple runs into a shaded mean±std error plot.

## Reproducibility
- If `TRAINING_CONFIG["seed"]` is set in `train/config.py`, all runs use that fixed seed.
- If not set, each run gets a different random seed; it is used, written into the run ID, and saved in `reports/run_config.json`.

---

## Hyperparameter study (dotted-key grid)
Use `hparams` to vary any combination across config dictionaries using dotted keys. Examples:
- `MODEL_CONFIG.hidden_size`
- `TRAINING_CONFIG.lr`
- `TRAINING_CONFIG.epochs`
- `TRAINING_CONFIG.iterations`
- `MESH_CONFIG.maxh`
- `MODEL_CONFIG.w_interior`, `MODEL_CONFIG.w_data`, `MODEL_CONFIG.w_bc`

Run with the default small grid:
```bash
python3 train/main.py hparams
```
Enable plot/image export for each run:
```bash
python3 train/main.py hparams --images
```

Provide a custom grid inline (JSON):
```bash
python3 train/main.py hparams --images '{"MODEL_CONFIG.hidden_size":[32,64],"TRAINING_CONFIG.lr":[0.001,0.0003],"MESH_CONFIG.maxh":[0.5,0.7]}'
```

Or from a JSON file:
```bash
python3 train/main.py hparams my_grid.json
```
Example `my_grid.json`:
```json
{
  "MODEL_CONFIG.hidden_size": [32, 64],
  "TRAINING_CONFIG.lr": [0.001, 0.0003],
  "TRAINING_CONFIG.epochs": [500],
  "TRAINING_CONFIG.iterations": [5],
  "MESH_CONFIG.maxh": [0.5, 0.7],
  "MODEL_CONFIG.w_interior": [1.0, 2.0]
}
```

Notes:
- Each combination generates a run ID like `YYYY-mm-dd_HH-MM-SS_hps-hs64-lr0.001-m0.5-seed1234`.
- Configs are restored after each combo; the grid does not permanently change `train/config.py`.

---

## Postprocessing: ablation plot
Aggregate multiple runs into a shaded mean±std error plot (requires each run to have `reports/histories.csv`).

Using the 7 most recent hparams runs:
```bash
python3 train/main.py ablate-plot $(ls -1dt outputs/*hps*/ | head -n 7 | xargs -n1 basename)
```
All runs under `outputs/`:
```bash
python3 train/main.py ablate-plot $(find outputs -mindepth 1 -maxdepth 1 -type d -exec basename {} \;)
```
Filter by pattern:
```bash
python3 train/main.py ablate-plot $(find outputs -mindepth 1 -maxdepth 1 -type d -name '*adapt-vs-rand*' -exec basename {} \;)
```

The plot is saved to `outputs/ablation_summary/ablation_error_shaded.png`.

---

## Tips
- To limit total runs, keep the grid small (Cartesian product grows quickly).
- Use `--images` only when you need figures; it saves time to skip during large sweeps.
- If PyVista/NGSolve are not installed, 3D exports are skipped with warnings; core training still runs.
