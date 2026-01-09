# Ablation Study Plan: Sampling Method Comparison

This document outlines the plan for generating comparative ablation plots across different collocation point sampling strategies.

## Background

We have implemented multiple sampling methods from Wu et al. (2022) "A Comprehensive Study of Non-adaptive and Residual-based Adaptive Sampling for PINNs":

### Available Methods (in `train/methods/`)

| Method | File | Description | Key Parameters |
|--------|------|-------------|----------------|
| `adaptive` | `adaptive.py` | Residual-based mesh refinement (our method) | `refinement_threshold` |
| `random` | `random.py` | Uniform random baseline | - |
| `halton` | `quasi_random.py` | Halton low-discrepancy sequence | `seed` |
| `sobol` | `quasi_random.py` | Sobol low-discrepancy sequence | `seed` |
| `random_r` | `random_r.py` | Uniform random with periodic resampling | `resample_period` |
| `rad` | `rad.py` | Residual-based Adaptive Distribution | `k`, `c`, `num_candidates` |

### Configuration (in `train/config.py`)

```python
RAD_CONFIG = {
    "k": 2.0,                # Residual exponent (recommended: 2)
    "c": 0.0,                # Regularization (0 = pure residual weighting)
    "num_candidates": 2000,  # Candidate points for residual evaluation
    "resample_period": 1,    # Resample every iteration
}

RANDOM_R_CONFIG = {
    "resample_period": 1,    # Resample every iteration
}

QUASI_RANDOM_CONFIG = {
    "seed": 42,              # Reproducibility
}
```

## Study Design

### Primary Comparison: Adaptive vs Competing Methods

**Objective**: Demonstrate that our residual-based adaptive mesh refinement outperforms:
1. Non-adaptive methods (random, halton, sobol)
2. Adaptive-but-fixed-mesh methods (random_r, RAD)

**Fair Comparison Constraints**:
- Same initial mesh and training dataset
- Same reference solution for error evaluation
- Match point counts across methods (use adaptive's progression)
- Same number of training epochs per iteration
- Same model architecture and hyperparameters

### Recommended Experiments

#### Experiment 1: Full Method Comparison (10 seeds)

```python
from experiments import run_complete_experiment
from paths import generate_run_id, set_active_run, write_run_metadata
from utils import set_global_seed
import os

METHODS = ['adaptive', 'random', 'halton', 'sobol', 'random_r', 'rad']
SEEDS = [42, 123, 456, 789, 1011, 2022, 3033, 4044, 5055, 6066]

for seed in SEEDS:
    set_global_seed(seed)
    run_id = generate_run_id(f"ablation-all-methods-seed{seed}")
    set_active_run(run_id)
    write_run_metadata({"study": "ablation", "seed": seed, "methods": METHODS})
    
    models = run_complete_experiment(
        mesh_size=0.5,
        num_adaptations=10,
        epochs=500,
        export_images=True,
        create_gifs=True,
        generate_report=True,
        methods_to_run=METHODS
    )
```

#### Experiment 2: RAD Hyperparameter Sensitivity

Test different values of `k` (residual exponent):

```python
# Modify config before running
from config import RAD_CONFIG

for k_val in [0.5, 1.0, 2.0, 3.0, 4.0]:
    RAD_CONFIG["k"] = k_val
    # Run experiment with rad method only
    # Compare final errors
```

Key combinations from Wu et al.:
- `k=0`: Uniform (equivalent to Random-R)
- `k=1`: Nabian et al.'s method
- `k=2, c=0`: Wu et al. recommended default

#### Experiment 3: Resampling Period Study

For `random_r` and `rad`, test resampling every N epochs:

```python
for period in [1, 5, 10, 25, 50]:
    RANDOM_R_CONFIG["resample_period"] = period
    RAD_CONFIG["resample_period"] = period
    # Run experiments
```

## Output Files

Each run produces (in `outputs/<run_id>/`):

| File | Location | Content |
|------|----------|---------|
| `all_methods_histories.csv` | `reports/` | Iteration-wise errors/point counts per method |
| `histories.csv` | `reports/` | Legacy format for adaptive+random |
| `errors_*.png` | `images/` | Error heatmaps per iteration |
| `residuals_*.png` | `images/` | Residual heatmaps per iteration |
| `meta.json` | root | Run metadata, config, seed |

### CSV Format: `all_methods_histories.csv`

```csv
method,iteration,total_error,boundary_error,point_count
adaptive,0,29225.83,59758.84,110
adaptive,1,21964.00,46066.05,125
...
halton,0,32271.45,65517.05,110
...
rad,0,32289.49,65547.91,110
...
```

## Plotting Scripts (To Implement)

### Plot 1: Error Convergence Curves

```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_error_convergence(run_dirs: list, output_path: str):
    """Plot error vs iteration for all methods across multiple seeds.
    
    Args:
        run_dirs: List of output directories to aggregate
        output_path: Where to save the figure
    """
    all_data = []
    for run_dir in run_dirs:
        csv_path = Path(run_dir) / "reports" / "all_methods_histories.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            all_data.append(df)
    
    combined = pd.concat(all_data, ignore_index=True)
    
    # Compute mean and std per method per iteration
    stats = combined.groupby(['method', 'iteration']).agg({
        'total_error': ['mean', 'std'],
        'point_count': 'mean'
    }).reset_index()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = {
        'adaptive': '#1f77b4',
        'random': '#ff7f0e',
        'halton': '#2ca02c',
        'sobol': '#d62728',
        'random_r': '#9467bd',
        'rad': '#8c564b'
    }
    
    for method in stats['method'].unique():
        method_data = stats[stats['method'] == method]
        iters = method_data['iteration']
        mean_err = method_data[('total_error', 'mean')]
        std_err = method_data[('total_error', 'std')]
        
        ax.plot(iters, mean_err, label=method, color=colors.get(method, 'gray'))
        ax.fill_between(iters, mean_err - std_err, mean_err + std_err, 
                        alpha=0.2, color=colors.get(method, 'gray'))
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Total Error (L² norm vs reference)')
    ax.set_title('Sampling Method Comparison: Error Convergence')
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
```

### Plot 2: Final Error Bar Chart

```python
def plot_final_error_comparison(run_dirs: list, output_path: str):
    """Bar chart of final errors with error bars."""
    # Similar aggregation as above
    # Extract last iteration per method
    # Plot grouped bar chart with std as error bars
```

### Plot 3: Error vs Computational Cost

```python
def plot_error_vs_points(run_dirs: list, output_path: str):
    """Plot error vs total collocation points used."""
    # X-axis: cumulative point count
    # Y-axis: total error
    # Shows efficiency of adaptive method
```

## Expected Results (from Wu et al.)

Based on the paper's findings:

1. **RAD (k=2, c=0)** should outperform random sampling significantly
2. **Quasi-random** (Halton/Sobol) should slightly outperform uniform random
3. **Random-R** should match or slightly beat fixed random (due to resampling)
4. **Our adaptive mesh refinement** should provide benefits from both:
   - Residual-focused point placement (like RAD)
   - Increasing resolution where needed (mesh refinement)

## How to Resume This Work

### Quick Start

```bash
cd /Users/arash/Documents/GitHub/pinn-point/train
~/.pyenv/versions/netgen/bin/python -c "
from experiments import run_complete_experiment

models = run_complete_experiment(
    mesh_size=0.5,
    num_adaptations=10,
    epochs=500,
    methods_to_run=['adaptive', 'random', 'halton', 'rad']
)
"
```

### Files to Check

- `train/experiments.py`: Contains `run_complete_experiment()` and `run_method_training_fair()`
- `train/methods/__init__.py`: Method registry with all 6 methods
- `train/config.py`: `RAD_CONFIG`, `RANDOM_R_CONFIG`, `QUASI_RANDOM_CONFIG`
- `docs/sampling_methods_paper.md`: Full details of Wu et al. (2022) paper

### Key Functions

| Function | File | Purpose |
|----------|------|---------|
| `run_complete_experiment()` | `experiments.py` | Main entry point for multi-method comparison |
| `run_method_training_fair()` | `experiments.py` | Generalized training with any registered method |
| `get_method(name, **kwargs)` | `methods/__init__.py` | Factory for method instances |
| `write_multi_method_histories_csv()` | `experiments.py` | Writes aggregated CSV for all methods |

### Method Signatures

```python
# Get a method instance
from methods import get_method, list_methods

method = get_method("rad", k=2.0, c=0.0, num_candidates=2000)
x, y = method.get_collocation_points(mesh, model=model, iteration=5, num_points=200)

# Run full experiment
from experiments import run_complete_experiment

models = run_complete_experiment(
    mesh_size=0.5,           # Initial mesh size
    num_adaptations=10,      # Number of refinement iterations  
    epochs=500,              # Epochs per iteration
    export_images=True,      # Save visualization PNGs
    create_gifs=True,        # Combine PNGs into GIFs
    generate_report=True,    # Write CSV and tables
    methods_to_run=['adaptive', 'random', 'halton', 'sobol', 'random_r', 'rad']
)
```

## Notes

- The RAD method requires the model to compute residuals, so first iteration uses uniform sampling
- All methods use the same domain bounds: `(0.0, 5.0)` matching `GEOMETRY_CONFIG["domain_size"]`
- Point counts are matched to adaptive method's progression for fair comparison
- Reference solution uses 190K+ points for accurate error measurement

---

*Created: 2024-12-19*
*Last Updated: 2024-12-19*
*Reference: Wu et al. (2022) DOI: 10.1016/j.cma.2022.115671*
