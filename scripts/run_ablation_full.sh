#!/usr/bin/env bash
set -euo pipefail

eval "$(pyenv init -)"
pyenv activate torch

python - <<'PY'
import sys
sys.path.append('train')
from experiments import run_complete_experiment
from paths import generate_run_id, set_active_run, write_run_metadata
from utils import set_global_seed

METHODS = ['adaptive', 'random', 'halton', 'sobol', 'random_r', 'rad']
SEEDS = [42, 123, 456, 789, 1011, 2022, 3033, 4044, 5055, 6066]

for seed in SEEDS:
    set_global_seed(seed)
    run_id = generate_run_id(f"ablation-all-methods-seed{seed}")
    set_active_run(run_id)
    write_run_metadata({"study": "ablation", "seed": seed, "methods": METHODS})

    run_complete_experiment(
        mesh_size=0.5,
        num_adaptations=10,
        epochs=500,
        export_images=True,
        create_gifs=True,
        generate_report=True,
        methods_to_run=METHODS,
    )
PY
