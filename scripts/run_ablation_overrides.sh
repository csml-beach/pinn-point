#!/usr/bin/env bash
set -euo pipefail

eval "$(pyenv init -)"
pyenv activate torch

# Override knobs
EPOCHS=800
REF_MESH_FACTOR=0.005
MESH_SIZE=0.7
NUM_ADAPTATIONS=10

python - <<PY
import sys
sys.path.append('train')
from experiments import run_complete_experiment
from paths import generate_run_id, set_active_run, write_run_metadata
from utils import set_global_seed
from config import TRAINING_CONFIG, MESH_CONFIG

METHODS = ['adaptive', 'random', 'halton', 'sobol', 'random_r', 'rad']
SEEDS = [42, 123, 456, 789, 1011, 2022, 3033, 4044, 5055, 6066]

TRAINING_CONFIG['epochs'] = ${EPOCHS}
MESH_CONFIG['reference_mesh_factor'] = ${REF_MESH_FACTOR}

for seed in SEEDS:
    set_global_seed(seed)
    run_id = generate_run_id(
        f"ablation-all-methods-epochs{${EPOCHS}}-ref{${REF_MESH_FACTOR}}-seed{seed}"
    )
    set_active_run(run_id)
    write_run_metadata({
        "study": "ablation",
        "seed": seed,
        "methods": METHODS,
        "epochs": ${EPOCHS},
        "reference_mesh_factor": ${REF_MESH_FACTOR},
        "mesh_size": ${MESH_SIZE},
        "num_adaptations": ${NUM_ADAPTATIONS},
    })

    run_complete_experiment(
        mesh_size=${MESH_SIZE},
        num_adaptations=${NUM_ADAPTATIONS},
        epochs=${EPOCHS},
        export_images=True,
        create_gifs=True,
        generate_report=True,
        methods_to_run=METHODS,
    )
PY
