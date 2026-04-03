#!/usr/bin/env bash
set -euo pipefail

eval "$(pyenv init -)"
pyenv activate torch

EXPORT_IMAGES=0
CREATE_GIFS=0

usage() {
  cat <<'EOF'
Usage: run_ablation_solutions_all.sh [--viz] [--gifs]

Defaults:
  --viz  off (export_images=false)
  --gifs off (create_gifs=false)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --viz) EXPORT_IMAGES=1; shift ;;
    --gifs) EXPORT_IMAGES=1; CREATE_GIFS=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

run_solution() {
  local label=$1
  local epochs=$2
  local ref_mesh=$3
  local mesh_size=$4
  local num_adapt=$5
  local w_bc=$6
  local hidden_size=$7

  EPOCHS=$epochs \
  REF_MESH_FACTOR=$ref_mesh \
  MESH_SIZE=$mesh_size \
  NUM_ADAPTATIONS=$num_adapt \
  W_BC=$w_bc \
  HIDDEN_SIZE=$hidden_size \
  EXPORT_IMAGES=$EXPORT_IMAGES \
  CREATE_GIFS=$CREATE_GIFS \
  LABEL=$label \
  python - <<'PY'
import os
import sys
sys.path.append('train')
from experiments import run_complete_experiment
from paths import generate_run_id, set_active_run, write_run_metadata
from utils import set_global_seed
from config import TRAINING_CONFIG, MESH_CONFIG, MODEL_CONFIG

METHODS = ['adaptive', 'adaptive_hybrid_anchor', 'random', 'halton', 'sobol', 'random_r', 'rad']
SEEDS = [42, 123, 456, 789, 1011, 2022, 3033, 4044, 5055, 6066]

label = os.environ['LABEL']
epochs = int(os.environ['EPOCHS'])
ref_mesh = float(os.environ['REF_MESH_FACTOR'])
mesh_size = float(os.environ['MESH_SIZE'])
num_adapt = int(os.environ['NUM_ADAPTATIONS'])

w_bc_env = os.environ.get('W_BC', 'None')
hs_env = os.environ.get('HIDDEN_SIZE', 'None')
w_bc = None if w_bc_env == 'None' else float(w_bc_env)
hidden_size = None if hs_env == 'None' else int(hs_env)
export_images = os.environ.get("EXPORT_IMAGES", "0") == "1"
create_gifs = os.environ.get("CREATE_GIFS", "0") == "1"

TRAINING_CONFIG['epochs'] = epochs
MESH_CONFIG['reference_mesh_factor'] = ref_mesh
if w_bc is not None:
    MODEL_CONFIG['w_bc'] = w_bc
if hidden_size is not None:
    MODEL_CONFIG['hidden_size'] = hidden_size

for seed in SEEDS:
    set_global_seed(seed)
    run_id = generate_run_id(
        f"ablation-{label}-epochs{epochs}-ref{ref_mesh}-seed{seed}"
    )
    set_active_run(run_id)
    write_run_metadata({
        "study": "ablation",
        "solution": label,
        "seed": seed,
        "methods": METHODS,
        "epochs": epochs,
        "reference_mesh_factor": ref_mesh,
        "mesh_size": mesh_size,
        "num_adaptations": num_adapt,
        "w_bc": w_bc,
        "hidden_size": hidden_size,
    })

    run_complete_experiment(
        mesh_size=mesh_size,
        num_adaptations=num_adapt,
        epochs=epochs,
        export_images=export_images,
        create_gifs=create_gifs,
        generate_report=True,
        methods_to_run=METHODS,
    )
PY
}

# Combined: higher w_bc + larger hidden_size + more epochs
run_solution "s123-wbc5-hs100-epochs1200" 1200 0.005 0.7 10 5.0 100
