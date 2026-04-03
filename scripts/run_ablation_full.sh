#!/usr/bin/env bash
set -euo pipefail

eval "$(pyenv init -)"
pyenv activate torch

EXPORT_IMAGES=0
CREATE_GIFS=0

usage() {
  cat <<'EOF'
Usage: run_ablation_full.sh [--viz] [--gifs]

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

EXPORT_IMAGES="$EXPORT_IMAGES" CREATE_GIFS="$CREATE_GIFS" python - <<'PY'
import os
import sys
sys.path.append('train')
from experiments import run_complete_experiment
from paths import generate_run_id, set_active_run, write_run_metadata
from utils import set_global_seed

METHODS = ['adaptive', 'adaptive_hybrid_anchor', 'random', 'halton', 'sobol', 'random_r', 'rad']
SEEDS = [42, 123, 456, 789, 1011, 2022, 3033, 4044, 5055, 6066]
EXPORT_IMAGES = os.environ.get("EXPORT_IMAGES", "0") == "1"
CREATE_GIFS = os.environ.get("CREATE_GIFS", "0") == "1"

for seed in SEEDS:
    set_global_seed(seed)
    run_id = generate_run_id(f"ablation-all-methods-seed{seed}")
    set_active_run(run_id)
    write_run_metadata({"study": "ablation", "seed": seed, "methods": METHODS})

    run_complete_experiment(
        mesh_size=0.5,
        num_adaptations=10,
        epochs=500,
        export_images=EXPORT_IMAGES,
        create_gifs=CREATE_GIFS,
        generate_report=True,
        methods_to_run=METHODS,
    )
PY
