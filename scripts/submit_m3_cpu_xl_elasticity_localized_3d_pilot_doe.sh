#!/usr/bin/env bash
set -euo pipefail

# DOE sweep for harder localized 3D elasticity on m3-cpu-xl.
# Axis: geometry/physics flavor (not sampler tuning knobs).

usage() {
  cat <<'EOF'
Usage:
  submit_m3_cpu_xl_elasticity_localized_3d_pilot_doe.sh [--flavors CSV]
                                                         [--methods CSV]
                                                         [--seeds CSV]
                                                         [--epochs N]
                                                         [--iterations N]
                                                         [--lr X]
                                                         [--mesh-size F]
                                                         [--ref-factor F]
                                                         [--parallel N]
                                                         [--threads N]
                                                         [--commit SHA]
                                                         [--config PATH]
                                                         [--sync-root PATH]
                                                         [--skip-setup]

Options:
  --flavors CSV      Flavor names to run
                     (default: localized_base,localized_narrow,localized_narrow_patch,localized_narrow_patch_strong)
  --methods CSV      Comma-separated method list
                     (default: adaptive_power_tempered,random,halton,rad)
  --seeds CSV        Comma-separated seeds (default: 42,123,456,789,1011)
  --epochs N         Training epochs per seed (default: 400)
  --iterations N     Iterations (default: 10)
  --lr X             Learning rate (default: 5e-4)
  --mesh-size F      Initial mesh size (default: 0.35)
  --ref-factor F     Reference mesh factor (default: 0.15)
  --parallel N       Parallel seeds per flavor (default: 5)
  --threads N        OMP/MKL threads per job (default: 1)
  --commit SHA       Remote commit to pin
  --config PATH      m3-cpu-xl config env file
  --sync-root PATH   Local output root
                     (default: outputs/m3-cpu-xl-elasticity-localized-3d-pilot-doe-<epochs>e-<iterations>i)
  --skip-setup       Skip remote bootstrap/sync after first launch
EOF
}

flavors_csv="localized_base,localized_narrow,localized_narrow_patch,localized_narrow_patch_strong"
methods_csv="adaptive_power_tempered,random,halton,rad"
seeds_csv="42,123,456,789,1011"
epochs=400
iterations=10
learning_rate="5e-4"
mesh_size="0.35"
reference_mesh_factor="0.15"
parallel_jobs=5
threads_per_job=1
commit_sha="$(git rev-parse origin/main)"
config_file=""
sync_root=""
skip_setup=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --flavors) flavors_csv="$2"; shift 2 ;;
    --methods) methods_csv="$2"; shift 2 ;;
    --seeds) seeds_csv="$2"; shift 2 ;;
    --epochs) epochs="$2"; shift 2 ;;
    --iterations) iterations="$2"; shift 2 ;;
    --lr) learning_rate="$2"; shift 2 ;;
    --mesh-size) mesh_size="$2"; shift 2 ;;
    --ref-factor) reference_mesh_factor="$2"; shift 2 ;;
    --parallel) parallel_jobs="$2"; shift 2 ;;
    --threads) threads_per_job="$2"; shift 2 ;;
    --commit) commit_sha="$2"; shift 2 ;;
    --config) config_file="$2"; shift 2 ;;
    --sync-root) sync_root="$2"; shift 2 ;;
    --skip-setup) skip_setup=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

repo_root="$(git rev-parse --show-toplevel)"
remote_ops_dir="$repo_root/../remote-ops/pinn-point"
if [[ -z "$config_file" ]]; then
  config_file="$remote_ops_dir/config.m3-cpu-xl.env"
fi
if [[ -z "$sync_root" ]]; then
  sync_root="$repo_root/outputs/m3-cpu-xl-elasticity-localized-3d-pilot-doe-${epochs}e-${iterations}i"
fi
if [[ ! -f "$config_file" ]]; then
  echo "Config file not found: $config_file" >&2
  exit 1
fi

single_script="$repo_root/scripts/submit_m3_large_cpu_elasticity_3d_screen_confirm.sh"
if [[ ! -x "$single_script" ]]; then
  echo "Base submit script not executable: $single_script" >&2
  exit 1
fi

# Physics/geometry flavors for elasticity_localized_cantilever_3d.
flavor_kwargs_localized_base='{}'
flavor_kwargs_localized_narrow='{"cylinder_radius":0.33}'
flavor_kwargs_localized_narrow_patch='{"cylinder_radius":0.33,"traction_patch_sigma":0.10}'
flavor_kwargs_localized_narrow_patch_strong='{"cylinder_radius":0.33,"traction_patch_sigma":0.08,"traction_magnitude":0.008}'

IFS=',' read -r -a flavors <<< "$flavors_csv"

first_launch=true
for flavor in "${flavors[@]}"; do
  kwargs_var="flavor_kwargs_${flavor}"
  if [[ -z "${!kwargs_var+x}" ]]; then
    echo "Unknown flavor: $flavor" >&2
    exit 1
  fi
  problem_kwargs="${!kwargs_var}"

  setup_flag=""
  if [[ "$skip_setup" == true ]] || [[ "$first_launch" != true ]]; then
    setup_flag="--skip-setup"
  fi
  first_launch=false

  session_prefix="cpu-el3d-localized-doe-${flavor//_/-}"
  run_root="$sync_root/${flavor}"

  echo ""
  echo "══════════════════════════════════════════════"
  echo " flavor=$flavor"
  echo " session_prefix=$session_prefix"
  echo " sync_root=$run_root"
  echo "══════════════════════════════════════════════"

  "$single_script" \
    --problem elasticity_localized_cantilever_3d \
    --problem-kwargs "$problem_kwargs" \
    --methods "$methods_csv" \
    --seeds "$seeds_csv" \
    --epochs "$epochs" \
    --iterations "$iterations" \
    --lr "$learning_rate" \
    --mesh-size "$mesh_size" \
    --parallel "$parallel_jobs" \
    --threads "$threads_per_job" \
    --reference-mesh-factor "$reference_mesh_factor" \
    --commit "$commit_sha" \
    --config "$config_file" \
    --sync-root "$run_root" \
    --session-prefix "$session_prefix" \
    ${setup_flag}
done

echo ""
echo "[doe] completed localized 3D elasticity pilot sweep under: $sync_root"
