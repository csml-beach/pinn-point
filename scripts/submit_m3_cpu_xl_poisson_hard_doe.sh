#!/usr/bin/env bash
set -euo pipefail

# DOE sweep for harder Poisson experiments on m3-cpu-xl.
# Axes:
#   1) geometry/source flavor on poisson_ring_hard
#   2) reference mesh factor tier
#
# Goal: find settings where adaptive methods gain clearer residual/error leverage.

usage() {
  cat <<'EOF'
Usage:
  submit_m3_cpu_xl_poisson_hard_doe.sh [--flavors CSV] [--ref-factors CSV]
                                        [--methods CSV] [--seeds CSV]
                                        [--epochs N] [--iterations N]
                                        [--mesh-size F] [--parallel N]
                                        [--threads N] [--commit SHA]
                                        [--config PATH] [--sync-root PATH]
                                        [--skip-setup]

Options:
  --flavors CSV      Flavor names to run (default: hard_base,hard_narrow,hard_obstacles,hard_obstacles_sharp)
  --ref-factors CSV  Reference mesh factors (default: 0.05,0.035,0.025)
  --methods CSV      Comma-separated method list
  --seeds CSV        Comma-separated seeds
  --epochs N         Training epochs per seed (default: 400)
  --iterations N     Iterations (default: 8)
  --mesh-size F      Initial mesh size (default: 0.35)
  --parallel N       Parallel seeds per flavor/ref tier (default: 5)
  --threads N        OMP/MKL threads per job (default: 1)
  --commit SHA       Remote commit to pin
  --config PATH      m3-cpu-xl config env file
  --sync-root PATH   Local output root (default: outputs/m3-cpu-xl-poisson-hard-doe-<epochs>e)
  --skip-setup       Skip remote bootstrap/sync after first launch
EOF
}

flavors_csv="hard_base,hard_narrow,hard_obstacles,hard_obstacles_sharp"
ref_factors_csv="0.05,0.035,0.025"
methods_csv="adaptive_power_tempered,adaptive_halton_base,random,halton,rad"
seeds_csv="42,123,456,789,1011"
epochs=400
iterations=8
mesh_size="0.35"
parallel_jobs=5
threads_per_job=1
commit_sha="$(git rev-parse origin/main)"
config_file=""
sync_root=""
skip_setup=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --flavors) flavors_csv="$2"; shift 2 ;;
    --ref-factors) ref_factors_csv="$2"; shift 2 ;;
    --methods) methods_csv="$2"; shift 2 ;;
    --seeds) seeds_csv="$2"; shift 2 ;;
    --epochs) epochs="$2"; shift 2 ;;
    --iterations) iterations="$2"; shift 2 ;;
    --mesh-size) mesh_size="$2"; shift 2 ;;
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
  sync_root="$repo_root/outputs/m3-cpu-xl-poisson-hard-doe-${epochs}e"
fi
if [[ ! -f "$config_file" ]]; then
  echo "Config file not found: $config_file" >&2
  exit 1
fi

single_script="$repo_root/scripts/submit_m3_large_cpu_poisson_ring_screen_confirm.sh"

# Flavors for poisson_ring_hard.
flavor_kwargs_hard_base='{}'
flavor_kwargs_hard_narrow='{"main_hole_radius":1.02,"top_hole_radius":0.36,"bottom_hole_radius":0.31,"main_hole_center_x":0.60,"main_hole_center_y":-0.12}'
flavor_kwargs_hard_obstacles='{"main_hole_radius":1.00,"top_hole_radius":0.34,"bottom_hole_radius":0.30,"left_mid_hole_radius":0.20,"left_mid_hole_center_x":-1.28,"left_mid_hole_center_y":0.24,"right_mid_hole_radius":0.18,"right_mid_hole_center_x":1.14,"right_mid_hole_center_y":0.80}'
flavor_kwargs_hard_obstacles_sharp='{"main_hole_radius":1.00,"top_hole_radius":0.34,"bottom_hole_radius":0.30,"left_mid_hole_radius":0.20,"left_mid_hole_center_x":-1.28,"left_mid_hole_center_y":0.24,"right_mid_hole_radius":0.18,"right_mid_hole_center_x":1.14,"right_mid_hole_center_y":0.80,"source_sigma_scale":0.60,"source_amplitude_scale":1.10}'

IFS=',' read -r -a flavors <<< "$flavors_csv"
IFS=',' read -r -a ref_factors <<< "$ref_factors_csv"

first_launch=true
for flavor in "${flavors[@]}"; do
  kwargs_var="flavor_kwargs_${flavor}"
  if [[ -z "${!kwargs_var+x}" ]]; then
    echo "Unknown flavor: $flavor" >&2
    exit 1
  fi
  problem_kwargs="${!kwargs_var}"

  for ref_factor in "${ref_factors[@]}"; do
    setup_flag=""
    if [[ "$skip_setup" == true ]] || [[ "$first_launch" != true ]]; then
      setup_flag="--skip-setup"
    fi
    first_launch=false

    session_prefix="cpu-poisson-hard-doe-${flavor//_/-}-ref${ref_factor//./p}"
    run_root="$sync_root/${flavor}/ref_${ref_factor//./p}"

    echo ""
    echo "══════════════════════════════════════════════"
    echo " flavor=$flavor ref_factor=$ref_factor"
    echo " session_prefix=$session_prefix"
    echo " sync_root=$run_root"
    echo "══════════════════════════════════════════════"

    "$single_script" \
      --problem poisson_ring_hard \
      --problem-kwargs "$problem_kwargs" \
      --methods "$methods_csv" \
      --seeds "$seeds_csv" \
      --epochs "$epochs" \
      --iterations "$iterations" \
      --mesh-size "$mesh_size" \
      --parallel "$parallel_jobs" \
      --threads "$threads_per_job" \
      --reference-mesh-factor "$ref_factor" \
      --commit "$commit_sha" \
      --config "$config_file" \
      --sync-root "$run_root" \
      --session-prefix "$session_prefix" \
      ${setup_flag}
  done
done

echo ""
echo "[doe] completed poisson hard DOE sweep under: $sync_root"
