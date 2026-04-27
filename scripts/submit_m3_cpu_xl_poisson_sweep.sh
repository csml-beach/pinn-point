#!/usr/bin/env bash
set -euo pipefail

# Orchestrate the 2x2 Poisson geometry/source sweep on m3-cpu-xl.
#
# Flavors (all use problem=poisson_ring):
#   baseline     default geometry and source
#   sharp        tighter source bumps (source_sigma_scale=0.55)
#   narrow       larger inner hole, narrower corridor (inner_radius=1.05)
#   narrow_sharp narrow corridor + tighter source (combined)

usage() {
  cat <<'EOF'
Usage:
  submit_m3_cpu_xl_poisson_sweep.sh [--flavors CSV] [--methods CSV] [--seeds CSV]
                                     [--epochs N] [--iterations N] [--mesh-size F]
                                     [--parallel N]
                                     [--threads N] [--reference-mesh-factor X]
                                     [--commit SHA] [--config PATH] [--sync-root PATH]
                                     [--skip-setup]

Options:
  --flavors CSV      Which flavors to run (default: baseline,sharp,narrow,narrow_sharp)
  --methods CSV      Comma-separated method list
  --seeds CSV        Comma-separated seed list
  --epochs N         Training epochs per seed (default: 400)
  --iterations N     Mesh-refinement iterations (default: 8)
  --mesh-size F      Initial collocation mesh size (default: unset, uses train default)
  --parallel N       Seeds to run in parallel per flavor (default: 10)
  --threads N        OMP/MKL threads per job (default: 1)
  --reference-mesh-factor X  FEM reference mesh factor (default: 0.05)
  --commit SHA       Remote git commit to checkout
  --config PATH      Machine config env file (default: config.m3-cpu-xl.env)
  --sync-root PATH   Local root for synced outputs (default: outputs/m3-cpu-xl-poisson-sweep-<epochs>e)
  --skip-setup       Skip remote bootstrap/sync for all flavors
EOF
}

# ── defaults ──────────────────────────────────────────────────────────────────
flavors_csv="baseline,sharp,narrow,narrow_sharp"
methods_csv="adaptive_power_tempered,adaptive_halton_base,adaptive_persistent,adaptive,random,halton,rad"
seeds_csv="42,123,456,789,1011,2022,3033,4044,5055,6066"
epochs=400
iterations=8
mesh_size=""
parallel_jobs=10
threads_per_job=1
reference_mesh_factor="0.05"
commit_sha="$(git rev-parse origin/codex/navier-stokes-channel-obstacle)"
config_file=""
sync_root=""
skip_setup=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --flavors)            flavors_csv="$2";           shift 2 ;;
    --methods)            methods_csv="$2";           shift 2 ;;
    --seeds)              seeds_csv="$2";             shift 2 ;;
    --epochs)             epochs="$2";                shift 2 ;;
    --iterations)         iterations="$2";            shift 2 ;;
    --mesh-size)          mesh_size="$2";             shift 2 ;;
    --parallel)           parallel_jobs="$2";         shift 2 ;;
    --threads)            threads_per_job="$2";       shift 2 ;;
    --reference-mesh-factor) reference_mesh_factor="$2"; shift 2 ;;
    --commit)             commit_sha="$2";            shift 2 ;;
    --config)             config_file="$2";           shift 2 ;;
    --sync-root)          sync_root="$2";             shift 2 ;;
    --skip-setup)         skip_setup=true;            shift ;;
    -h|--help)            usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

repo_root="$(git rev-parse --show-toplevel)"
remote_ops_dir="$repo_root/../remote-ops/pinn-point"

if [[ -z "$config_file" ]]; then
  config_file="$remote_ops_dir/config.m3-cpu-xl.env"
fi
if [[ -z "$sync_root" ]]; then
  if [[ -n "$mesh_size" ]]; then
    sync_root="$repo_root/outputs/m3-cpu-xl-poisson-sweep-${epochs}e-ms${mesh_size//./p}"
  else
    sync_root="$repo_root/outputs/m3-cpu-xl-poisson-sweep-${epochs}e"
  fi
fi

if [[ ! -f "$config_file" ]]; then
  echo "Config file not found: $config_file" >&2; exit 1
fi

# ── flavor definitions ────────────────────────────────────────────────────────
flavor_kwargs_baseline='{}' 
flavor_kwargs_sharp='{"source_sigma_scale":0.55,"source_amplitude_scale":1.0}'
flavor_kwargs_narrow='{"inner_radius":1.05,"inner_center_x":0.55,"inner_center_y":-0.20}'
flavor_kwargs_narrow_sharp='{"inner_radius":1.05,"inner_center_x":0.55,"inner_center_y":-0.20,"source_sigma_scale":0.55,"source_amplitude_scale":1.0}'

single_script="$repo_root/scripts/submit_m3_large_cpu_poisson_ring_screen_confirm.sh"

# ── run flavors ───────────────────────────────────────────────────────────────
IFS=',' read -r -a flavors <<< "$flavors_csv"

first_flavor=true

for flavor in "${flavors[@]}"; do
  kwargs_var="flavor_kwargs_${flavor}"
  if [[ -z "${!kwargs_var+x}" ]]; then
    echo "Unknown flavor: $flavor (expected: baseline, sharp, narrow, narrow_sharp)" >&2
    exit 1
  fi
  problem_kwargs="${!kwargs_var}"
  session_prefix="cpu-poisson-sweep-${flavor//_/-}"

  flavor_skip_setup=false
  if [[ "$skip_setup" == true ]] || [[ "$first_flavor" != true ]]; then
    flavor_skip_setup=true
  fi
  first_flavor=false

  echo ""
  echo "══════════════════════════════════════════════"
  echo " Flavor: $flavor"
  echo " Session prefix: $session_prefix"
  echo " problem-kwargs: $problem_kwargs"
  echo "══════════════════════════════════════════════"

  skip_flag=""
  [[ "$flavor_skip_setup" == true ]] && skip_flag="--skip-setup"

  mesh_size_flag=""
  [[ -n "$mesh_size" ]] && mesh_size_flag="--mesh-size $mesh_size"

  "$single_script" \
    --problem          poisson_ring \
    --problem-kwargs   "$problem_kwargs" \
    --methods          "$methods_csv" \
    --seeds            "$seeds_csv" \
    --epochs           "$epochs" \
    --iterations       "$iterations" \
    ${mesh_size_flag} \
    --parallel         "$parallel_jobs" \
    --threads          "$threads_per_job" \
    --reference-mesh-factor "$reference_mesh_factor" \
    --commit           "$commit_sha" \
    --config           "$config_file" \
    --sync-root        "$sync_root/$flavor" \
    --session-prefix   "$session_prefix" \
    ${skip_flag}
done

echo ""
echo "[sweep] all flavors completed → $sync_root"
