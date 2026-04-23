#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  submit_m3_large_cpu_approved_suite.sh [--problems CSV] [--seeds CSV] [--commit SHA] [--config PATH]

Description:
  Orchestrate the approved benchmark suite on m3-large-cpu using the current
  paper-facing configs for:
    - Allen-Cahn obstacles
    - advection-diffusion
    - Navier-Stokes channel obstacle

  The suite runs one benchmark family at a time on the shared CPU host. Each
  family uses its own internal seed-level parallelism.
EOF
}

problems_csv="allen_cahn,advection_diffusion,navier_stokes"
seeds_csv="42,123,456,789,1011,2022,3033,4044,5055,6066,7077,8088,9099,11111,12121,13131,14141,15151,16161,17171"
commit_sha="$(git rev-parse origin/codex/navier-stokes-channel-obstacle)"
config_file=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --problems)
      problems_csv="$2"
      shift 2
      ;;
    --seeds)
      seeds_csv="$2"
      shift 2
      ;;
    --commit)
      commit_sha="$2"
      shift 2
      ;;
    --config)
      config_file="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

repo_root="$(git rev-parse --show-toplevel)"
if [[ -z "$config_file" ]]; then
  config_file="$repo_root/../remote-ops/pinn-point/config.m3-large-cpu.env"
fi

IFS=',' read -r -a problems <<< "$problems_csv"

run_allen_cahn() {
  "$repo_root/scripts/submit_m3_large_cpu_allen_cahn_obstacles_screen_confirm.sh" \
    --commit "$commit_sha" \
    --config "$config_file" \
    --seeds "$seeds_csv" \
    --epochs 400 \
    --iterations 6 \
    --parallel 10 \
    --threads 1 \
    --mesh-size 0.18 \
    --reference-mesh-factor 0.05 \
    --sync-root "$repo_root/outputs/m3-large-cpu-allen-cahn-obstacles-screen-confirm-400e"
}

run_advection_diffusion() {
  "$repo_root/scripts/submit_m3_large_cpu_advection_screen_confirm.sh" \
    --commit "$commit_sha" \
    --config "$config_file" \
    --seeds "$seeds_csv" \
    --epochs 300 \
    --iterations 8 \
    --parallel 10 \
    --threads 1 \
    --reference-mesh-factor 0.05 \
    --sync-root "$repo_root/outputs/m3-large-cpu-advection-diffusion-screen-confirm-300e"
}

run_navier_stokes() {
  "$repo_root/scripts/submit_m3_large_cpu_ns_screen_confirm.sh" \
    --commit "$commit_sha" \
    --config "$config_file" \
    --seeds "$seeds_csv" \
    --epochs 200 \
    --iterations 6 \
    --parallel 10 \
    --threads 1 \
    --reference-mesh-factor 0.035 \
    --problem-kwargs '{"t_end":1.0,"dt":0.001,"supervised_time_slices":11,"reference_time_slices":21}' \
    --sync-root "$repo_root/outputs/m3-large-cpu-navier-stokes-screen-confirm-tend1p0-ref0035-dt0001-20seed"
}

for problem in "${problems[@]}"; do
  case "$problem" in
    allen_cahn)
      echo "[suite] launching Allen-Cahn"
      run_allen_cahn
      ;;
    advection_diffusion)
      echo "[suite] launching advection-diffusion"
      run_advection_diffusion
      ;;
    navier_stokes)
      echo "[suite] launching Navier-Stokes"
      run_navier_stokes
      ;;
    *)
      echo "Unknown suite problem: $problem" >&2
      exit 1
      ;;
  esac
done

echo "[suite] all requested benchmark families completed"
