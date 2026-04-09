#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  submit_jetstream_cpu_advection_confirm.sh [--parallel N] [--threads N] [--commit SHA] [--config PATH]

Description:
  Submit the 10-seed advection-diffusion confirm benchmark to the sibling
  remote-ops runner on the Jetstream CPU host. The remote repo is bootstrapped
  and pinned to the requested commit once before launching parallel jobs, which
  avoids git checkout races during submission.
EOF
}

parallel_jobs=5
threads_per_job=3
commit_sha="$(git rev-parse --short HEAD)"
config_file=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --parallel)
      parallel_jobs="$2"
      shift 2
      ;;
    --threads)
      threads_per_job="$2"
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
remote_ops_dir="$repo_root/../remote-ops/pinn-point"
if [[ -z "$config_file" ]]; then
  config_file="$remote_ops_dir/config.jetstream-medium.env"
fi

if [[ ! -f "$config_file" ]]; then
  echo "Config file not found: $config_file" >&2
  exit 1
fi

if [[ ! -d "$remote_ops_dir" ]]; then
  echo "remote-ops dir not found: $remote_ops_dir" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "$remote_ops_dir/lib.sh"

spec_dir="/tmp/pinn_point_jetstream_advdiff_confirm_specs"
mkdir -p "$spec_dir"

seeds=(42 123 456 789 1011 2022 3033 4044 5055 6066)

echo "[submit] bootstrapping remote repo"
CONFIG_FILE="$config_file" "$remote_ops_dir/bootstrap_remote.sh"

echo "[submit] pinning remote repo to commit $commit_sha"
remote_exec "
  set -euo pipefail
  cd '$REMOTE_REPO_PATH'
  git fetch --all --prune
  git checkout '$commit_sha'
"

submit_seed() {
  local seed=\"$1\"
  local spec=\"$spec_dir/cpu-advdiff-confirm-seed${seed}.json\"
  cat > \"$spec\" <<JSON
{
  \"tag\": \"cpu-advdiff-confirm-seed${seed}\",
  \"device\": \"cpu\",
  \"num_threads\": ${threads_per_job},
  \"seed\": ${seed},
  \"mesh_size\": 0.7,
  \"num_adaptations\": 8,
  \"epochs\": 300,
  \"export_images\": false,
  \"create_gifs\": false,
  \"generate_report\": true,
  \"methods_to_run\": [\"adaptive\", \"random\", \"halton\", \"rad\"],
  \"problem_name\": \"advection_diffusion\",
  \"problem_kwargs\": {},
  \"reference_mesh_factor\": 0.05
}
JSON

  echo \"[$(date -u +%Y-%m-%dT%H:%M:%SZ)] submit seed=${seed} threads=${threads_per_job}\"
  CONFIG_FILE=\"$config_file\" \"$remote_ops_dir/go.sh\" \
    --spec \"$spec\" \
    --session \"cpu-advdiff-confirm-seed${seed}\" \
    --wait
  echo \"[$(date -u +%Y-%m-%dT%H:%M:%SZ)] done seed=${seed}\"
}

active_jobs=0
for seed in \"${seeds[@]}\"; do
  submit_seed \"$seed\" &
  active_jobs=$((active_jobs + 1))
  if (( active_jobs >= parallel_jobs )); then
    wait -n
    active_jobs=$((active_jobs - 1))
  fi
done

wait
