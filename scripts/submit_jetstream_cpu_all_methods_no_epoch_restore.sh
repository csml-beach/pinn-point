#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  submit_jetstream_cpu_all_methods_no_epoch_restore.sh [--parallel N] [--threads N] [--commit SHA] [--config PATH] [--sync-root PATH]

Description:
  Submit a 10-seed Jetstream CPU study for adaptive_persistent, adaptive,
  random, halton, and rad on advection_diffusion with iteration-level
  checkpoint selection enabled and epoch-level best-checkpoint restore disabled.
EOF
}

parallel_jobs=5
threads_per_job=3
commit_sha="$(git rev-parse --short HEAD)"
config_file=""
sync_root=""

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
    --sync-root)
      sync_root="$2"
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
if [[ -z "$sync_root" ]]; then
  sync_root="$repo_root/outputs/jetstream-medium-advection-persistent-fiveway-no-epoch-restore"
fi

if [[ ! -f "$config_file" ]]; then
  echo "Config file not found: $config_file" >&2
  exit 1
fi
if [[ ! -d "$remote_ops_dir" ]]; then
  echo "remote-ops dir not found: $remote_ops_dir" >&2
  exit 1
fi

export CONFIG_FILE="$config_file"
# shellcheck disable=SC1090
source "$remote_ops_dir/lib.sh"

runner_dir="/tmp/pinn_point_jetstream_persistent_fiveway_no_epoch_restore_runners"
manifest_dir="$sync_root/_manifests"
mkdir -p "$runner_dir" "$manifest_dir" "$sync_root"

seeds=(42 123 456 789 1011 2022 3033 4044 5055 6066)
env_prefix="$(remote_env_prefix)"
ld_prefix=""
if [[ -n "${REMOTE_VENV_PATH:-}" ]]; then
  ld_prefix="LD_LIBRARY_PATH=$REMOTE_VENV_PATH/lib "
fi

echo "[submit] bootstrapping remote repo"
CONFIG_FILE="$config_file" "$remote_ops_dir/bootstrap_remote.sh"

echo "[submit] pinning remote repo to commit $commit_sha"
remote_exec "
  set -euo pipefail
  cd '$REMOTE_REPO_PATH'
  git fetch --all --prune
  git checkout '$commit_sha'
  mkdir -p '$REMOTE_REPO_PATH/.remote_opps/logs'
"

submit_seed() {
  local seed="$1"
  local session="cpu-persist-fiveway-seed${seed}"
  local remote_log="$REMOTE_REPO_PATH/.remote_opps/logs/${session}.log"
  local remote_runner="$REMOTE_REPO_PATH/.remote_opps/run_${session}.sh"
  local local_runner="$runner_dir/${session}.sh"
  local local_manifest="$manifest_dir/${session}.manifest.json"

  cat > "$local_runner" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd '$REMOTE_REPO_PATH'
LOG_FILE='$remote_log'
exec > >(tee -a "\$LOG_FILE") 2>&1

echo "[remote_entry] started_at=\$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "[remote_entry] session=$session"

export OMP_NUM_THREADS=$threads_per_job
export MKL_NUM_THREADS=$threads_per_job
export OPENBLAS_NUM_THREADS=$threads_per_job
export NUMEXPR_NUM_THREADS=$threads_per_job

VALIDATION_OPTIONS='{"restore_best_epoch_checkpoint": false}'

set +e
${env_prefix}${ld_prefix}PYTHONUNBUFFERED=1 '$REMOTE_PYTHON' train/main.py \\
  --device cpu \\
  screen \\
  --seed $seed \\
  --problem advection_diffusion \\
  --methods adaptive_persistent,adaptive,random,halton,rad \\
  --iterations 8 \\
  --epochs 300 \\
  --reference-mesh-factor 0.05 \\
  --validation-options "\$VALIDATION_OPTIONS"
rc=\$?
set -e

run_id=\$(grep -Eo 'Screen run ID: [^[:space:]]+' "\$LOG_FILE" | tail -n1 | awk '{print \$4}')
if [[ -n "\$run_id" ]]; then
  echo "[remote_entry] run_id=\$run_id"
fi
echo "[remote_entry] exit_code=\$rc"
echo "[remote_entry] finished_at=\$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
exit "\$rc"
EOF

  chmod +x "$local_runner"
  remote_copy_to "$local_runner" "$remote_runner"

  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] submit seed=${seed} session=${session}"
  remote_exec "
    set -euo pipefail
    rm -f '$remote_log'
    chmod +x '$remote_runner'
    tmux kill-session -t '$session' >/dev/null 2>&1 || true
    tmux new-session -d -s '$session' \"'$remote_runner'\"
  "

  while remote_exec "tmux has-session -t '$session' >/dev/null 2>&1"; do
    sleep 10
  done

  local run_id
  run_id="$(
    remote_exec "grep -Eo '\\[remote_entry\\] run_id=[^[:space:]]+' '$remote_log' | tail -n1 | cut -d= -f2" \
      | tr -d '\r'
  )"
  if [[ -z "$run_id" ]]; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] failed seed=${seed}: could not resolve run_id" >&2
    return 1
  fi

  local exit_code
  exit_code="$(
    remote_exec "grep -Eo '\\[remote_entry\\] exit_code=[0-9]+' '$remote_log' | tail -n1 | cut -d= -f2" \
      | tr -d '\r'
  )"
  if [[ -z "$exit_code" ]]; then
    exit_code="null"
  fi

  local local_output="$sync_root/$run_id"
  mkdir -p "$local_output"
  remote_copy_from "$REMOTE_REPO_PATH/outputs/$run_id/" "$local_output/"

  cat > "$local_manifest" <<JSON
{
  "session": "$session",
  "seed": $seed,
  "commit": "$commit_sha",
  "run_id": "$run_id",
  "local_output_path": "$local_output",
  "exit_code": $exit_code
}
JSON

  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] done seed=${seed} run_id=${run_id} exit_code=${exit_code}"
}

for seed in "${seeds[@]}"; do
  while (( $(jobs -pr | wc -l | tr -d ' ') >= parallel_jobs )); do
    sleep 1
  done
  submit_seed "$seed" &
done

wait
echo "[submit] all seeds completed"
