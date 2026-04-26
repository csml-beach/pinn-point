#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  submit_m3_large_cpu_allen_cahn_obstacles_screen_confirm.sh [--parallel N] [--threads N] [--epochs N] [--iterations N] [--seeds CSV] [--methods CSV] [--commit SHA] [--config PATH] [--sync-root PATH] [--reference-mesh-factor X] [--mesh-size X] [--session-prefix NAME] [--skip-setup]

Description:
  Submit an Allen-Cahn obstacles screen confirm study on m3-large-cpu
  for adaptive_power_tempered, adaptive_halton_base, adaptive_persistent,
  adaptive, random, halton, and rad.
EOF
}

parallel_jobs=10
threads_per_job=1
epochs=200
iterations=6
mesh_size="0.18"
commit_sha="$(git rev-parse origin/codex/navier-stokes-channel-obstacle)"
config_file=""
sync_root=""
reference_mesh_factor="0.05"
session_prefix="cpu-allen-cahn-screen"
skip_setup=false
seeds_csv="42,123,456,789,1011,2022,3033,4044,5055,6066,7077,8088,9099,11111,12121,13131,14141,15151,16161,17171"
# Paper-facing default suite. Negative/tuning variants remain selectable via
# --methods, but are intentionally not included by default.
methods_csv="adaptive_power_tempered,adaptive_halton_base,adaptive_persistent,adaptive,random,halton,rad"

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
    --epochs)
      epochs="$2"
      shift 2
      ;;
    --iterations)
      iterations="$2"
      shift 2
      ;;
    --mesh-size)
      mesh_size="$2"
      shift 2
      ;;
    --seeds)
      seeds_csv="$2"
      shift 2
      ;;
    --methods)
      methods_csv="$2"
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
    --reference-mesh-factor)
      reference_mesh_factor="$2"
      shift 2
      ;;
    --session-prefix)
      session_prefix="$2"
      shift 2
      ;;
    --skip-setup)
      skip_setup=true
      shift
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
  config_file="$remote_ops_dir/config.m3-large-cpu.env"
fi
if [[ -z "$sync_root" ]]; then
  sync_root="$repo_root/outputs/m3-large-cpu-allen-cahn-obstacles-screen-confirm-${epochs}e"
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

runner_dir="/tmp/pinn_point_m3_allen_cahn_screen_confirm_runners"
manifest_dir="$sync_root/_manifests"
mkdir -p "$runner_dir" "$manifest_dir" "$sync_root"

IFS=',' read -r -a seeds <<< "$seeds_csv"
env_prefix="$(remote_env_prefix)"
ld_prefix=""
if [[ -n "${REMOTE_VENV_PATH:-}" ]]; then
  ld_prefix="LD_LIBRARY_PATH=$REMOTE_VENV_PATH/lib:\${LD_LIBRARY_PATH:-} "
fi

if [[ "$skip_setup" != true ]]; then
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

  echo "[submit] syncing local Allen-Cahn overlay"
  remote_copy_to "$repo_root/train/config.py" "$REMOTE_REPO_PATH/train/config.py"
  remote_copy_to "$repo_root/train/experiments.py" "$REMOTE_REPO_PATH/train/experiments.py"
  remote_copy_to "$repo_root/train/main.py" "$REMOTE_REPO_PATH/train/main.py"
  remote_copy_to "$repo_root/train/mesh_refinement.py" "$REMOTE_REPO_PATH/train/mesh_refinement.py"
  remote_copy_to "$repo_root/train/pinn_model.py" "$REMOTE_REPO_PATH/train/pinn_model.py"
  remote_copy_to "$repo_root/train/training.py" "$REMOTE_REPO_PATH/train/training.py"
  remote_copy_to "$repo_root/train/utils.py" "$REMOTE_REPO_PATH/train/utils.py"
  remote_copy_to "$repo_root/train/visualization.py" "$REMOTE_REPO_PATH/train/visualization.py"
  for local_path in "$repo_root"/train/methods/*.py; do
    remote_copy_to "$local_path" "$REMOTE_REPO_PATH/train/methods/$(basename "$local_path")"
  done
  for local_path in "$repo_root"/train/problems/*.py; do
    remote_copy_to "$local_path" "$REMOTE_REPO_PATH/train/problems/$(basename "$local_path")"
  done
else
  echo "[submit] skipping remote setup; using existing repo at $REMOTE_REPO_PATH"
fi

submit_seed() {
  local seed="$1"
  local session="${session_prefix}-seed${seed}"
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
export VECLIB_MAXIMUM_THREADS=$threads_per_job
export NGS_NUM_THREADS=$threads_per_job
export PINN_RUN_TAG_SUFFIX='$session_prefix'

VALIDATION_OPTIONS='{"restore_best_epoch_checkpoint": false}'

set +e
${env_prefix}${ld_prefix}PYTHONUNBUFFERED=1 '$REMOTE_PYTHON' train/main.py \\
  --device cpu \\
  screen \\
  --problem allen_cahn_obstacles_2d \\
  --methods $methods_csv \\
  --seed $seed \\
  --iterations $iterations \\
  --epochs $epochs \\
  --mesh-size $mesh_size \\
  --reference-mesh-factor $reference_mesh_factor \\
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
