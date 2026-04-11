#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  submit_jetstream_gpu_cvae_fashion.sh [options] [-- <extra-cvae-args>]

Description:
  Submit scripts/cvae_fashion_mnist.py to Jetstream GPU via sibling remote-ops.
  Bootstraps the remote repo/venv, ensures torch+torchvision are installed,
  launches a tmux session, and optionally waits/syncs artifacts.

Options:
  --gpu N              GPU id to expose via CUDA_VISIBLE_DEVICES (default: inferred from --stage)
  --stage NAME         Stage lock name: gpu0|gpu1 (default: gpu0)
  --session NAME       Remote tmux session name (default: cvae-fashion-<timestamp>)
  --commit SHA         Commit/branch/tag to checkout remotely (default: current local HEAD short SHA)
  --config PATH        remote-ops config file (default: ../remote-ops/pinn-point/config.jetstream-medium.env)
  --no-bootstrap       Skip remote bootstrap step
  --no-install-torch   Skip remote torch/torchvision check/install
  --wait               Wait for job completion and sync output directory back

  CVAE args:
    --epochs N
    --batch-size N
    --hidden-dim N
    --latent-dim N
    --beta X
    --lr X
    --num-workers N
    --seed N
    --sample-cols N
    --log-every N
    --limit-train-batches N
    --limit-test-batches N
    --data-dir PATH
    --out-subdir PATH  Remote output subdir under repo root (default: outputs/jetstream-cvae-fashion/<session>)
EOF
}

gpu_id=""
stage="gpu0"
session_name_raw=""
commit_sha="$(git rev-parse --short HEAD)"
config_file=""
do_bootstrap=1
do_install_torch=1
wait_and_sync=0

epochs=10
batch_size=128
hidden_dim=512
latent_dim=16
beta=1.0
lr=1e-3
num_workers=4
seed=42
sample_cols=8
log_every=100
limit_train_batches=0
limit_test_batches=0
data_dir="data"
out_subdir=""
extra_cvae_args=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)
      gpu_id="$2"
      shift 2
      ;;
    --stage)
      stage="$2"
      shift 2
      ;;
    --session)
      session_name_raw="$2"
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
    --no-bootstrap)
      do_bootstrap=0
      shift
      ;;
    --no-install-torch)
      do_install_torch=0
      shift
      ;;
    --wait)
      wait_and_sync=1
      shift
      ;;
    --epochs)
      epochs="$2"
      shift 2
      ;;
    --batch-size)
      batch_size="$2"
      shift 2
      ;;
    --hidden-dim)
      hidden_dim="$2"
      shift 2
      ;;
    --latent-dim)
      latent_dim="$2"
      shift 2
      ;;
    --beta)
      beta="$2"
      shift 2
      ;;
    --lr)
      lr="$2"
      shift 2
      ;;
    --num-workers)
      num_workers="$2"
      shift 2
      ;;
    --seed)
      seed="$2"
      shift 2
      ;;
    --sample-cols)
      sample_cols="$2"
      shift 2
      ;;
    --log-every)
      log_every="$2"
      shift 2
      ;;
    --limit-train-batches)
      limit_train_batches="$2"
      shift 2
      ;;
    --limit-test-batches)
      limit_test_batches="$2"
      shift 2
      ;;
    --data-dir)
      data_dir="$2"
      shift 2
      ;;
    --out-subdir)
      out_subdir="$2"
      shift 2
      ;;
    --)
      shift
      extra_cvae_args=("$@")
      break
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

if [[ ! -d "$remote_ops_dir" ]]; then
  echo "remote-ops dir not found: $remote_ops_dir" >&2
  exit 1
fi
if [[ ! -f "$config_file" ]]; then
  echo "Config file not found: $config_file" >&2
  exit 1
fi

case "$stage" in
  gpu0)
    stage_gpu="0"
    ;;
  gpu1)
    stage_gpu="1"
    ;;
  "")
    stage_gpu=""
    ;;
  *)
    echo "Invalid --stage '$stage' (expected gpu0|gpu1 or empty)." >&2
    exit 1
    ;;
esac

if [[ -z "$gpu_id" && -n "$stage_gpu" ]]; then
  gpu_id="$stage_gpu"
fi
if [[ -n "$stage_gpu" && -n "$gpu_id" && "$gpu_id" != "$stage_gpu" ]]; then
  echo "--gpu ($gpu_id) conflicts with --stage $stage (gpu $stage_gpu)." >&2
  exit 1
fi
if [[ -z "$gpu_id" ]]; then
  echo "No GPU selected. Provide --gpu N or --stage gpu0|gpu1." >&2
  exit 1
fi

timestamp="$(date '+%Y%m%d-%H%M%S')"
if [[ -z "$session_name_raw" ]]; then
  session_name_raw="cvae-fashion-${timestamp}"
fi
session_name="$(printf '%s' "$session_name_raw" | tr -c 'A-Za-z0-9_-' '_')"
if [[ -z "$session_name" ]]; then
  echo "Invalid session name: $session_name_raw" >&2
  exit 1
fi

if [[ -z "$out_subdir" ]]; then
  out_subdir="outputs/jetstream-cvae-fashion/${session_name}"
fi

export CONFIG_FILE="$config_file"
# shellcheck disable=SC1090
source "$remote_ops_dir/lib.sh"

echo "[submit] remote target: $REMOTE_SSH_TARGET"
echo "[submit] session: $session_name"
echo "[submit] gpu: $gpu_id"
echo "[submit] commit: $commit_sha"
echo "[submit] out-subdir: $out_subdir"

if [[ "$do_bootstrap" -eq 1 ]]; then
  echo "[submit] bootstrapping remote repo/env"
  "$remote_ops_dir/bootstrap_remote.sh"
fi

echo "[submit] pinning remote repo to commit $commit_sha"
remote_exec "
  set -euo pipefail
  cd '$REMOTE_REPO_PATH'
  git fetch --all --prune
  git checkout '$commit_sha'
"

if [[ "$do_install_torch" -eq 1 ]]; then
  echo "[submit] ensuring torch + torchvision in remote env"
  remote_exec "
    set -euo pipefail
    '$REMOTE_PYTHON' -m pip install --upgrade pip
    if ! '$REMOTE_PYTHON' - <<'PY'
import importlib.util
import sys
missing = [m for m in ('torch', 'torchvision') if importlib.util.find_spec(m) is None]
if missing:
    print('missing: ' + ','.join(missing))
    sys.exit(1)
PY
    then
      # Prefer a CUDA 12.1 build that is compatible with older 12.x drivers.
      '$REMOTE_PYTHON' -m pip install --upgrade \
        --index-url https://download.pytorch.org/whl/cu121 \
        torch==2.5.1 torchvision==0.20.1
    fi
    if ! '$REMOTE_PYTHON' - <<'PY'
import sys
import torch
import torchvision
print(f'torch={torch.__version__}')
print(f'torchvision={torchvision.__version__}')
print(f'cuda_available={torch.cuda.is_available()}')
if not torch.cuda.is_available():
    sys.exit(1)
PY
    then
      echo '[submit] CUDA still unavailable; forcing cu121 torch/vision pair'
      '$REMOTE_PYTHON' -m pip install --upgrade \
        --index-url https://download.pytorch.org/whl/cu121 \
        torch==2.5.1 torchvision==0.20.1
    fi
    '$REMOTE_PYTHON' - <<'PY'
import torch
import torchvision
print(f'torch={torch.__version__}')
print(f'torchvision={torchvision.__version__}')
print(f'cuda_available={torch.cuda.is_available()}')
if not torch.cuda.is_available():
    raise SystemExit('CUDA is still unavailable in remote python env.')
PY
  "
fi

remote_stage_lock=""
if [[ -n "$stage" ]]; then
  remote_stage_lock="$REMOTE_REPO_PATH/.remote_opps/locks/${stage}.lock"
fi

echo "[submit] preparing remote runner"
ensure_remote_dirs
remote_exec "mkdir -p '$REMOTE_REPO_PATH/.remote_opps/logs' '$REMOTE_REPO_PATH/.remote_opps/locks'"

if [[ -n "$remote_stage_lock" ]]; then
  echo "[submit] acquiring stage lock: $stage"
  remote_exec "
    set -euo pipefail
    lock='$remote_stage_lock'
    if [[ -f \"\$lock\" ]]; then
      owner=\$(cat \"\$lock\" 2>/dev/null || true)
      if [[ -n \"\$owner\" ]] && tmux has-session -t \"\$owner\" >/dev/null 2>&1; then
        echo \"stage '$stage' is busy by session \$owner\" >&2
        exit 9
      fi
      rm -f \"\$lock\"
    fi
    printf '%s\n' '$session_name' > \"\$lock\"
  "
fi

remote_log="$REMOTE_REPO_PATH/.remote_opps/logs/${session_name}.log"
remote_runner="$REMOTE_REPO_PATH/.remote_opps/run_${session_name}.sh"
remote_out_abs="$REMOTE_REPO_PATH/$out_subdir"
remote_cvae_script="$REMOTE_REPO_PATH/scripts/cvae_fashion_mnist.py"

if [[ ! -f "$repo_root/scripts/cvae_fashion_mnist.py" ]]; then
  echo "Local CVAE script not found: $repo_root/scripts/cvae_fashion_mnist.py" >&2
  exit 1
fi
echo "[submit] syncing local CVAE script to remote repo checkout"
remote_exec "mkdir -p '$REMOTE_REPO_PATH/scripts'"
remote_copy_to "$repo_root/scripts/cvae_fashion_mnist.py" "$remote_cvae_script"

env_prefix="$(remote_env_prefix)"
cuda_prefix="CUDA_VISIBLE_DEVICES=$gpu_id "
ld_prefix=""
if [[ -n "${REMOTE_VENV_PATH:-}" ]]; then
  ld_prefix="LD_LIBRARY_PATH=$REMOTE_VENV_PATH/lib "
fi

cvae_cmd=(
  "$REMOTE_PYTHON"
  "$remote_cvae_script"
  --device cuda
  --data-dir "$data_dir"
  --out-dir "$out_subdir"
  --epochs "$epochs"
  --batch-size "$batch_size"
  --hidden-dim "$hidden_dim"
  --latent-dim "$latent_dim"
  --beta "$beta"
  --lr "$lr"
  --num-workers "$num_workers"
  --seed "$seed"
  --sample-cols "$sample_cols"
  --log-every "$log_every"
  --limit-train-batches "$limit_train_batches"
  --limit-test-batches "$limit_test_batches"
)
if [[ "${#extra_cvae_args[@]}" -gt 0 ]]; then
  cvae_cmd+=("${extra_cvae_args[@]}")
fi

cvae_cmd_quoted=""
for arg in "${cvae_cmd[@]}"; do
  printf -v q "%q" "$arg"
  cvae_cmd_quoted+="$q "
done
cvae_cmd_quoted="${cvae_cmd_quoted% }"

remote_exec "
cat > '$remote_runner' <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
cd '$REMOTE_REPO_PATH'
LOCK_FILE='$remote_stage_lock'
LOG_FILE='$remote_log'
OUT_DIR='$remote_out_abs'

mkdir -p \"\$(dirname \"\$OUT_DIR\")\"
exec > >(tee -a \"\$LOG_FILE\") 2>&1

cleanup() {
  if [[ -n \"\$LOCK_FILE\" ]] && [[ -f \"\$LOCK_FILE\" ]]; then
    owner=\$(cat \"\$LOCK_FILE\" 2>/dev/null || true)
    if [[ \"\$owner\" == '$session_name' ]]; then
      rm -f \"\$LOCK_FILE\"
    fi
  fi
}
trap cleanup EXIT INT TERM

echo \"[cvae_entry] started_at=\$(date -u '+%Y-%m-%dT%H:%M:%SZ')\"
echo \"[cvae_entry] out_dir=$out_subdir\"
set +e
${env_prefix}${cuda_prefix}${ld_prefix}PYTHONUNBUFFERED=1 $cvae_cmd_quoted
rc=\$?
set -e
echo \"[cvae_entry] exit_code=\$rc\"
echo \"[cvae_entry] finished_at=\$(date -u '+%Y-%m-%dT%H:%M:%SZ')\"
exit \"\$rc\"
EOF
chmod +x '$remote_runner'
"

echo "[submit] launching tmux session"
remote_exec "tmux new-session -d -s '$session_name' \"'$remote_runner'\""

echo
echo "Started session: $session_name"
echo "Remote log: $remote_log"
echo "Remote output: $remote_out_abs"
echo
echo "Monitor:"
echo "  CONFIG_FILE=$config_file $remote_ops_dir/status.sh --session $session_name --follow"
echo "  CONFIG_FILE=$config_file $remote_ops_dir/logs.sh --session $session_name --follow"
echo
echo "Sync later:"
echo "  rsync -az --partial -e \"ssh\" $REMOTE_SSH_TARGET:$remote_out_abs/ $LOCAL_SYNC_ROOT/$session_name/"

if [[ "$wait_and_sync" -eq 1 ]]; then
  echo "[submit] waiting for session to finish..."
  while remote_exec "tmux has-session -t '$session_name' >/dev/null 2>&1"; do
    sleep "$STATUS_POLL_SECONDS"
  done
  echo "[submit] session finished; syncing outputs"
  local_out="$LOCAL_SYNC_ROOT/$session_name"
  mkdir -p "$local_out"
  rsync -az --partial -e "ssh ${ssh_opts[*]}" "$REMOTE_SSH_TARGET:$remote_out_abs/" "$local_out/"
  echo "[submit] synced: $local_out"
fi
