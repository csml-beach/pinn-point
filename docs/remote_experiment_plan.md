# Remote Experiment Execution Plan

## Goal

Add the lightest useful system for launching experiments on a remote GPU machine, checking whether they are still running, and syncing the finished results back into this repo.

The goal is to support many experiments soon without bloating the codebase.

## Recommendation

Start with a minimal SSH-based setup.

Do not start with:
- a job queue
- worker daemons
- heartbeat files
- JSON manifest orchestration
- Kubernetes

Those can all come later if we actually need them.

## Why This Minimal Version

This repo is still evolving:
- the `ngsolve/netgen` environment is specialized
- experiment commands are still changing
- the comparison protocol is still being refined

So the right first step is:
- one remote machine
- one pinned environment
- one repo checkout
- a few small shell scripts

That gives us remote execution without turning the repo into an experiment platform.

## Minimal Architecture

### Remote Machine

Use one SSH-accessible GPU machine with:
- NVIDIA GPU and working CUDA for PyTorch
- the pinned `netgen` environment
- a repo checkout
- `tmux`
- `rsync`

### Local Interface

Add a very small script layer:
- `scripts/remote_run.sh`
- `scripts/remote_status.sh`
- `scripts/remote_sync.sh`
- optional `scripts/remote_attach.sh`

These scripts should use plain:
- `ssh`
- `tmux`
- `rsync`

No persistent Python service is needed in the first version.

## Proposed Workflow

### 1. Launch

`remote_run.sh` should:
- connect to the remote host over SSH
- move into the repo checkout
- optionally `git fetch` and `git checkout` a chosen commit
- activate the correct environment
- start the experiment inside a named `tmux` session
- redirect stdout/stderr to a log file

Example remote behavior:

```bash
tmux new-session -d -s pinn-seed123 \
  'cd ~/runs/pinn-point/repo && \
   source ~/.zshrc && \
   export PINN_DEVICE=cuda:0 && \
   ~/.pyenv/versions/netgen/bin/python train/main.py main --seed 123 \
   > outputs/remote-seed123.log 2>&1'
```

### 2. Status

`remote_status.sh` should:
- show running `tmux` sessions
- optionally run `nvidia-smi`
- optionally tail the current log file for a named run

That is enough to answer:
- is the run still alive?
- is the GPU busy?
- what did the last few log lines say?

### 3. Attach

Optional:

`remote_attach.sh` should:
- attach to a chosen remote `tmux` session for live inspection

This is useful when debugging a failing or slow run.

### 4. Sync Results

`remote_sync.sh` should:
- use `rsync`
- pull one run or a set of runs from remote `outputs/`
- copy them into local `outputs/`

Two useful modes:
- sync one run by run ID
- sync all recent remote runs matching a pattern

## Minimal Configuration

Keep remote configuration simple.

One small config file or `.env` should be enough, for example:
- remote host
- remote user
- remote repo path
- remote python path

This avoids baking machine-specific values into multiple scripts.

## Implementation Scope

### Step 1. Remote Config

Add one lightweight config source for:
- `REMOTE_HOST`
- `REMOTE_USER`
- `REMOTE_REPO_PATH`
- `REMOTE_PYTHON`

This can be a sourced shell file, not a Python system.

### Step 2. `remote_run.sh`

Responsibilities:
- choose GPU/device
- choose command/mode
- choose seed/tag
- launch in remote `tmux`

Initial version can accept a raw command tail after a few required options.

### Step 3. `remote_status.sh`

Responsibilities:
- list remote `tmux` sessions
- show `nvidia-smi`
- optionally tail a log

### Step 4. `remote_sync.sh`

Responsibilities:
- sync one run or many runs from remote `outputs/`

### Step 5. Optional `remote_attach.sh`

Responsibilities:
- attach to a remote `tmux` session

## Logging

Keep logging minimal too.

For the first version, each launched run only needs:
- the normal run artifacts already written under `outputs/<run-id>/`
- one shell log file capturing stdout/stderr

We do not need extra run-state JSON yet.

## What We Are Explicitly Not Building Yet

Not in the first version:
- queueing
- multi-worker scheduling
- structured job manifests
- automatic retries
- heartbeat monitoring
- stale-job recovery
- Kubernetes job specs
- container images

Those are all valid future steps, but only if the simple SSH model becomes painful.

## When To Upgrade Later

We should only add the heavier machinery if one of these becomes true:
- we regularly run many overlapping jobs
- we need multiple GPUs managed automatically
- we lose track of runs with only `tmux` and logs
- we need resumable/retriable batch execution
- we move to shared cluster infrastructure

At that point, the next step should probably be:
- simple JSON manifests
- a single Python worker

Still not Kubernetes yet unless scale truly demands it.

## Kubernetes Later

Kubernetes remains a possible later backend, but it should stay out of scope for the first implementation.

It would require:
- CUDA container images
- a reproducible Netgen/NGSolve installation in the image
- job specs
- persistent storage
- log routing
- cluster-side monitoring

That is a separate project, not the first remote-run solution.

## First Implementation Target

Build only this:
- one remote-machine config
- `scripts/remote_run.sh`
- `scripts/remote_status.sh`
- `scripts/remote_sync.sh`
- optional `scripts/remote_attach.sh`

If that works well, it is enough for the next experimental phase.
