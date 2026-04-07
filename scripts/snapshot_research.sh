#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  snapshot_research.sh [-m message] [-d artifact_dirs] [--dvc-commit] [--dry-run]

Options:
  -m, --message        Git commit message.
  -d, --artifact-dirs  Comma-separated dirs to DVC-track.
                       Default: artifacts/figures,artifacts/metrics,artifacts/animations,outputs
      --dvc-commit     Run `dvc commit` after git commit when a DVC pipeline is in use.
      --dry-run        Print planned actions only.
  -h, --help           Show this help.
USAGE
}

find_dvc() {
  if command -v dvc >/dev/null 2>&1; then
    command -v dvc
    return 0
  fi

  if [[ -x "${HOME}/.pyenv/versions/jax/bin/dvc" ]]; then
    printf '%s\n' "${HOME}/.pyenv/versions/jax/bin/dvc"
    return 0
  fi

  return 1
}

if ! command -v git >/dev/null 2>&1; then
  echo "error: git is required" >&2
  exit 1
fi

commit_msg=""
artifact_dirs="${ARTIFACT_DIRS:-artifacts/figures,artifacts/metrics,artifacts/animations,outputs}"
run_dvc_commit=0
dry_run=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -m|--message)
      commit_msg="$2"
      shift 2
      ;;
    -d|--artifact-dirs)
      artifact_dirs="$2"
      shift 2
      ;;
    --dvc-commit)
      run_dvc_commit=1
      shift
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "error: run from inside a git repository" >&2
  exit 1
fi

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

echo "repo root: $repo_root"

stage_path() {
  local path="$1"
  if [[ $dry_run -eq 1 ]]; then
    echo "[dry-run] git add $path"
  else
    git add "$path"
  fi
}

has_files() {
  local dir="$1"
  [[ -d "$dir" ]] || return 1
  find "$dir" -type f -not -name '.DS_Store' -print -quit | grep -q .
}

dvc_cmd=""
if dvc_cmd="$(find_dvc)"; then
  if [[ ! -d .dvc ]]; then
    if [[ $dry_run -eq 1 ]]; then
      echo "[dry-run] $dvc_cmd init -q"
    else
      "$dvc_cmd" init -q
      stage_path ".dvc"
      [[ -f .dvcignore ]] && stage_path ".dvcignore"
    fi
  fi

  IFS=',' read -r -a dirs <<< "$artifact_dirs"
  for raw in "${dirs[@]}"; do
    dir="$(echo "$raw" | sed 's/^ *//; s/ *$//')"
    [[ -n "$dir" ]] || continue
    if has_files "$dir"; then
      if [[ $dry_run -eq 1 ]]; then
        echo "[dry-run] $dvc_cmd add $dir"
      else
        "$dvc_cmd" add "$dir"
      fi
      [[ -f "${dir}.dvc" ]] && stage_path "${dir}.dvc"
      [[ -f "$dir/.gitignore" ]] && stage_path "$dir/.gitignore"
    fi
  done

  if [[ -d .dvc ]]; then
    stage_path ".dvc"
    [[ -f .dvcignore ]] && stage_path ".dvcignore"
  fi
else
  echo "warning: dvc not found; skipping dvc add"
fi

if [[ $dry_run -eq 1 ]]; then
  echo "[dry-run] git add -u"
else
  git add -u
fi

while IFS= read -r path; do
  [[ -n "$path" ]] || continue
  case "$path" in
    *.py|*.sh|*.R|*.jl|*.c|*.cc|*.cpp|*.h|*.hpp|*.rs|*.ipynb|*.md|*.tex|*.bib|*.toml|*.yaml|*.yml|*.json|*.txt|*.csv|*.tsv|*.dvc|README.md|docs/*|paper/*|scripts/*|src/*|experiments/*|configs/*|config/*|notebooks/*|.github/*|dvc.yaml|dvc.lock|.gitignore|*/.gitignore)
      stage_path "$path"
      ;;
  esac
done < <(git ls-files --others --exclude-standard)

if [[ $dry_run -eq 1 ]]; then
  echo "[dry-run] snapshot preparation complete"
  exit 0
fi

if git diff --cached --quiet; then
  echo "no staged changes; nothing to commit"
  exit 0
fi

if [[ -z "$commit_msg" ]]; then
  commit_msg="snapshot: $(date +'%Y-%m-%d %H:%M:%S')"
fi

git commit -m "$commit_msg"

if [[ $run_dvc_commit -eq 1 ]] && [[ -n "$dvc_cmd" ]] && [[ -f dvc.yaml ]]; then
  "$dvc_cmd" commit || true
fi

echo "snapshot complete"
