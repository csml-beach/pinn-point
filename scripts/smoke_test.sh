#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-$HOME/.pyenv/versions/netgen/bin/python}"

if [[ ! -x "$python_bin" ]]; then
  echo "Error: python interpreter not found: $python_bin" >&2
  exit 1
fi

export MPLBACKEND="${MPLBACKEND:-Agg}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/pinn-point-mpl}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/pinn-point-xdg-cache}"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

if [[ -z "${FONTCONFIG_PATH:-}" && -d "/opt/homebrew/etc/fonts" ]]; then
  export FONTCONFIG_PATH="/opt/homebrew/etc/fonts"
fi

cd "$repo_root"
exec "$python_bin" train/main.py smoke "$@"
