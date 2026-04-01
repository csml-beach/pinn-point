#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <main.tex>" >&2
  exit 2
fi

main_tex="$1"
if [[ ! -f "$main_tex" ]]; then
  echo "Error: file not found: $main_tex" >&2
  exit 2
fi

workdir="$(cd "$(dirname "$main_tex")" && pwd)"
file="$(basename "$main_tex")"
stem="${file%.tex}"

cd "$workdir"

if command -v latexmk >/dev/null 2>&1; then
  latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error "$file"
  exit 0
fi

if ! command -v pdflatex >/dev/null 2>&1; then
  echo "Error: neither latexmk nor pdflatex is available on PATH." >&2
  exit 127
fi

pdflatex -interaction=nonstopmode -halt-on-error -file-line-error "$file"

if command -v bibtex >/dev/null 2>&1; then
  if [[ -f "${stem}.aux" ]] && grep -q '\\bibdata' "${stem}.aux"; then
    bibtex "$stem"
  fi
fi

pdflatex -interaction=nonstopmode -halt-on-error -file-line-error "$file"
pdflatex -interaction=nonstopmode -halt-on-error -file-line-error "$file"
