#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ -d ".venv" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

if [[ $# -gt 0 ]]; then
  "${PYTHON_BIN}" -m src.training.continual_update --incremental-data "$1" --max-examples "${MAX_EXAMPLES:-200}"
else
  "${PYTHON_BIN}" -m src.training.continual_update --max-examples "${MAX_EXAMPLES:-200}"
fi
