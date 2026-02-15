#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ -d ".venv" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

PERSONA="${1:-creative_gamified}"
"${PYTHON_BIN}" -m src.training.train_persona_lora --persona "${PERSONA}" --max-examples "${MAX_EXAMPLES:-400}"
