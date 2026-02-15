#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ -d ".venv" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

DOMAIN="${1:-math}"
"${PYTHON_BIN}" -m src.training.train_domain_lora --domain "${DOMAIN}" --max-examples "${MAX_EXAMPLES:-400}"
