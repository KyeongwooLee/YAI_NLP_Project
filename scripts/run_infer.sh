#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ -d ".venv" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

QUERY="${1:-Can you explain Newton's second law in a simple way?}"
"${PYTHON_BIN}" -m src.inference.generate \
  --query "${QUERY}" \
  --student-preference "${STUDENT_PREFERENCE:-}" \
  --teacher-preference "${TEACHER_PREFERENCE:-}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-180}"
