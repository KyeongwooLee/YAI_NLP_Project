#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ -d ".venv" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

"${PYTHON_BIN}" -m src.pipeline.run_full_pipeline \
  --max-domains "${MAX_DOMAINS:-2}" \
  --max-examples-per-adapter "${MAX_EXAMPLES_PER_ADAPTER:-120}" \
  --eval-sample-size "${EVAL_SAMPLE_SIZE:-3}" \
  --query "${DEMO_QUERY:-Can you explain a topic using concise steps?}"
