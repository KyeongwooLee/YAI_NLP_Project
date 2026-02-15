#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ -d ".venv" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

if [[ -z "${BASE_MODEL_NAME:-}" ]]; then
  echo "BASE_MODEL_NAME is not set. Using config default: meta-llama/Llama-3.1-8B-Instruct"
fi

export LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-0}"

"${PYTHON_BIN}" -m src.pipeline.run_full_pipeline \
  --max-domains "${MAX_DOMAINS:-3}" \
  --max-examples-per-adapter "${MAX_EXAMPLES_PER_ADAPTER:-200}" \
  --eval-sample-size "${EVAL_SAMPLE_SIZE:-3}" \
  --query "${DEMO_QUERY:-Can you explain Archimedes principle with a simple example?}"
