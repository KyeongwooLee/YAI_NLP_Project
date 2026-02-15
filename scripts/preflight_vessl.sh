#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ -d ".venv" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

echo "[1/5] GPU check"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
else
  echo "Warning: nvidia-smi not found."
fi

echo "[2/5] Python package check"
"${PYTHON_BIN}" - <<'PY'
import importlib
mods = ["torch", "transformers", "peft", "datasets", "accelerate", "safetensors"]
for m in mods:
    importlib.import_module(m)
print("Core packages import OK")
PY

echo "[3/5] Torch CUDA check"
"${PYTHON_BIN}" - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch.cuda:", torch.version.cuda)
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu_count:", torch.cuda.device_count())
    print("gpu_name:", torch.cuda.get_device_name(0))
PY

echo "[4/5] Model config check"
if [[ -z "${BASE_MODEL_NAME:-}" ]]; then
  echo "BASE_MODEL_NAME is not set. Default model will be used: meta-llama/Llama-3.1-8B-Instruct"
else
  echo "BASE_MODEL_NAME=${BASE_MODEL_NAME}"
fi
echo "LOCAL_FILES_ONLY=${LOCAL_FILES_ONLY:-unset}"
if [[ "${LOCAL_FILES_ONLY:-0}" != "1" ]]; then
  if [[ -z "${HUGGING_FACE_HUB_TOKEN:-}" && -z "${HF_TOKEN:-}" ]]; then
    echo "Warning: HF token is not set. Llama-3 gated model download may fail."
  fi
fi

echo "[5/5] Dataset path check"
if [[ -f "data/Education-Dialogue-Dataset-main/conversations_train1.json" ]]; then
  echo "Dataset path OK"
else
  echo "Warning: dataset files not found at expected path."
fi

echo "Preflight check complete."
