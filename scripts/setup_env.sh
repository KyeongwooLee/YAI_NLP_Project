#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
CREATE_VENV="${CREATE_VENV:-1}"
OFFLINE_MODE="${OFFLINE_MODE:-0}"
INSTALL_TORCH="${INSTALL_TORCH:-0}"
USE_4BIT="${USE_4BIT:-0}"

if [[ "${CREATE_VENV}" == "1" && ! -d "${VENV_DIR}" ]]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

if [[ -d "${VENV_DIR}" ]]; then
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
fi

"${PYTHON_BIN}" -m pip install --upgrade pip setuptools wheel

if [[ "${OFFLINE_MODE}" != "1" ]]; then
  if [[ "${INSTALL_TORCH}" == "1" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
      "${PYTHON_BIN}" -m pip install --upgrade \
        --index-url https://download.pytorch.org/whl/cu121 \
        torch torchvision torchaudio
    else
      "${PYTHON_BIN}" -m pip install --upgrade \
        --index-url https://download.pytorch.org/whl/cpu \
        torch torchvision torchaudio
    fi
  else
    echo "INSTALL_TORCH=0: keeping existing torch from container image."
  fi
  "${PYTHON_BIN}" -m pip install --upgrade -r requirements.txt
  if [[ "${USE_4BIT}" == "1" ]]; then
    "${PYTHON_BIN}" -m pip install --upgrade bitsandbytes || {
      echo "Warning: bitsandbytes install failed. USE_4BIT=1 may not work."
    }
  fi
else
  echo "OFFLINE_MODE=1: skipping pip download/install steps."
fi

"${PYTHON_BIN}" - <<'PY'
import importlib
required = [
    "torch",
    "transformers",
    "huggingface_hub",
    "peft",
    "datasets",
    "safetensors",
    "accelerate",
]
for name in required:
    importlib.import_module(name)
try:
    import torch
    print(f"torch={torch.__version__}, cuda={torch.version.cuda}, cuda_available={torch.cuda.is_available()}")
except Exception:
    pass
print("Dependency check passed.")
PY

echo "Environment setup finished."
