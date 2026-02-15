from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def _load_state(adapter_dir: Path) -> tuple[dict[str, torch.Tensor], bool]:
    safe_path = adapter_dir / "adapter_model.safetensors"
    bin_path = adapter_dir / "adapter_model.bin"
    if safe_path.exists():
        return load_file(str(safe_path)), True
    if bin_path.exists():
        return torch.load(bin_path, map_location="cpu"), False
    raise FileNotFoundError(f"No adapter weights found in {adapter_dir}")


def _save_state(
    adapter_dir: Path,
    state_dict: dict[str, torch.Tensor],
    as_safetensors: bool,
) -> None:
    if as_safetensors:
        save_file(state_dict, str(adapter_dir / "adapter_model.safetensors"))
    else:
        torch.save(state_dict, adapter_dir / "adapter_model.bin")


def apply_orthogonal_projection(adapter_dir: Path) -> None:
    state_dict, as_safetensors = _load_state(adapter_dir)
    updated = {}
    for key, value in state_dict.items():
        if "lora_A" in key and value.ndim == 2:
            matrix = value.float()
            q_matrix, _ = torch.linalg.qr(matrix, mode="reduced")
            projected = q_matrix.to(value.dtype)
            if projected.shape != value.shape:
                projected = value
            updated[key] = projected.contiguous()
        else:
            updated[key] = value.contiguous()
    _save_state(adapter_dir, updated, as_safetensors)
