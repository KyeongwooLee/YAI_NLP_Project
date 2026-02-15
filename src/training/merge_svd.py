from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def _load_adapter_config(adapter_dir: Path) -> dict:
    config_path = adapter_dir / "adapter_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing adapter_config.json in {adapter_dir}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def _load_adapter_state(adapter_dir: Path) -> tuple[dict[str, torch.Tensor], str]:
    safe_path = adapter_dir / "adapter_model.safetensors"
    if safe_path.exists():
        return load_file(str(safe_path)), "safetensors"
    bin_path = adapter_dir / "adapter_model.bin"
    if bin_path.exists():
        return torch.load(bin_path, map_location="cpu"), "bin"
    raise FileNotFoundError(f"No adapter weights found in {adapter_dir}")


def _save_adapter_state(
    state: dict[str, torch.Tensor], output_dir: Path, storage: str
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if storage == "safetensors":
        save_file(state, str(output_dir / "adapter_model.safetensors"))
    else:
        torch.save(state, output_dir / "adapter_model.bin")


def _svd_merge_single(
    lora_a_1: torch.Tensor,
    lora_b_1: torch.Tensor,
    scale_1: float,
    lora_a_2: torch.Tensor,
    lora_b_2: torch.Tensor,
    scale_2: float,
    target_rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    delta = scale_1 * (lora_b_1.float() @ lora_a_1.float()) + scale_2 * (
        lora_b_2.float() @ lora_a_2.float()
    )
    u_mat, singular_values, vh_mat = torch.linalg.svd(delta, full_matrices=False)
    rank = max(1, min(target_rank, singular_values.numel()))
    singular_sqrt = torch.sqrt(singular_values[:rank])
    merged_b = u_mat[:, :rank] * singular_sqrt.unsqueeze(0)
    merged_a = singular_sqrt.unsqueeze(1) * vh_mat[:rank, :]
    return merged_a, merged_b


def merge_adapters_svd(
    adapter_a_dir: Path,
    adapter_b_dir: Path,
    output_dir: Path,
    target_rank: int = 8,
) -> Path:
    config_a = _load_adapter_config(adapter_a_dir)
    config_b = _load_adapter_config(adapter_b_dir)
    state_a, storage_type = _load_adapter_state(adapter_a_dir)
    state_b, _ = _load_adapter_state(adapter_b_dir)

    scale_a = float(config_a.get("lora_alpha", config_a.get("r", target_rank))) / float(
        config_a.get("r", target_rank)
    )
    scale_b = float(config_b.get("lora_alpha", config_b.get("r", target_rank))) / float(
        config_b.get("r", target_rank)
    )

    merged_state: dict[str, torch.Tensor] = {}
    for key, tensor_a in state_a.items():
        if "lora_A" not in key:
            if key in state_b and tensor_a.shape == state_b[key].shape:
                merged_state[key] = 0.5 * (tensor_a + state_b[key])
            else:
                merged_state[key] = tensor_a
            continue

        key_b = key.replace("lora_A", "lora_B")
        if key_b not in state_a or key not in state_b or key_b not in state_b:
            continue
        merged_a, merged_b = _svd_merge_single(
            lora_a_1=tensor_a,
            lora_b_1=state_a[key_b],
            scale_1=scale_a,
            lora_a_2=state_b[key],
            lora_b_2=state_b[key_b],
            scale_2=scale_b,
            target_rank=target_rank,
        )
        merged_state[key] = merged_a.to(tensor_a.dtype)
        merged_state[key_b] = merged_b.to(state_a[key_b].dtype)

    output_dir.mkdir(parents=True, exist_ok=True)
    new_config = dict(config_a)
    new_config["r"] = target_rank
    new_config["lora_alpha"] = target_rank
    (output_dir / "adapter_config.json").write_text(
        json.dumps(new_config, indent=2), encoding="utf-8"
    )
    _save_adapter_state(merged_state, output_dir, storage=storage_type)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="SVD merge two LoRA adapters.")
    parser.add_argument("--adapter-a", type=Path, required=True)
    parser.add_argument("--adapter-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-rank", type=int, default=8)
    args = parser.parse_args()
    merge_adapters_svd(
        adapter_a_dir=args.adapter_a,
        adapter_b_dir=args.adapter_b,
        output_dir=args.output,
        target_rank=args.target_rank,
    )


if __name__ == "__main__":
    main()
