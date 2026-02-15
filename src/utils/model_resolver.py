from __future__ import annotations

from pathlib import Path

from src.config import ProjectConfig


def resolve_base_model_name(config: ProjectConfig) -> str:
    candidate = Path(config.base_model_name)
    if candidate.exists():
        return str(candidate)

    try:
        from huggingface_hub import snapshot_download

        local_snapshot = snapshot_download(
            repo_id=config.base_model_name,
            local_files_only=config.local_files_only,
        )
        return str(local_snapshot)
    except Exception:
        return config.base_model_name
