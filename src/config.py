from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "Education-Dialogue-Dataset-main"
DEFAULT_ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DEFAULT_ADAPTERS_DIR = DEFAULT_ARTIFACTS_DIR / "adapters"
DEFAULT_LOGS_DIR = DEFAULT_ARTIFACTS_DIR / "logs"
DEFAULT_REPORTS_DIR = DEFAULT_ARTIFACTS_DIR / "reports"
LORA_HUB_DOMAIN_DIR_ENV = os.getenv("LORA_HUB_DOMAIN_DIR")


@dataclass
class ProjectConfig:
    base_model_name: str = os.getenv(
        "BASE_MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct"
    )
    local_files_only: bool = os.getenv("LOCAL_FILES_ONLY", "1") == "1"
    use_4bit: bool = os.getenv("USE_4BIT", "0") == "1"
    train_batch_size: int = int(os.getenv("TRAIN_BATCH_SIZE", "1"))
    eval_batch_size: int = int(os.getenv("EVAL_BATCH_SIZE", "1"))
    gradient_accumulation_steps: int = int(os.getenv("GRAD_ACC_STEPS", "4"))
    learning_rate: float = float(os.getenv("LEARNING_RATE", "2e-4"))
    num_train_epochs: int = float(os.getenv("NUM_TRAIN_EPOCHS", "1"))
    warmup_ratio: float = float(os.getenv("WARMUP_RATIO", "0.03"))
    max_seq_length: int = int(os.getenv("MAX_SEQ_LENGTH", "512"))
    logging_steps: int = int(os.getenv("LOGGING_STEPS", "10"))
    save_steps: int = int(os.getenv("SAVE_STEPS", "100"))
    seed: int = int(os.getenv("SEED", "42"))
    train_max_examples: int = int(os.getenv("TRAIN_MAX_EXAMPLES", "1200"))
    eval_max_examples: int = int(os.getenv("EVAL_MAX_EXAMPLES", "200"))
    continual_similarity_threshold: float = float(
        os.getenv("CONTINUAL_SIM_THRESHOLD", "0.35")
    )
    merge_adapter_threshold: int = int(os.getenv("MERGE_ADAPTER_THRESHOLD", "5"))
    merge_target_rank: int = int(os.getenv("MERGE_TARGET_RANK", "8"))

    data_dir: Path = Path(os.getenv("DATA_DIR", str(DEFAULT_DATA_DIR)))
    artifacts_dir: Path = Path(os.getenv("ARTIFACTS_DIR", str(DEFAULT_ARTIFACTS_DIR)))
    adapters_dir: Path = Path(os.getenv("ADAPTERS_DIR", str(DEFAULT_ADAPTERS_DIR)))
    logs_dir: Path = Path(os.getenv("LOGS_DIR", str(DEFAULT_LOGS_DIR)))
    reports_dir: Path = Path(os.getenv("REPORTS_DIR", str(DEFAULT_REPORTS_DIR)))

    domain_adapters_dir: Path = Path(
        os.getenv("DOMAIN_ADAPTERS_DIR", str(DEFAULT_ADAPTERS_DIR / "domain"))
    )
    persona_adapters_dir: Path = Path(
        os.getenv("PERSONA_ADAPTERS_DIR", str(DEFAULT_ADAPTERS_DIR / "persona"))
    )
    router_dir: Path = Path(os.getenv("ROUTER_DIR", str(DEFAULT_ARTIFACTS_DIR / "router")))
    lora_hub_domain_dir: Path | None = (
        Path(LORA_HUB_DOMAIN_DIR_ENV) if LORA_HUB_DOMAIN_DIR_ENV else None
    )

    def __post_init__(self) -> None:
        if "ADAPTERS_DIR" not in os.environ:
            self.adapters_dir = self.artifacts_dir / "adapters"
        if "LOGS_DIR" not in os.environ:
            self.logs_dir = self.artifacts_dir / "logs"
        if "REPORTS_DIR" not in os.environ:
            self.reports_dir = self.artifacts_dir / "reports"
        if "DOMAIN_ADAPTERS_DIR" not in os.environ:
            self.domain_adapters_dir = self.adapters_dir / "domain"
        if "PERSONA_ADAPTERS_DIR" not in os.environ:
            self.persona_adapters_dir = self.adapters_dir / "persona"
        if "ROUTER_DIR" not in os.environ:
            self.router_dir = self.artifacts_dir / "router"
        if self.local_files_only:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def ensure_project_dirs(config: ProjectConfig) -> None:
    directories = [
        config.artifacts_dir,
        config.adapters_dir,
        config.logs_dir,
        config.reports_dir,
        config.domain_adapters_dir,
        config.persona_adapters_dir,
        config.router_dir,
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
