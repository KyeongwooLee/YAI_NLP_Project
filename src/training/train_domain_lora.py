from __future__ import annotations

import argparse
from pathlib import Path

from src.config import ProjectConfig, ensure_project_dirs
from src.data.loader import iter_student_teacher_pairs, load_dialogue_corpus
from src.data.preprocess import (
    attach_domain_and_persona,
    filter_examples_by_domain,
    split_train_eval,
    summarize_label_distribution,
)
from src.training.common import dump_config, save_examples_metadata, train_lora_adapter
from src.utils.model_resolver import resolve_base_model_name


def train_domain_adapter(
    domain: str,
    config: ProjectConfig,
    output_dir: Path | None = None,
    max_examples: int | None = None,
) -> dict:
    ensure_project_dirs(config)
    config.base_model_name = resolve_base_model_name(config)
    records = load_dialogue_corpus(config.data_dir, split="train")
    examples = list(iter_student_teacher_pairs(records))
    examples = attach_domain_and_persona(examples)

    selected = filter_examples_by_domain(
        examples,
        domain=domain,
        max_examples=max_examples or config.train_max_examples,
    )
    train_examples, eval_examples = split_train_eval(selected, seed=config.seed)
    adapter_dir = output_dir or (config.domain_adapters_dir / domain)
    metrics = train_lora_adapter(
        config=config,
        adapter_output_dir=adapter_dir,
        train_examples=train_examples,
        eval_examples=eval_examples,
        rank=16,
        alpha=32,
        dropout=0.05,
    )
    save_examples_metadata(
        adapter_dir,
        domain=domain,
        extra={
            "distribution": summarize_label_distribution(selected),
            "selected_examples": len(selected),
        },
    )
    dump_config(config, adapter_dir / "config_snapshot.json")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a domain LoRA adapter.")
    parser.add_argument("--domain", type=str, required=True)
    parser.add_argument("--max-examples", type=int, default=None)
    args = parser.parse_args()
    config = ProjectConfig()
    train_domain_adapter(args.domain, config, max_examples=args.max_examples)


if __name__ == "__main__":
    main()
