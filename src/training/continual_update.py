from __future__ import annotations

import argparse
import json
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

from src.config import ProjectConfig, ensure_project_dirs
from src.data.loader import (
    iter_student_teacher_pairs,
    load_dialogue_corpus,
    load_dialogue_file,
)
from src.data.preprocess import (
    attach_domain_and_persona,
    filter_examples_by_domain,
    split_train_eval,
)
from src.training.anti_forgetting import apply_orthogonal_projection
from src.training.common import save_examples_metadata, train_lora_adapter
from src.training.merge_svd import merge_adapters_svd
from src.utils.model_resolver import resolve_base_model_name


def _pick_target_domain(
    dominant_domain: str,
    existing_domains: list[str],
    similarity_threshold: float,
) -> tuple[str, bool]:
    if dominant_domain in existing_domains:
        return dominant_domain, True
    if not existing_domains:
        return dominant_domain, False

    best_domain = max(
        existing_domains,
        key=lambda name: SequenceMatcher(None, dominant_domain, name).ratio(),
    )
    best_score = SequenceMatcher(None, dominant_domain, best_domain).ratio()
    if best_score >= similarity_threshold:
        return best_domain, True
    return dominant_domain, False


def _load_incremental_records(config: ProjectConfig, incremental_data_path: Path | None):
    if incremental_data_path is not None:
        return load_dialogue_file(incremental_data_path)
    return load_dialogue_corpus(config.data_dir, split="eval")


def run_continual_update(
    config: ProjectConfig,
    *,
    incremental_data_path: Path | None = None,
    max_examples: int | None = None,
) -> dict:
    ensure_project_dirs(config)
    config.base_model_name = resolve_base_model_name(config)
    records = _load_incremental_records(config, incremental_data_path)
    examples = attach_domain_and_persona(list(iter_student_teacher_pairs(records)))
    if not examples:
        return {"status": "skipped", "reason": "No incremental examples available."}

    domain_counter = Counter(example.domain for example in examples)
    dominant_domain, _ = domain_counter.most_common(1)[0]
    existing_domain_dirs = [path for path in config.domain_adapters_dir.iterdir() if path.is_dir()]
    existing_domains = [path.name for path in existing_domain_dirs]
    target_domain, should_update = _pick_target_domain(
        dominant_domain=dominant_domain,
        existing_domains=existing_domains,
        similarity_threshold=config.continual_similarity_threshold,
    )

    domain_examples = filter_examples_by_domain(
        examples,
        domain=dominant_domain,
        max_examples=max_examples or config.eval_max_examples,
    )
    train_examples, eval_examples = split_train_eval(domain_examples, seed=config.seed)
    target_dir = config.domain_adapters_dir / target_domain
    resume_path = target_dir if should_update and target_dir.exists() else None
    metrics = train_lora_adapter(
        config=config,
        adapter_output_dir=target_dir,
        train_examples=train_examples,
        eval_examples=eval_examples,
        rank=16,
        alpha=32,
        dropout=0.05,
        resume_adapter_path=resume_path,
    )
    apply_orthogonal_projection(target_dir)
    save_examples_metadata(
        target_dir,
        domain=target_domain,
        extra={
            "dominant_domain": dominant_domain,
            "update_mode": "update" if should_update else "new_adapter",
            "new_example_count": len(domain_examples),
        },
    )

    all_adapters = sorted([path for path in config.domain_adapters_dir.iterdir() if path.is_dir()])
    merge_info: dict[str, str] | None = None
    if len(all_adapters) >= config.merge_adapter_threshold:
        adapter_a, adapter_b = all_adapters[0], all_adapters[1]
        merged_name = f"merged_{adapter_a.name}_{adapter_b.name}"
        merged_dir = config.domain_adapters_dir / merged_name
        merge_adapters_svd(
            adapter_a_dir=adapter_a,
            adapter_b_dir=adapter_b,
            output_dir=merged_dir,
            target_rank=config.merge_target_rank,
        )
        merge_info = {
            "adapter_a": adapter_a.name,
            "adapter_b": adapter_b.name,
            "merged": merged_name,
        }

    summary = {
        "status": "ok",
        "dominant_domain": dominant_domain,
        "target_domain": target_domain,
        "mode": "update" if should_update else "new_adapter",
        "metrics": metrics,
        "merge_info": merge_info,
    }
    (config.logs_dir / "continual_update_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run continual adapter update pipeline.")
    parser.add_argument("--incremental-data", type=Path, default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    args = parser.parse_args()
    config = ProjectConfig()
    run_continual_update(
        config,
        incremental_data_path=args.incremental_data,
        max_examples=args.max_examples,
    )


if __name__ == "__main__":
    main()
