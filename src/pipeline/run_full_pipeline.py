from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

from src.config import ProjectConfig, ensure_project_dirs
from src.data.loader import iter_student_teacher_pairs, load_dialogue_corpus
from src.data.preprocess import PERSONA_LABELS, attach_domain_and_persona
from src.eval.build_report import build_report
from src.eval.eval_correctness import run_correctness_eval
from src.eval.eval_personalization import run_personalization_eval
from src.eval.eval_system import run_system_eval
from src.inference.generate import run_inference
from src.inference.router_v1 import RouterV1
from src.inference.router_v2 import RouterV2
from src.training.continual_update import run_continual_update
from src.training.train_domain_lora import train_domain_adapter
from src.training.train_persona_lora import train_persona_adapter
from src.utils.model_resolver import resolve_base_model_name

DOMAIN_BOOTSTRAP_ORDER: tuple[str, ...] = ("science", "math", "language")
DOMAIN_HUB_ALIASES: dict[str, tuple[str, ...]] = {
    "science": ("science", "sci"),
    "math": ("math", "mathematics"),
    "language": ("language", "lang", "english"),
}


def _validate_model_access(config: ProjectConfig) -> None:
    from transformers import AutoConfig

    try:
        AutoConfig.from_pretrained(
            config.base_model_name,
            local_files_only=config.local_files_only,
        )
    except Exception as error:
        raise RuntimeError(
            "Base model is not accessible. If you are offline, set BASE_MODEL_NAME to a local checkpoint path. "
            "If you want online download, set LOCAL_FILES_ONLY=0."
        ) from error


def _train_routers(config: ProjectConfig, examples) -> dict:
    router_v1 = RouterV1()
    router_v1.fit(examples)
    router_v1_path = config.router_dir / "router_v1.pkl"
    router_v1.save(router_v1_path)

    router_v2 = RouterV2()
    router_v2.fit(examples)
    router_v2_path = config.router_dir / "router_v2.pkl"
    router_v2.save(router_v2_path)
    return {"router_v1_path": str(router_v1_path), "router_v2_path": str(router_v2_path)}


def _select_domains(domain_counts: Counter[str], max_domains: int) -> list[str]:
    selected_domains = [
        domain
        for domain in DOMAIN_BOOTSTRAP_ORDER
        if domain_counts.get(domain, 0) > 0
    ][:max_domains]
    if len(selected_domains) < max_domains:
        for domain, _ in domain_counts.most_common():
            if domain in {"general", *selected_domains}:
                continue
            selected_domains.append(domain)
            if len(selected_domains) >= max_domains:
                break
    if not selected_domains and domain_counts:
        selected_domains = [domain_counts.most_common(1)[0][0]]
    return selected_domains


def _is_valid_adapter_dir(path: Path) -> bool:
    return path.is_dir() and (path / "adapter_config.json").exists()


def _seed_domain_adapter_from_hub(config: ProjectConfig, domain: str, target_dir: Path) -> str | None:
    if config.lora_hub_domain_dir is None:
        return None

    alias_candidates = DOMAIN_HUB_ALIASES.get(domain, (domain,))
    source_dir = None
    for alias in alias_candidates:
        candidate = config.lora_hub_domain_dir / alias
        if _is_valid_adapter_dir(candidate):
            source_dir = candidate
            break
    if source_dir is None:
        return None

    if _is_valid_adapter_dir(target_dir):
        return str(target_dir)

    target_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_dir, target_dir, dirs_exist_ok=True)
    return str(source_dir)


def run_full_pipeline(
    config: ProjectConfig,
    *,
    max_domains: int = 3,
    max_examples_per_adapter: int | None = None,
    eval_sample_size: int = 3,
    query: str = "Can you explain Archimedes principle with a simple example?",
) -> dict:
    ensure_project_dirs(config)
    config.base_model_name = resolve_base_model_name(config)
    _validate_model_access(config)
    train_records = load_dialogue_corpus(config.data_dir, split="train")
    train_examples = attach_domain_and_persona(list(iter_student_teacher_pairs(train_records)))
    if not train_examples:
        raise RuntimeError("No training examples were built from the dataset.")

    domain_counts = Counter(example.domain for example in train_examples)
    selected_domains = _select_domains(domain_counts, max_domains)

    persona_counts = Counter(example.persona for example in train_examples)
    selected_personas = [label for label in PERSONA_LABELS if persona_counts.get(label, 0) > 0]
    if not selected_personas and persona_counts:
        selected_personas = [persona_counts.most_common(1)[0][0]]

    domain_training_results = {}
    domain_seed_sources = {}
    for domain in selected_domains:
        adapter_dir = config.domain_adapters_dir / domain
        seed_source = _seed_domain_adapter_from_hub(config, domain, adapter_dir)
        if seed_source is not None:
            domain_seed_sources[domain] = seed_source
        resume_adapter_path = adapter_dir if _is_valid_adapter_dir(adapter_dir) else None
        domain_training_results[domain] = train_domain_adapter(
            domain=domain,
            config=config,
            output_dir=adapter_dir,
            max_examples=max_examples_per_adapter or config.train_max_examples,
            resume_adapter_path=resume_adapter_path,
        )

    persona_training_results = {}
    for persona_label in selected_personas:
        persona_training_results[persona_label] = train_persona_adapter(
            persona_label=persona_label,
            config=config,
            max_examples=max_examples_per_adapter or config.train_max_examples,
        )

    router_result = _train_routers(config, train_examples)
    continual_result = run_continual_update(
        config=config,
        max_examples=config.eval_max_examples,
    )
    inference_result = run_inference(query, config=config)
    correctness_result = run_correctness_eval(config=config, sample_size=eval_sample_size)
    personalization_result = run_personalization_eval(
        config=config,
        sample_size=eval_sample_size,
    )
    system_result = run_system_eval(config=config, query=query)
    report = build_report(config)

    summary = {
        "trained_domains": selected_domains,
        "trained_personas": selected_personas,
        "domain_training_results": domain_training_results,
        "domain_seed_sources": domain_seed_sources,
        "persona_training_results": persona_training_results,
        "router_result": router_result,
        "continual_result": continual_result,
        "inference_result": inference_result,
        "correctness_result": correctness_result,
        "personalization_result": personalization_result,
        "system_result": system_result,
        "report_path": str(config.reports_dir / "final_report.json"),
    }
    (config.logs_dir / "pipeline_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full Personal LLM Tutor pipeline.")
    parser.add_argument("--max-domains", type=int, default=3)
    parser.add_argument("--max-examples-per-adapter", type=int, default=None)
    parser.add_argument("--eval-sample-size", type=int, default=3)
    parser.add_argument(
        "--query",
        type=str,
        default="Can you explain Archimedes principle with a simple example?",
    )
    args = parser.parse_args()
    config = ProjectConfig()
    summary = run_full_pipeline(
        config=config,
        max_domains=args.max_domains,
        max_examples_per_adapter=args.max_examples_per_adapter,
        eval_sample_size=args.eval_sample_size,
        query=args.query,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
