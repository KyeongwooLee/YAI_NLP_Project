from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from src.config import ProjectConfig, ensure_project_dirs
from src.data.loader import iter_student_teacher_pairs, load_dialogue_corpus
from src.data.preprocess import attach_domain_and_persona
from src.inference.generate import run_inference


def _tokenize_for_overlap(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-zA-Z0-9]+", text.lower()) if len(token) > 2}


def run_correctness_eval(
    config: ProjectConfig,
    sample_size: int = 8,
) -> dict:
    ensure_project_dirs(config)
    records = load_dialogue_corpus(config.data_dir, split="eval")
    examples = attach_domain_and_persona(list(iter_student_teacher_pairs(records)))
    selected = examples[:sample_size]
    if not selected:
        result = {"status": "skipped", "reason": "No eval examples"}
        (config.reports_dir / "correctness_eval.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        return result

    scores = []
    for example in selected:
        inference = run_inference(example.prompt, config=config, max_new_tokens=120)
        prediction_tokens = _tokenize_for_overlap(inference["answer"])
        target_tokens = _tokenize_for_overlap(example.response)
        overlap = len(prediction_tokens & target_tokens)
        denom = max(1, len(target_tokens))
        score = overlap / denom
        scores.append(score)

    result = {
        "sample_size": len(selected),
        "average_keyword_overlap": sum(scores) / max(1, len(scores)),
        "scores": scores,
    }
    (config.reports_dir / "correctness_eval.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run correctness evaluation.")
    parser.add_argument("--sample-size", type=int, default=8)
    args = parser.parse_args()
    config = ProjectConfig()
    result = run_correctness_eval(config=config, sample_size=args.sample_size)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
