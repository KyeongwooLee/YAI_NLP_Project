from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import ProjectConfig, ensure_project_dirs
from src.data.loader import iter_student_teacher_pairs, load_dialogue_corpus
from src.data.preprocess import attach_domain_and_persona
from src.inference.generate import run_inference


PERSONA_RULES: dict[str, tuple[str, ...]] = {
    "creative_gamified": ("imagine", "story", "challenge", "game"),
    "direct_instruction": ("step", "definition", "first", "second"),
    "hands_on": ("try", "exercise", "practice", "example"),
    "neutral": ("explain", "example"),
}


def _persona_score(answer: str, persona: str) -> float:
    keywords = PERSONA_RULES.get(persona, PERSONA_RULES["neutral"])
    lowered = answer.lower()
    matches = sum(1 for keyword in keywords if keyword in lowered)
    return matches / max(1, len(keywords))


def run_personalization_eval(
    config: ProjectConfig,
    sample_size: int = 8,
) -> dict:
    ensure_project_dirs(config)
    records = load_dialogue_corpus(config.data_dir, split="eval")
    examples = attach_domain_and_persona(list(iter_student_teacher_pairs(records)))
    selected = examples[:sample_size]
    if not selected:
        result = {"status": "skipped", "reason": "No eval examples"}
        (config.reports_dir / "personalization_eval.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        return result

    scores = []
    for example in selected:
        inference = run_inference(example.prompt, config=config, max_new_tokens=120)
        score = _persona_score(inference["answer"], example.persona)
        scores.append(score)

    result = {
        "sample_size": len(selected),
        "average_persona_style_score": sum(scores) / max(1, len(scores)),
        "scores": scores,
    }
    (config.reports_dir / "personalization_eval.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run personalization evaluation.")
    parser.add_argument("--sample-size", type=int, default=8)
    args = parser.parse_args()
    config = ProjectConfig()
    result = run_personalization_eval(config=config, sample_size=args.sample_size)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
