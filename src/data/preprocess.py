from __future__ import annotations

import random
import re
from collections import defaultdict
from typing import Iterable

from src.data.loader import TrainingExample


DOMAIN_KEYWORDS: dict[str, tuple[str, ...]] = {
    "math": ("algebra", "equation", "geometry", "calculus", "fraction", "number"),
    "science": ("physics", "chemistry", "biology", "experiment", "force", "energy"),
    "language": ("grammar", "essay", "writing", "reading", "vocabulary", "literature"),
    "history": ("war", "empire", "king", "revolution", "century", "historical"),
}


def detect_domain(topic: str, text: str) -> str:
    combined = f"{topic} {text}".lower()
    scores: dict[str, int] = {}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        scores[domain] = sum(1 for keyword in keywords if keyword in combined)
    best_domain, best_score = max(scores.items(), key=lambda item: item[1])
    return best_domain if best_score > 0 else "general"


def normalize_persona_label(student_preferences: str) -> str:
    value = student_preferences.lower().strip()
    if not value:
        return "neutral"
    if "creative" in value or "story" in value or "gamification" in value:
        return "creative_gamified"
    if "direct" in value or "lecture" in value:
        return "direct_instruction"
    if "hands-on" in value or "experiential" in value:
        return "hands_on"
    return re.sub(r"[^a-z0-9_]+", "_", value)[:40] or "neutral"


def attach_domain_and_persona(
    examples: Iterable[TrainingExample],
) -> list[TrainingExample]:
    enriched: list[TrainingExample] = []
    for example in examples:
        domain = detect_domain(example.topic, f"{example.prompt}\n{example.response}")
        persona = normalize_persona_label(example.persona)
        enriched.append(
            TrainingExample(
                topic=example.topic,
                domain=domain,
                persona=persona,
                prompt=example.prompt,
                response=example.response,
            )
        )
    return enriched


def filter_examples_by_domain(
    examples: list[TrainingExample], domain: str, max_examples: int
) -> list[TrainingExample]:
    selected = [example for example in examples if example.domain == domain]
    if not selected:
        selected = examples
    return selected[:max_examples]


def filter_examples_by_persona(
    examples: list[TrainingExample], persona: str, max_examples: int
) -> list[TrainingExample]:
    selected = [example for example in examples if example.persona == persona]
    if not selected:
        selected = examples
    return selected[:max_examples]


def split_train_eval(
    examples: list[TrainingExample], eval_ratio: float = 0.1, seed: int = 42
) -> tuple[list[TrainingExample], list[TrainingExample]]:
    if len(examples) <= 1:
        return examples, examples
    random.Random(seed).shuffle(examples)
    eval_size = max(1, int(len(examples) * eval_ratio))
    eval_size = min(eval_size, len(examples) - 1)
    return examples[eval_size:], examples[:eval_size]


def summarize_label_distribution(examples: list[TrainingExample]) -> dict[str, dict[str, int]]:
    domain_counts: dict[str, int] = defaultdict(int)
    persona_counts: dict[str, int] = defaultdict(int)
    for example in examples:
        domain_counts[example.domain] += 1
        persona_counts[example.persona] += 1
    return {"domain": dict(domain_counts), "persona": dict(persona_counts)}
