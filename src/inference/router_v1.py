from __future__ import annotations

import math
import pickle
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from src.data.loader import TrainingExample
from src.data.preprocess import DOMAIN_KEYWORDS


@dataclass
class RouteDecision:
    domain: str
    persona: str
    domain_weight: float
    persona_weight: float
    scores: dict[str, float]


def _tokenize(text: str) -> list[str]:
    return [token for token in re.findall(r"[a-zA-Z0-9]+", text.lower()) if len(token) > 2]


def _cosine(counter_a: Counter[str], counter_b: Counter[str]) -> float:
    if not counter_a or not counter_b:
        return 0.0
    common = set(counter_a) & set(counter_b)
    numerator = sum(counter_a[token] * counter_b[token] for token in common)
    denominator_a = math.sqrt(sum(value * value for value in counter_a.values()))
    denominator_b = math.sqrt(sum(value * value for value in counter_b.values()))
    denominator = denominator_a * denominator_b
    if denominator == 0:
        return 0.0
    return numerator / denominator


class RouterV1:
    def __init__(self) -> None:
        self.domain_profiles: dict[str, Counter[str]] = {}
        self.persona_counts: dict[str, int] = {}
        self.is_fitted = False

    def fit(self, examples: list[TrainingExample]) -> None:
        if not examples:
            raise ValueError("RouterV1 requires at least one example.")
        profile_builder: dict[str, Counter[str]] = defaultdict(Counter)
        for example in examples:
            text = f"{example.topic} {example.prompt} {example.response}"
            profile_builder[example.domain].update(_tokenize(text))
            self.persona_counts[example.persona] = self.persona_counts.get(example.persona, 0) + 1
        self.domain_profiles = dict(profile_builder)
        self.is_fitted = True

    def _keyword_scores(self, query: str) -> dict[str, float]:
        lowered = query.lower()
        scores: dict[str, float] = {}
        for domain, keywords in DOMAIN_KEYWORDS.items():
            scores[domain] = float(sum(1 for keyword in keywords if keyword in lowered))
        scores["general"] = 0.1
        return scores

    def route(self, query: str, persona_hint: str = "neutral") -> RouteDecision:
        if not self.is_fitted:
            raise RuntimeError("RouterV1 is not fitted.")
        query_counter = Counter(_tokenize(query))
        keyword_scores = self._keyword_scores(query)
        score_map: dict[str, float] = {}
        for domain, profile in self.domain_profiles.items():
            semantic = _cosine(query_counter, profile)
            keyword = keyword_scores.get(domain, 0.0)
            score_map[domain] = 0.75 * semantic + 0.25 * keyword

        chosen_domain = max(score_map, key=score_map.get)
        persona_weight = 0.35 if persona_hint == "neutral" else 0.45
        domain_weight = 1.0 - persona_weight
        return RouteDecision(
            domain=chosen_domain,
            persona=persona_hint,
            domain_weight=domain_weight,
            persona_weight=persona_weight,
            scores=score_map,
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load(path: Path) -> "RouterV1":
        with path.open("rb") as file:
            return pickle.load(file)
