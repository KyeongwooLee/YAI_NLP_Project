from __future__ import annotations

import math
import pickle
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from src.data.loader import TrainingExample
from src.data.preprocess import normalize_persona_label


@dataclass
class RouterV2Decision:
    domain: str
    persona: str
    domain_weight: float
    persona_weight: float
    domain_probability: float
    persona_probability: float


def _tokenize(text: str) -> list[str]:
    return [token for token in re.findall(r"[a-zA-Z0-9]+", text.lower()) if len(token) > 2]


class _MultinomialNB:
    def __init__(self) -> None:
        self.class_token_counts: dict[str, Counter[str]] = {}
        self.class_doc_counts: dict[str, int] = {}
        self.vocab: set[str] = set()
        self.total_docs = 0

    def fit(self, texts: list[str], labels: list[str]) -> None:
        class_token_counts: dict[str, Counter[str]] = defaultdict(Counter)
        class_doc_counts: dict[str, int] = defaultdict(int)
        for text, label in zip(texts, labels):
            tokens = _tokenize(text)
            class_doc_counts[label] += 1
            class_token_counts[label].update(tokens)
            self.vocab.update(tokens)

        self.class_token_counts = dict(class_token_counts)
        self.class_doc_counts = dict(class_doc_counts)
        self.total_docs = len(texts)

    def predict_proba(self, text: str) -> dict[str, float]:
        tokens = _tokenize(text)
        vocab_size = max(1, len(self.vocab))
        log_probs: dict[str, float] = {}

        for class_name, token_counts in self.class_token_counts.items():
            class_docs = self.class_doc_counts[class_name]
            prior = class_docs / max(1, self.total_docs)
            total_tokens = sum(token_counts.values())
            log_prob = math.log(prior + 1e-12)
            for token in tokens:
                token_count = token_counts.get(token, 0)
                likelihood = (token_count + 1.0) / (total_tokens + vocab_size)
                log_prob += math.log(likelihood)
            log_probs[class_name] = log_prob

        max_log = max(log_probs.values())
        exp_scores = {key: math.exp(value - max_log) for key, value in log_probs.items()}
        normalizer = sum(exp_scores.values())
        return {key: value / max(1e-12, normalizer) for key, value in exp_scores.items()}


class RouterV2:
    def __init__(self) -> None:
        self.domain_model = _MultinomialNB()
        self.persona_model = _MultinomialNB()
        self.default_persona = "direct_instruction"
        self._is_fitted = False

    def fit(self, examples: list[TrainingExample]) -> None:
        if not examples:
            raise ValueError("RouterV2 requires at least one example.")
        texts = [
            (
                f"{example.topic} {example.prompt} "
                f"student_pref={example.student_preference} "
                f"teacher_pref={example.teacher_preference} "
                f"teacher_style={example.teacher_style}"
            )
            for example in examples
        ]
        domains = [example.domain for example in examples]
        personas = [example.persona for example in examples]

        self.domain_model.fit(texts, domains)
        self.persona_model.fit(texts, personas)
        self.default_persona = Counter(personas).most_common(1)[0][0]
        self._is_fitted = True

    def route(
        self,
        query: str,
        student_preference_hint: str = "",
        teacher_preference_hint: str = "",
    ) -> RouterV2Decision:
        if not self._is_fitted:
            raise RuntimeError("RouterV2 is not fitted.")

        domain_probs = self.domain_model.predict_proba(query)
        domain = max(domain_probs, key=domain_probs.get)
        domain_probability = float(domain_probs[domain])

        persona_probs = self.persona_model.predict_proba(query)
        persona = max(persona_probs, key=persona_probs.get) if persona_probs else self.default_persona
        hinted_persona = normalize_persona_label(
            student_preference_hint,
            teacher_preference_hint,
        )
        if student_preference_hint or teacher_preference_hint:
            persona = hinted_persona
        persona_probability = float(persona_probs.get(persona, 1.0))

        persona_weight = min(0.5, max(0.2, 1.0 - domain_probability))
        domain_weight = 1.0 - persona_weight
        return RouterV2Decision(
            domain=domain,
            persona=persona,
            domain_weight=domain_weight,
            persona_weight=persona_weight,
            domain_probability=domain_probability,
            persona_probability=persona_probability,
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load(path: Path) -> "RouterV2":
        with path.open("rb") as file:
            return pickle.load(file)
