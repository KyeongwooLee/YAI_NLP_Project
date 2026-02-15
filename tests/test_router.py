from __future__ import annotations

from src.config import DEFAULT_DATA_DIR
from src.data.loader import iter_student_teacher_pairs, load_dialogue_corpus
from src.data.preprocess import attach_domain_and_persona
from src.inference.router_v1 import RouterV1
from src.inference.router_v2 import RouterV2


def _sample_examples(limit: int = 100):
    records = load_dialogue_corpus(DEFAULT_DATA_DIR, split="train")[:10]
    examples = attach_domain_and_persona(list(iter_student_teacher_pairs(records)))
    return examples[:limit]


def test_router_v1_routes_query() -> None:
    examples = _sample_examples()
    router = RouterV1()
    router.fit(examples)
    decision = router.route("Can you explain this physics force equation?")
    assert decision.domain
    assert 0 <= decision.domain_weight <= 1
    assert 0 <= decision.persona_weight <= 1


def test_router_v2_routes_query() -> None:
    examples = _sample_examples()
    router = RouterV2()
    router.fit(examples)
    decision = router.route("Can you explain this grammar rule in short steps?")
    assert decision.domain
    assert decision.domain_probability >= 0
