from __future__ import annotations

from src.config import DEFAULT_DATA_DIR
from src.data.loader import iter_student_teacher_pairs, load_dialogue_corpus
from src.data.preprocess import attach_domain_and_persona


def test_load_train_split_not_empty() -> None:
    records = load_dialogue_corpus(DEFAULT_DATA_DIR, split="train")
    assert len(records) > 0
    assert all(record.topic for record in records[:5])


def test_pair_extraction_and_label_attachment() -> None:
    records = load_dialogue_corpus(DEFAULT_DATA_DIR, split="train")[:2]
    examples = list(iter_student_teacher_pairs(records))
    assert examples
    enriched = attach_domain_and_persona(examples)
    assert all(example.domain for example in enriched)
    assert all(example.persona for example in enriched)
