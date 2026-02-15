from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence


@dataclass
class ConversationTurn:
    role: str
    text: str


@dataclass
class DialogueRecord:
    topic: str
    student_preferences: str
    teacher_preferences: str
    turns: list[ConversationTurn]


@dataclass
class TrainingExample:
    topic: str
    domain: str
    persona: str
    prompt: str
    response: str
    student_preference: str = ""
    teacher_preference: str = ""
    teacher_style: str = ""


def _normalize_role(role: str) -> str:
    normalized = role.strip().lower()
    if normalized.startswith("teach"):
        return "Teacher"
    if normalized.startswith("stud"):
        return "Student"
    return role.strip().title()


def load_dialogue_file(path: Path) -> list[DialogueRecord]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    records: list[DialogueRecord] = []
    for item in payload:
        background_info = item.get("background_info", {})
        topic = str(background_info.get("topic", "")).strip() or "Unknown Topic"
        student_preferences = str(
            background_info.get("student_prefrences", background_info.get("student_preferences", ""))
        ).strip()
        teacher_preferences = str(
            background_info.get("teacher_prefrences", background_info.get("teacher_preferences", ""))
        ).strip()

        turns: list[ConversationTurn] = []
        for turn in item.get("conversation", []):
            role = _normalize_role(str(turn.get("role", "")))
            text = str(turn.get("text", "")).strip()
            if not text:
                continue
            turns.append(ConversationTurn(role=role, text=text))

        if turns:
            records.append(
                DialogueRecord(
                    topic=topic,
                    student_preferences=student_preferences,
                    teacher_preferences=teacher_preferences,
                    turns=turns,
                )
            )

    return records


def load_dialogue_corpus(data_dir: Path, split: str = "train") -> list[DialogueRecord]:
    if split == "train":
        files = sorted(data_dir.glob("conversations_train*.json"))
    elif split == "eval":
        files = [data_dir / "conversations_eval.json"]
    else:
        raise ValueError(f"Unsupported split: {split}")

    corpus: list[DialogueRecord] = []
    for path in files:
        if path.exists():
            corpus.extend(load_dialogue_file(path))
    return corpus


def iter_student_teacher_pairs(records: Sequence[DialogueRecord]) -> Iterator[TrainingExample]:
    for record in records:
        history: list[str] = []
        for index, turn in enumerate(record.turns):
            history.append(f"{turn.role}: {turn.text}")
            if turn.role != "Teacher":
                continue
            if index == 0:
                continue
            previous_turn = record.turns[index - 1]
            if previous_turn.role != "Student":
                continue

            context_window = history[:-1][-6:]
            preference_header = [
                f"Student preference: {record.student_preferences}".strip(),
                f"Teacher preference: {record.teacher_preferences}".strip(),
            ]
            prompt = "\n".join(preference_header + context_window)
            response = turn.text
            yield TrainingExample(
                topic=record.topic,
                domain="general",
                persona=record.student_preferences,
                prompt=prompt,
                response=response,
                student_preference=record.student_preferences,
                teacher_preference=record.teacher_preferences,
            )
