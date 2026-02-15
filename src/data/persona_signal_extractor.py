from __future__ import annotations

from dataclasses import dataclass

from src.data.preprocess import normalize_persona_label


@dataclass
class PersonaProfile:
    label: str
    style_instruction: str


PERSONA_STYLE_MAP: dict[str, str] = {
    "creative_gamified": "Explain with creative analogies, short stories, and optional game-like challenges.",
    "direct_instruction": "Explain directly in concise steps with clear definitions and minimal narrative.",
    "hands_on_applied": "Explain through practical examples, real-world applications, and short exercises.",
    "interactive_inquiry": "Explain with guided questions and interactive checkpoints before giving the final answer.",
}


def build_persona_profile(raw_student_preference: str, raw_teacher_preference: str = "") -> PersonaProfile:
    label = normalize_persona_label(raw_student_preference, raw_teacher_preference)
    instruction = PERSONA_STYLE_MAP.get(label, PERSONA_STYLE_MAP["direct_instruction"])
    return PersonaProfile(label=label, style_instruction=instruction)
