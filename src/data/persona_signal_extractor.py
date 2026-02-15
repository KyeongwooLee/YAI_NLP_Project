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
    "hands_on": "Explain through practical examples and small actionable exercises.",
    "neutral": "Explain clearly with balanced detail and one quick example.",
}


def build_persona_profile(raw_preference: str) -> PersonaProfile:
    label = normalize_persona_label(raw_preference)
    instruction = PERSONA_STYLE_MAP.get(label, PERSONA_STYLE_MAP["neutral"])
    return PersonaProfile(label=label, style_instruction=instruction)
