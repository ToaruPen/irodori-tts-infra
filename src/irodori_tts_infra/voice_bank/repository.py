from __future__ import annotations

from typing import TYPE_CHECKING

from irodori_tts_infra.voice_bank.captions import (
    DEFAULT_GENERIC_DIALOGUE_CAPTION,
    DEFAULT_NARRATOR_CAPTION,
    load_characters_markdown,
)
from irodori_tts_infra.voice_bank.models import VoiceProfile

if TYPE_CHECKING:
    from pathlib import Path


def find_characters_markdown(turn_file: Path) -> Path | None:
    current = turn_file.parent
    while current != current.parent:
        candidate = current / "characters.md"
        if candidate.is_file():
            return candidate
        if current.name == "chat":
            break
        current = current.parent
    return None


def load_voice_profile(
    characters_md: Path | None,
    *,
    narrator_caption: str = DEFAULT_NARRATOR_CAPTION,
    generic_dialogue_caption: str = DEFAULT_GENERIC_DIALOGUE_CAPTION,
) -> VoiceProfile:
    characters = {}
    if characters_md is not None and characters_md.is_file():
        characters = load_characters_markdown(characters_md.read_text(encoding="utf-8"))

    return VoiceProfile(
        characters=characters,
        narrator_caption=narrator_caption,
        generic_dialogue_caption=generic_dialogue_caption,
    )
