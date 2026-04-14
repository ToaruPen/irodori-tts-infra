from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping


@dataclass(frozen=True, slots=True)
class CharacterVoice:
    name: str
    caption: str


@dataclass(frozen=True, slots=True)
class VoiceProfile:
    characters: Mapping[str, CharacterVoice]
    narrator_caption: str
    generic_dialogue_caption: str
