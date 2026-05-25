from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping


@dataclass(frozen=True, slots=True)
class SpeakerEmbeddingProfile:
    ref_embed: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "ref_embed", Path(self.ref_embed))


@dataclass(frozen=True, slots=True)
class CharacterVoice:
    name: str
    speaker: SpeakerEmbeddingProfile


@dataclass(frozen=True, slots=True)
class VoiceProfile:
    characters: Mapping[str, CharacterVoice]
    narrator: SpeakerEmbeddingProfile

    def __post_init__(self) -> None:
        object.__setattr__(self, "characters", MappingProxyType(dict(self.characters)))
