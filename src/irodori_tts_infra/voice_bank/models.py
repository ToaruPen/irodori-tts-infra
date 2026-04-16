from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping


@dataclass(frozen=True, slots=True)
class RVCProfile:
    model_path: Path
    sample_rate: int
    neutral_prototype: Path | None = None
    state_prototypes: Mapping[str, Path] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_path", Path(self.model_path))
        if self.neutral_prototype is not None:
            object.__setattr__(self, "neutral_prototype", Path(self.neutral_prototype))
        if not isinstance(self.sample_rate, int) or isinstance(self.sample_rate, bool):
            msg = "sample_rate must be an integer"
            raise TypeError(msg)
        if self.sample_rate <= 0:
            msg = "sample_rate must be greater than 0"
            raise ValueError(msg)
        state_prototypes = {
            state: Path(prototype) for state, prototype in self.state_prototypes.items()
        }
        object.__setattr__(
            self,
            "state_prototypes",
            MappingProxyType(state_prototypes),
        )


@dataclass(frozen=True, slots=True)
class CharacterVoice:
    name: str
    caption: str
    rvc: RVCProfile | None = None


@dataclass(frozen=True, slots=True)
class VoiceProfile:
    characters: Mapping[str, CharacterVoice]
    narrator_caption: str
    generic_dialogue_caption: str
