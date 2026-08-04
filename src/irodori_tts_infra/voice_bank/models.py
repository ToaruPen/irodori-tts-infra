from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, cast

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
class PortableVoice:
    id: str
    label: str
    aliases: tuple[str, ...]
    default: bool
    speaker: SpeakerEmbeddingProfile

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            msg = "portable voice id must be a non-blank string"
            raise ValueError(msg)
        if not isinstance(self.label, str) or not self.label.strip():
            msg = "portable voice label must be a non-blank string"
            raise ValueError(msg)
        raw_aliases = cast("object", self.aliases)
        if not isinstance(raw_aliases, (list, tuple)):
            msg = "portable voice aliases must be a sequence"
            raise TypeError(msg)
        normalized_aliases: list[str] = []
        for alias in raw_aliases:
            if not isinstance(alias, str) or not alias.strip():
                msg = "portable voice aliases must contain non-blank strings"
                raise ValueError(msg)
            stripped = alias.strip()
            if stripped in normalized_aliases:
                msg = "portable voice aliases must be unique"
                raise ValueError(msg)
            normalized_aliases.append(stripped)
        raw_default = cast("object", self.default)
        if not isinstance(raw_default, bool):
            msg = "portable voice default must be a boolean"
            raise TypeError(msg)
        raw_speaker = cast("object", self.speaker)
        if not isinstance(raw_speaker, SpeakerEmbeddingProfile):
            msg = "portable voice speaker must be a SpeakerEmbeddingProfile"
            raise TypeError(msg)
        object.__setattr__(self, "id", self.id.strip())
        object.__setattr__(self, "label", self.label.strip())
        object.__setattr__(self, "aliases", tuple(normalized_aliases))


@dataclass(frozen=True, slots=True)
class VoiceProfile:
    characters: Mapping[str, CharacterVoice]
    narrator: SpeakerEmbeddingProfile
    catalog: tuple[PortableVoice, ...] = ()
    _catalog_lookup: Mapping[str, PortableVoice] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "characters", MappingProxyType(dict(self.characters)))
        catalog = tuple(self.catalog)
        if sum(voice.default for voice in catalog) > 1:
            msg = "voice catalog must contain at most one default"
            raise ValueError(msg)
        lookup: dict[str, PortableVoice] = {}
        for voice in catalog:
            if voice.id in lookup:
                msg = "voice catalog IDs and aliases must be unique"
                raise ValueError(msg)
            lookup[voice.id] = voice
        for voice in catalog:
            for alias in voice.aliases:
                if alias in lookup:
                    msg = "voice catalog IDs and aliases must be unique"
                    raise ValueError(msg)
                lookup[alias] = voice
        object.__setattr__(self, "catalog", catalog)
        object.__setattr__(self, "_catalog_lookup", MappingProxyType(lookup))

    def resolve_voice_id(self, voice_id: str) -> PortableVoice:
        return self._catalog_lookup[voice_id]
