"""Speaker embedding manifest schema.

    [narrator]
    ref_embed = "speakers/narrator.speaker.safetensors"

    [characters.<name>]
    ref_embed = "speakers/mika.speaker.safetensors"

Paths resolve relative to the TOML file. Manifest entries for characters absent from
characters.md are rejected by load_voice_profile when characters.md is provided.
"""

from __future__ import annotations

import tomllib
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import TYPE_CHECKING, cast

from irodori_tts_infra.voice_bank.captions import load_characters_markdown
from irodori_tts_infra.voice_bank.models import (
    CharacterVoice,
    SpeakerEmbeddingProfile,
    VoiceProfile,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

SPEAKER_MANIFEST_FILENAME = "voice_bank_speakers.toml"


def find_characters_markdown(turn_file: Path) -> Path | None:
    return _find_upwards(turn_file, "characters.md")


def find_speaker_manifest(turn_file: Path) -> Path | None:
    return _find_upwards(turn_file, SPEAKER_MANIFEST_FILENAME)


def load_voice_profile(
    characters_md: Path | None,
    *,
    speaker_manifest: Path | None = None,
) -> VoiceProfile:
    if speaker_manifest is None:
        msg = "speaker manifest is required"
        raise ValueError(msg)

    known_names: set[str] = set()
    resolved_characters_md = (
        characters_md if characters_md is not None and characters_md.is_file() else None
    )
    if resolved_characters_md is not None:
        known_names = load_characters_markdown(
            resolved_characters_md.read_text(encoding="utf-8"),
        )

    narrator, characters = _load_speaker_manifest(speaker_manifest)
    unknown_names = sorted(set(characters) - known_names) if resolved_characters_md else []
    if unknown_names:
        msg = (
            "speaker manifest contains characters not present in characters.md: "
            f"{', '.join(unknown_names)}"
        )
        raise ValueError(msg)

    return VoiceProfile(characters=characters, narrator=narrator)


def _find_upwards(turn_file: Path, filename: str) -> Path | None:
    current = turn_file.parent
    while current != current.parent:
        candidate = current / filename
        if candidate.is_file():
            return candidate
        if current.name == "chat":
            break
        current = current.parent
    return None


def _load_speaker_manifest(
    manifest: Path,
) -> tuple[SpeakerEmbeddingProfile, dict[str, CharacterVoice]]:
    data = tomllib.loads(manifest.read_text(encoding="utf-8"))
    if "narrator" not in data:
        msg = "narrator is required"
        raise ValueError(msg)

    narrator_table = _as_table(data["narrator"], "narrator")
    if "characters" in narrator_table:
        _as_table(narrator_table["characters"], "characters")
    narrator = _parse_speaker_profile(
        narrator_table,
        context="narrator",
        base_dir=manifest.parent,
    )
    character_tables = _as_table(data.get("characters", {}), "characters")
    characters: dict[str, CharacterVoice] = {}
    for name, value in character_tables.items():
        speaker = _parse_speaker_profile(
            _as_table(value, f"characters.{name}"),
            context=f"characters.{name}",
            base_dir=manifest.parent,
        )
        characters[name] = CharacterVoice(name=name, speaker=speaker)
    return narrator, characters


def _parse_speaker_profile(
    table: Mapping[str, object],
    *,
    context: str,
    base_dir: Path,
) -> SpeakerEmbeddingProfile:
    return SpeakerEmbeddingProfile(
        ref_embed=_required_path(table, "ref_embed", context, base_dir=base_dir),
    )


def _required_path(
    table: Mapping[str, object],
    key: str,
    context: str,
    *,
    base_dir: Path,
) -> Path:
    return _resolve_manifest_path(
        _string_value(table.get(key), f"{context}.{key}"),
        base_dir=base_dir,
    )


def _string_value(value: object, context: str) -> str:
    if value is None:
        msg = f"{context} is required"
        raise ValueError(msg)
    if not isinstance(value, str):
        msg = f"{context} must be a string"
        raise TypeError(msg)
    if not value.strip():
        msg = f"{context} must not be blank"
        raise ValueError(msg)
    return value


def _resolve_manifest_path(value: str, *, base_dir: Path) -> Path:
    path = Path(value)
    if (
        path.is_absolute()
        or PurePosixPath(value).is_absolute()
        or PureWindowsPath(value).is_absolute()
    ):
        msg = "manifest path values must be relative paths"
        raise ValueError(msg)
    return base_dir / path


def _as_table(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        msg = f"{context} must be a TOML table"
        raise TypeError(msg)
    return cast("Mapping[str, object]", value)
