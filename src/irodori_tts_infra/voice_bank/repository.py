"""RVC sidecar manifest schema.

    [characters.<name>]
    model_path = "..."
    sample_rate = 40000
    neutral_prototype = "..."

    [characters.<name>.state_prototypes]
    happy = "..."

Paths resolve relative to the TOML file. Manifest entries for characters absent
from characters.md are rejected by load_voice_profile.
"""

from __future__ import annotations

import tomllib
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import TYPE_CHECKING, cast

from irodori_tts_infra.voice_bank.captions import (
    DEFAULT_GENERIC_DIALOGUE_CAPTION,
    DEFAULT_NARRATOR_CAPTION,
    load_characters_markdown,
)
from irodori_tts_infra.voice_bank.models import CharacterVoice, RVCProfile, VoiceProfile

if TYPE_CHECKING:
    from collections.abc import Mapping

RVC_MANIFEST_FILENAME = "voice_bank_rvc.toml"


def find_characters_markdown(turn_file: Path) -> Path | None:
    return _find_upwards(turn_file, "characters.md")


def find_rvc_manifest(turn_file: Path) -> Path | None:
    return _find_upwards(turn_file, RVC_MANIFEST_FILENAME)


def load_voice_profile(
    characters_md: Path | None,
    *,
    narrator_caption: str = DEFAULT_NARRATOR_CAPTION,
    generic_dialogue_caption: str = DEFAULT_GENERIC_DIALOGUE_CAPTION,
    rvc_manifest: Path | None = None,
) -> VoiceProfile:
    characters: dict[str, CharacterVoice] = {}
    if characters_md is not None and characters_md.is_file():
        characters = load_characters_markdown(characters_md.read_text(encoding="utf-8"))

    if rvc_manifest is not None:
        rvc_profiles = _load_rvc_manifest(rvc_manifest)
        unknown_names = sorted(set(rvc_profiles) - set(characters))
        if unknown_names:
            msg = (
                "RVC manifest contains characters not present in characters.md: "
                f"{', '.join(unknown_names)}"
            )
            raise ValueError(msg)
        characters = _merge_rvc_profiles(characters, rvc_profiles)

    return VoiceProfile(
        characters=characters,
        narrator_caption=narrator_caption,
        generic_dialogue_caption=generic_dialogue_caption,
    )


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


def _load_rvc_manifest(manifest: Path) -> dict[str, RVCProfile]:
    data = tomllib.loads(manifest.read_text(encoding="utf-8"))
    character_tables = _as_table(data.get("characters", {}), "characters")
    profiles: dict[str, RVCProfile] = {}
    for name, value in character_tables.items():
        profiles[name] = _parse_rvc_profile(
            name,
            _as_table(value, f"characters.{name}"),
            base_dir=manifest.parent,
        )
    return profiles


def _parse_rvc_profile(
    name: str,
    table: Mapping[str, object],
    *,
    base_dir: Path,
) -> RVCProfile:
    neutral_prototype = table.get("neutral_prototype")
    if neutral_prototype is not None and not isinstance(neutral_prototype, str):
        msg = f"characters.{name}.neutral_prototype must be a string"
        raise TypeError(msg)

    return RVCProfile(
        model_path=_required_path(table, "model_path", f"characters.{name}", base_dir=base_dir),
        sample_rate=_required_int(table, "sample_rate", f"characters.{name}"),
        neutral_prototype=(
            _resolve_manifest_path(neutral_prototype, base_dir=base_dir)
            if neutral_prototype is not None
            else None
        ),
        state_prototypes=_state_prototypes(name, table, base_dir=base_dir),
    )


def _state_prototypes(
    name: str,
    table: Mapping[str, object],
    *,
    base_dir: Path,
) -> dict[str, Path]:
    raw_state_prototypes = table.get("state_prototypes", {})
    state_prototypes = _as_table(
        raw_state_prototypes,
        f"characters.{name}.state_prototypes",
    )
    return {
        state: _resolve_manifest_path(
            _string_value(value, f"characters.{name}.state_prototypes.{state}"),
            base_dir=base_dir,
        )
        for state, value in state_prototypes.items()
    }


def _merge_rvc_profiles(
    characters: Mapping[str, CharacterVoice],
    rvc_profiles: Mapping[str, RVCProfile],
) -> dict[str, CharacterVoice]:
    return {
        name: CharacterVoice(
            name=character.name,
            caption=character.caption,
            rvc=rvc_profiles.get(name, character.rvc),
        )
        for name, character in characters.items()
    }


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


def _required_int(table: Mapping[str, object], key: str, context: str) -> int:
    value = table.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        msg = f"{context}.{key} must be an integer"
        raise TypeError(msg)
    return value


def _string_value(value: object, context: str) -> str:
    if not isinstance(value, str):
        msg = f"{context} must be a string"
        raise TypeError(msg)
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
