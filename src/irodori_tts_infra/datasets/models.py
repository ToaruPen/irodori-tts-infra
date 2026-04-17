from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Self

MIN_SAMPLE_RATE = 16_000
MAX_SAMPLE_RATE = 48_000


@dataclass(frozen=True, slots=True)
class ExtractedClip:
    path: str
    duration_s: float

    def __post_init__(self) -> None:
        _require_clip_path(self.path)
        _require_positive_number(self.duration_s, "clip duration")

    def to_json_dict(self) -> dict[str, Any]:
        return {"path": self.path, "duration_s": self.duration_s}

    @classmethod
    def from_json_dict(cls, payload: Mapping[str, Any]) -> Self:
        return cls(
            path=_require_str(payload, "path"),
            duration_s=_require_number(payload, "duration_s"),
        )


@dataclass(frozen=True, slots=True)
class ExtractionIndex:
    dataset: str
    sample_rate: int
    include_nsfw: bool
    total_bytes: int
    total_duration_s: float
    characters: Mapping[str, tuple[ExtractedClip, ...]]

    def __post_init__(self) -> None:
        dataset = _require_str_value(self.dataset, "dataset")
        sample_rate: object = self.sample_rate
        if not isinstance(sample_rate, int) or isinstance(sample_rate, bool):
            msg = "sample_rate must be an integer"
            raise TypeError(msg)
        include_nsfw: object = self.include_nsfw
        if not isinstance(include_nsfw, bool):
            msg = "include_nsfw must be a boolean"
            raise TypeError(msg)
        total_bytes = _require_int_value(self.total_bytes, "total_bytes")
        total_duration_s = _require_number_value(self.total_duration_s, "total_duration_s")
        raw_characters: object = self.characters
        if not isinstance(raw_characters, Mapping):
            msg = "characters must be a mapping"
            raise TypeError(msg)

        if not dataset.strip():
            msg = "dataset must not be blank"
            raise ValueError(msg)
        if not MIN_SAMPLE_RATE <= sample_rate <= MAX_SAMPLE_RATE:
            msg = "sample_rate must be between 16000 and 48000"
            raise ValueError(msg)
        if total_bytes < 0:
            msg = "total_bytes must be non-negative"
            raise ValueError(msg)
        if not math.isfinite(total_duration_s) or total_duration_s < 0:
            msg = "total_duration_s must be non-negative"
            raise ValueError(msg)

        object.__setattr__(self, "characters", _normalize_characters(raw_characters))

    @property
    def path_durations_by_character(self) -> dict[str, tuple[tuple[str, float], ...]]:
        return {
            character: tuple((clip.path, clip.duration_s) for clip in clips)
            for character, clips in self.characters.items()
        }

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "characters": {
                character: list(path_durations)
                for character, path_durations in self.path_durations_by_character.items()
            },
            "dataset": self.dataset,
            "include_nsfw": self.include_nsfw,
            "sample_rate": self.sample_rate,
            "total_bytes": self.total_bytes,
            "total_duration_s": self.total_duration_s,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_json_dict(), ensure_ascii=False, indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, payload: str) -> Self:
        data = json.loads(payload)
        if not isinstance(data, dict):
            msg = "index payload must be a JSON object"
            raise TypeError(msg)
        raw_characters = data["characters"]
        if not isinstance(raw_characters, dict):
            msg = "characters payload must be a mapping"
            raise TypeError(msg)
        include_nsfw = data["include_nsfw"]
        if not isinstance(include_nsfw, bool):
            msg = "include_nsfw must be a boolean"
            raise TypeError(msg)
        return cls(
            dataset=_require_str(data, "dataset"),
            sample_rate=_require_int(data, "sample_rate"),
            include_nsfw=include_nsfw,
            total_bytes=_require_int(data, "total_bytes"),
            total_duration_s=_require_number(data, "total_duration_s"),
            characters={
                str(character): tuple(
                    ExtractedClip.from_json_dict({"path": path, "duration_s": duration_s})
                    for path, duration_s in entries
                )
                for character, entries in raw_characters.items()
            },
        )


def _normalize_characters(
    raw_characters: Mapping[object, object],
) -> Mapping[str, tuple[ExtractedClip, ...]]:
    normalized: dict[str, tuple[ExtractedClip, ...]] = {}
    for character, clips in raw_characters.items():
        if not isinstance(character, str):
            msg = "character keys must be strings"
            raise TypeError(msg)
        if not character.strip():
            msg = "character keys must not be blank"
            raise ValueError(msg)
        raw_clips: object = clips
        if not isinstance(raw_clips, Sequence) or isinstance(
            raw_clips,
            (str, bytes, bytearray),
        ):
            msg = "character clip values must be a sequence of ExtractedClip instances"
            raise TypeError(msg)
        normalized_clips: list[ExtractedClip] = []
        for clip in raw_clips:
            if not isinstance(clip, ExtractedClip):
                msg = "character clip values must contain only ExtractedClip instances"
                raise TypeError(msg)
            normalized_clips.append(clip)
        normalized[character] = tuple(normalized_clips)
    return MappingProxyType(normalized)


def _require_clip_path(value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        msg = "clip path must be non-empty string and not blank"
        raise ValueError(msg)


def _require_positive_number(value: object, key: str) -> None:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value <= 0
    ):
        msg = f"{key} must be positive number"
        raise ValueError(msg)


def _require_str_value(value: object, key: str) -> str:
    if not isinstance(value, str):
        msg = f"{key} must be a string"
        raise TypeError(msg)
    return value


def _require_int_value(value: object, key: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        msg = f"{key} must be an integer"
        raise TypeError(msg)
    return value


def _require_number_value(value: object, key: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        msg = f"{key} must be a number"
        raise TypeError(msg)
    return float(value)


def _require_str(payload: Mapping[str, Any], key: str) -> str:
    return _require_str_value(payload[key], key)


def _require_int(payload: Mapping[str, Any], key: str) -> int:
    return _require_int_value(payload[key], key)


def _require_number(payload: Mapping[str, Any], key: str) -> float:
    return _require_number_value(payload[key], key)
