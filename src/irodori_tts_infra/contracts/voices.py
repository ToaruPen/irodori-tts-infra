from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class VoiceProfileResponse(BaseModel):
    name: str = Field(min_length=1)
    caption: str = Field(min_length=1)
    aliases: tuple[str, ...] = ()

    @field_validator("name", "caption")
    @classmethod
    def _reject_blank_text(cls, value: str) -> str:
        if not value.strip():
            msg = "voice profile text fields must not be blank"
            raise ValueError(msg)
        return value

    @field_validator("aliases", mode="before")
    @classmethod
    def _normalize_aliases(cls, value: object) -> object:
        if not isinstance(value, (list, tuple)):
            return value
        seen: set[str] = set()
        normalized: list[str] = []
        for raw in value:
            if not isinstance(raw, str) or not raw.strip():
                msg = "aliases must be non-blank strings"
                raise ValueError(msg)
            stripped = raw.strip()
            if stripped in seen:
                continue
            seen.add(stripped)
            normalized.append(stripped)
        return tuple(normalized)
