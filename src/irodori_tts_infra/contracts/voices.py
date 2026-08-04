from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class _ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class VoiceCapability(_ContractModel):
    id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    aliases: tuple[str, ...] = ()
    default: bool = False

    @field_validator("id", "label")
    @classmethod
    def _normalize_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            msg = "voice capability text fields must not be blank"
            raise ValueError(msg)
        return stripped

    @field_validator("aliases", mode="before")
    @classmethod
    def _normalize_aliases(cls, value: object) -> object:
        if not isinstance(value, (list, tuple)):
            return value
        normalized: list[str] = []
        for raw in value:
            if not isinstance(raw, str) or not raw.strip():
                msg = "aliases must be non-blank strings"
                raise ValueError(msg)
            stripped = raw.strip()
            if stripped in normalized:
                msg = "aliases must be unique"
                raise ValueError(msg)
            normalized.append(stripped)
        return tuple(normalized)


class VoiceProfileResponse(_ContractModel):
    name: str = Field(min_length=1)
    aliases: tuple[str, ...] = ()

    @field_validator("name")
    @classmethod
    def _reject_blank_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            msg = "voice profile text fields must not be blank"
            raise ValueError(msg)
        return stripped

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
