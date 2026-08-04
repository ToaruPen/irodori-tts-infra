from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from irodori_tts_infra.contracts.voices import VoiceCapability  # noqa: TC001

Readiness = Literal["ready", "model_loading", "model_not_loaded", "voice_bank_invalid"]


class _ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DeliveryCaptionCapability(_ContractModel):
    supported: Literal[False] = False  # noqa: V107 - serialized contract field.
    max_chars: None = None


class EmojiCapability(_ContractModel):
    supported: bool = True  # noqa: V107 - serialized contract field.


class ConditioningCapabilities(_ContractModel):
    delivery_caption: DeliveryCaptionCapability = Field(  # noqa: V107
        default_factory=DeliveryCaptionCapability
    )
    emoji: EmojiCapability = Field(default_factory=EmojiCapability)  # noqa: V107


class CapabilitiesResponse(_ContractModel):
    contract_version: Literal[1] = 1  # noqa: V107 - serialized contract field.
    generation: str = Field(min_length=1)
    ready: bool
    readiness: Readiness
    voices: tuple[VoiceCapability, ...]
    conditioning: ConditioningCapabilities = Field(  # noqa: V107
        default_factory=ConditioningCapabilities
    )

    @field_validator("generation")
    @classmethod
    def _normalize_generation(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            msg = "generation must not be blank"
            raise ValueError(msg)
        return stripped

    @model_validator(mode="after")
    def _validate_readiness_and_catalog(self) -> Self:
        if self.ready != (self.readiness == "ready"):
            msg = "ready must be true exactly when readiness is ready"
            raise ValueError(msg)

        ids = [voice.id for voice in self.voices]
        aliases = [alias for voice in self.voices for alias in voice.aliases]
        if len(ids) != len(set(ids)):
            msg = "voice catalog IDs must be unique"
            raise ValueError(msg)
        if len(aliases) != len(set(aliases)):
            msg = "voice catalog aliases must be unique"
            raise ValueError(msg)
        if set(ids) & set(aliases):
            msg = "voice catalog aliases must not collide with IDs"
            raise ValueError(msg)
        if sum(voice.default for voice in self.voices) > 1:
            msg = "voice catalog must contain at most one default"
            raise ValueError(msg)
        return self
