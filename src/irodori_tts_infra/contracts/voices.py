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
