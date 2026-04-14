from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

ErrorDetailValue = str | int | float | bool | None


class ErrorPayload(BaseModel):
    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    details: dict[str, ErrorDetailValue] = Field(default_factory=dict)

    @field_validator("code", "message")
    @classmethod
    def _reject_blank_text(cls, value: str) -> str:
        if not value.strip():
            msg = "error payload text fields must not be blank"
            raise ValueError(msg)
        return value
