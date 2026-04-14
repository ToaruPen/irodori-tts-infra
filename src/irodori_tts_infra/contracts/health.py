from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"] = "ok"
    model_loaded: bool = False
    detail: str | None = Field(default=None, min_length=1)

    @field_validator("detail")
    @classmethod
    def _reject_blank_detail(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            msg = "detail must not be blank"
            raise ValueError(msg)
        return value
