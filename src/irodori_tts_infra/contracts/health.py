from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"] = "ok"
    model_loaded: bool = False
    detail: str | None = Field(default=None, min_length=1)
