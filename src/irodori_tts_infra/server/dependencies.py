from __future__ import annotations

from typing import TYPE_CHECKING, cast

from fastapi import HTTPException, Request

from irodori_tts_infra.contracts import MAX_CHUNK_SIZE_BYTES, HealthResponse

if TYPE_CHECKING:
    from irodori_tts_infra.engine.pipeline import SynthesisPipeline


def get_pipeline(request: Request) -> SynthesisPipeline:
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        msg = "Synthesis pipeline is not configured"
        raise HTTPException(status_code=500, detail=msg)
    return cast("SynthesisPipeline", pipeline)


def get_max_chunk_size(request: Request) -> int:
    return int(getattr(request.app.state, "max_chunk_size", MAX_CHUNK_SIZE_BYTES))


def get_health_response(request: Request) -> HealthResponse:
    model_loaded = bool(getattr(request.app.state, "model_loaded", False))
    detail = cast("str | None", getattr(request.app.state, "health_detail", None))
    return HealthResponse(
        status="ok" if model_loaded else "degraded",
        model_loaded=model_loaded,
        detail=detail,
        max_chunk_size=get_max_chunk_size(request),
    )
