from __future__ import annotations

from typing import TYPE_CHECKING, cast

from fastapi import Request  # noqa: TC002 - FastAPI resolves dependency annotations.

from irodori_tts_infra.contracts import (
    MAX_CHUNK_SIZE_BYTES,
    CapabilitiesResponse,
    ConditioningCapabilities,
    EmojiCapability,
    HealthResponse,
    Readiness,
    VoiceCapability,
)
from irodori_tts_infra.engine.errors import ModelNotLoadedError, VoiceBankInvalidError

if TYPE_CHECKING:
    from irodori_tts_infra.engine.pipeline import SynthesisPipeline


def get_pipeline(request: Request) -> SynthesisPipeline:
    readiness = getattr(request.app.state, "readiness", "model_not_loaded")
    if readiness == "voice_bank_invalid":
        msg = "Voice catalog is unavailable"
        raise VoiceBankInvalidError(msg)
    if readiness != "ready":
        msg = "Synthesis model is not loaded"
        raise ModelNotLoadedError(msg)
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        msg = "Synthesis model is not loaded"
        raise ModelNotLoadedError(msg)
    return cast("SynthesisPipeline", pipeline)


def get_max_chunk_size(request: Request) -> int:
    return int(getattr(request.app.state, "max_chunk_size", MAX_CHUNK_SIZE_BYTES))


def get_capabilities_response(request: Request) -> CapabilitiesResponse:
    pipeline = cast(
        "SynthesisPipeline | None",
        getattr(request.app.state, "pipeline", None),
    )
    readiness = cast(
        "Readiness",
        getattr(request.app.state, "readiness", "model_not_loaded"),
    )
    voices: tuple[VoiceCapability, ...] = ()
    if readiness == "ready" and pipeline is not None:
        voices = tuple(
            VoiceCapability(
                id=voice.id,
                label=voice.label,
                aliases=voice.aliases,
                default=voice.default,
            )
            for voice in pipeline.voice_profile.catalog
        )
    return CapabilitiesResponse(
        generation=str(getattr(request.app.state, "generation", "unconfigured")),
        ready=readiness == "ready",
        readiness=readiness,
        voices=voices,
        conditioning=ConditioningCapabilities(
            emoji=EmojiCapability(
                supported=bool(getattr(request.app.state, "emoji_conditioning_supported", True))
            )
        ),
    )


def get_health_response(request: Request) -> HealthResponse:
    model_loaded = bool(getattr(request.app.state, "model_loaded", False))
    detail = cast("str | None", getattr(request.app.state, "health_detail", None))
    return HealthResponse(
        status="ok" if model_loaded else "degraded",
        model_loaded=model_loaded,
        detail=detail,
        max_chunk_size=get_max_chunk_size(request),
    )
