from __future__ import annotations

from irodori_tts_infra.contracts.errors import ErrorDetailValue, ErrorPayload
from irodori_tts_infra.contracts.health import HealthResponse
from irodori_tts_infra.contracts.synthesis import (
    BatchSynthesisRequest,
    BatchSynthesisResult,
    StreamChunkHeader,
    SynthesisRequest,
    SynthesisResult,
    SynthesisSegment,
)
from irodori_tts_infra.contracts.voices import VoiceProfileResponse

__all__ = [
    "BatchSynthesisRequest",
    "BatchSynthesisResult",
    "ErrorDetailValue",
    "ErrorPayload",
    "HealthResponse",
    "StreamChunkHeader",
    "SynthesisRequest",
    "SynthesisResult",
    "SynthesisSegment",
    "VoiceProfileResponse",
]
