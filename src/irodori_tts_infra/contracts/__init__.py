from __future__ import annotations

from irodori_tts_infra.contracts.errors import ErrorDetailValue, ErrorPayload
from irodori_tts_infra.contracts.health import HealthResponse
from irodori_tts_infra.contracts.synthesis import (
    MAX_CHUNK_SIZE_BYTES,
    MAX_SEGMENT_INDEX,
    STREAM_HEADER_VERSION,
    BatchSynthesisRequest,
    BatchSynthesisResult,
    StreamChunkHeader,
    StreamHandshakeHeader,
    SynthesisRequest,
    SynthesisResult,
    SynthesisSegment,
)
from irodori_tts_infra.contracts.voices import VoiceProfileResponse

__all__ = [
    "MAX_CHUNK_SIZE_BYTES",
    "MAX_SEGMENT_INDEX",
    "STREAM_HEADER_VERSION",
    "BatchSynthesisRequest",
    "BatchSynthesisResult",
    "ErrorDetailValue",
    "ErrorPayload",
    "HealthResponse",
    "StreamChunkHeader",
    "StreamHandshakeHeader",
    "SynthesisRequest",
    "SynthesisResult",
    "SynthesisSegment",
    "VoiceProfileResponse",
]
