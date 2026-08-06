from __future__ import annotations

from irodori_tts_infra.contracts.capabilities import (
    CapabilitiesResponse,
    ConditioningCapabilities,
    DeliveryCaptionCapability,
    EmojiCapability,
    Readiness,
)
from irodori_tts_infra.contracts.errors import ErrorDetailValue, ErrorPayload
from irodori_tts_infra.contracts.health import HealthResponse
from irodori_tts_infra.contracts.synthesis import (
    DEFAULT_NUM_STEPS,
    MAX_CHUNK_SIZE_BYTES,
    MAX_NUM_CANDIDATES,
    MAX_NUM_STEPS,
    MAX_SEGMENT_INDEX,
    STREAM_HEADER_VERSION,
    BatchSynthesisRequest,
    BatchSynthesisResult,
    IrodoriStyle,
    StreamChunkHeader,
    StreamErrorCode,
    StreamHandshakeHeader,
    SynthesisRequest,
    SynthesisResult,
    SynthesisSegment,
    style_caption,
)
from irodori_tts_infra.contracts.voices import VoiceCapability, VoiceProfileResponse

__all__ = [
    "DEFAULT_NUM_STEPS",
    "MAX_CHUNK_SIZE_BYTES",
    "MAX_NUM_CANDIDATES",
    "MAX_NUM_STEPS",
    "MAX_SEGMENT_INDEX",
    "STREAM_HEADER_VERSION",
    "BatchSynthesisRequest",
    "BatchSynthesisResult",
    "CapabilitiesResponse",
    "ConditioningCapabilities",
    "DeliveryCaptionCapability",
    "EmojiCapability",
    "ErrorDetailValue",
    "ErrorPayload",
    "HealthResponse",
    "IrodoriStyle",
    "Readiness",
    "StreamChunkHeader",
    "StreamErrorCode",
    "StreamHandshakeHeader",
    "SynthesisRequest",
    "SynthesisResult",
    "SynthesisSegment",
    "VoiceCapability",
    "VoiceProfileResponse",
    "style_caption",
]
