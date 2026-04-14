from __future__ import annotations

from io import BytesIO

import pytest
from pydantic import ValidationError

from irodori_tts_infra.contracts import (
    MAX_CHUNK_SIZE_BYTES,
    MAX_SEGMENT_INDEX,
    STREAM_HEADER_VERSION,
    BatchSynthesisRequest,
    BatchSynthesisResult,
    ErrorPayload,
    HealthResponse,
    StreamChunkHeader,
    StreamHandshakeHeader,
    SynthesisRequest,
    SynthesisResult,
    SynthesisSegment,
    VoiceProfileResponse,
)

pytestmark = pytest.mark.unit

DEFAULT_NUM_STEPS = 30
DEFAULT_CFG_SCALE_TEXT = 3.0
DEFAULT_CFG_SCALE_CAPTION = 3.5


def test_synthesis_request_defaults_and_validation() -> None:
    request = SynthesisRequest(text="こんにちは", caption="女性が話している。")

    assert request.num_steps == DEFAULT_NUM_STEPS
    assert request.cfg_scale_text == pytest.approx(DEFAULT_CFG_SCALE_TEXT)
    assert request.cfg_scale_caption == pytest.approx(DEFAULT_CFG_SCALE_CAPTION)
    assert request.no_ref is True

    with pytest.raises(ValidationError, match="text"):
        SynthesisRequest(text="", caption="女性が話している。")

    with pytest.raises(ValidationError, match="caption"):
        SynthesisRequest(text="こんにちは", caption="   ")


def test_contracts_round_trip_through_json() -> None:
    request = BatchSynthesisRequest(
        segments=[
            SynthesisSegment(
                segment_index=0,
                text="地の文です。",
                caption="女性が読み上げている。",
            ),
            SynthesisSegment(segment_index=1, text="台詞です。", caption="男性が話している。"),
        ],
    )
    result = BatchSynthesisRequest.model_validate_json(request.model_dump_json())

    assert result == request

    health = HealthResponse(status="ok", model_loaded=True)
    voice = VoiceProfileResponse(name="Narrator", caption="落ち着いた女性の声。")
    error = ErrorPayload(code="validation_error", message="invalid request")
    synthesis_result = SynthesisResult(
        segment_index=0, wav_bytes=b"RIFF-data", elapsed_seconds=0.25
    )

    assert HealthResponse.model_validate_json(health.model_dump_json()) == health
    assert VoiceProfileResponse.model_validate_json(voice.model_dump_json()) == voice
    assert ErrorPayload.model_validate_json(error.model_dump_json()) == error
    assert (
        SynthesisResult.model_validate_json(synthesis_result.model_dump_json()) == synthesis_result
    )


def test_batch_results_must_be_ordered_by_segment_index() -> None:
    ordered = BatchSynthesisResult(
        results=[
            SynthesisResult(segment_index=0, wav_bytes=b"first", elapsed_seconds=0.5),
            SynthesisResult(segment_index=1, wav_bytes=b"second", elapsed_seconds=0.7),
        ],
        total_elapsed_seconds=1.2,
    )

    assert [result.segment_index for result in ordered.results] == [0, 1]

    with pytest.raises(ValidationError, match="ordered"):
        BatchSynthesisResult(
            results=[
                SynthesisResult(segment_index=1, wav_bytes=b"second", elapsed_seconds=0.7),
                SynthesisResult(segment_index=0, wav_bytes=b"first", elapsed_seconds=0.5),
            ],
            total_elapsed_seconds=1.2,
        )

    # Gap in the sequence (0, 2) is rejected
    with pytest.raises(ValidationError, match="ordered"):
        BatchSynthesisResult(
            results=[
                SynthesisResult(segment_index=0, wav_bytes=b"first", elapsed_seconds=0.5),
                SynthesisResult(segment_index=2, wav_bytes=b"third", elapsed_seconds=0.9),
            ],
            total_elapsed_seconds=1.4,
        )

    # Non-zero start (1, 2) is rejected
    with pytest.raises(ValidationError, match="ordered"):
        BatchSynthesisResult(
            results=[
                SynthesisResult(segment_index=1, wav_bytes=b"second", elapsed_seconds=0.7),
                SynthesisResult(segment_index=2, wav_bytes=b"third", elapsed_seconds=0.9),
            ],
            total_elapsed_seconds=1.6,
        )


def test_stream_header_serialization_reconstructs_byte_exact_chunks() -> None:
    payloads = [b"RIFF\x00\x00first-wav", b"RIFF\x00\x01second-wav"]
    stream = b"".join(
        StreamChunkHeader(
            segment_index=index,
            byte_length=len(payload),
            elapsed_seconds=0.123 + index,
            final=index == len(payloads) - 1,
        ).to_bytes()
        + payload
        for index, payload in enumerate(payloads)
    )

    reader = BytesIO(stream)
    reconstructed: list[tuple[int, bytes, bool, int]] = []
    while header_line := reader.readline():
        header = StreamChunkHeader.from_bytes(header_line)
        reconstructed.append(
            (
                header.segment_index,
                reader.read(header.byte_length),
                header.final,
                header.header_version,
            ),
        )

    assert reconstructed == [
        (0, payloads[0], False, STREAM_HEADER_VERSION),
        (1, payloads[1], True, STREAM_HEADER_VERSION),
    ]


def test_stream_header_boundary_values() -> None:
    zero = StreamChunkHeader(segment_index=0, byte_length=0, final=True)
    assert StreamChunkHeader.from_bytes(zero.to_bytes()) == zero

    at_max = StreamChunkHeader(segment_index=0, byte_length=MAX_CHUNK_SIZE_BYTES)
    assert at_max.byte_length == MAX_CHUNK_SIZE_BYTES

    with pytest.raises(ValidationError, match="byte_length"):
        StreamChunkHeader(segment_index=0, byte_length=MAX_CHUNK_SIZE_BYTES + 1)

    at_index_cap = StreamChunkHeader(segment_index=MAX_SEGMENT_INDEX, byte_length=0)
    assert at_index_cap.segment_index == MAX_SEGMENT_INDEX
    with pytest.raises(ValidationError, match="segment_index"):
        StreamChunkHeader(segment_index=MAX_SEGMENT_INDEX + 1, byte_length=0)


def test_stream_header_defaults_include_version() -> None:
    header = StreamChunkHeader(segment_index=3, byte_length=128)
    assert header.header_version == STREAM_HEADER_VERSION
    assert header.final is False


def test_health_response_rejects_whitespace_only_detail() -> None:
    with pytest.raises(ValidationError, match="detail"):
        HealthResponse(status="degraded", model_loaded=False, detail="   ")


def test_stream_header_from_bytes_accepts_optional_trailing_newline() -> None:
    header = StreamChunkHeader(segment_index=1, byte_length=32)
    wire = header.to_bytes()
    assert wire.endswith(b"\n")
    assert StreamChunkHeader.from_bytes(wire) == header
    assert StreamChunkHeader.from_bytes(wire.rstrip(b"\n")) == header


def test_health_response_advertises_max_chunk_size() -> None:
    default = HealthResponse()
    assert default.max_chunk_size == MAX_CHUNK_SIZE_BYTES

    lowered_cap = 1024
    override = HealthResponse(max_chunk_size=lowered_cap)
    assert override.max_chunk_size == lowered_cap

    with pytest.raises(ValidationError, match="max_chunk_size"):
        HealthResponse(max_chunk_size=MAX_CHUNK_SIZE_BYTES + 1)


def test_stream_handshake_header_roundtrip_and_kind_discriminator() -> None:
    lowered_cap = 1024
    handshake = StreamHandshakeHeader(max_chunk_size=lowered_cap)
    wire = handshake.to_bytes()
    assert StreamHandshakeHeader.from_bytes(wire) == handshake

    chunk = StreamChunkHeader(segment_index=0, byte_length=4)
    assert chunk.kind == "chunk"
    chunk_json = chunk.model_dump(mode="json", by_alias=True)
    assert chunk_json["kind"] == "chunk"

    handshake_json = handshake.model_dump(mode="json", by_alias=True)
    assert handshake_json["kind"] == "handshake"
    assert "segment_index" not in handshake_json
    assert "byte_length" not in handshake_json


def test_stream_handshake_header_rejects_out_of_range_max_chunk_size() -> None:
    with pytest.raises(ValidationError, match="max_chunk_size"):
        StreamHandshakeHeader(max_chunk_size=0)
    with pytest.raises(ValidationError, match="max_chunk_size"):
        StreamHandshakeHeader(max_chunk_size=MAX_CHUNK_SIZE_BYTES + 1)


def test_voice_profile_aliases_validation() -> None:
    profile = VoiceProfileResponse(
        name="Narrator",
        caption="落ち着いた声。",
        aliases=("Narrator-JP", "  Narrator-JP  ", "語り手"),
    )
    assert profile.aliases == ("Narrator-JP", "語り手")

    with pytest.raises(ValidationError, match="aliases"):
        VoiceProfileResponse(name="X", caption="Y", aliases=("   ",))
