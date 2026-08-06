from __future__ import annotations

from io import BytesIO

import pytest
from pydantic import ValidationError

from irodori_tts_infra.contracts import (
    MAX_CHUNK_SIZE_BYTES,
    MAX_NUM_CANDIDATES,
    MAX_NUM_STEPS,
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
    style_caption,
)

pytestmark = pytest.mark.unit

DEFAULT_NUM_STEPS = 40
DEFAULT_CFG_SCALE_TEXT = 3.0
DEFAULT_CFG_SCALE_CAPTION = 3.0
DEFAULT_CFG_SCALE_SPEAKER = 5.0
WIRE_TEST_CHUNK_BYTES = 4


def test_synthesis_request_defaults_and_validation() -> None:
    request = SynthesisRequest(text="こんにちは")

    assert "ref_embed" not in SynthesisRequest.model_fields
    assert request.num_steps == DEFAULT_NUM_STEPS
    assert request.cfg_scale_text == pytest.approx(DEFAULT_CFG_SCALE_TEXT)
    assert request.cfg_scale_caption == pytest.approx(DEFAULT_CFG_SCALE_CAPTION)
    assert request.cfg_scale_speaker == pytest.approx(DEFAULT_CFG_SCALE_SPEAKER)
    assert request.style == "neutral"
    assert request.seed is None
    assert request.duration_scale == pytest.approx(1.0)
    assert request.num_candidates == 1
    assert request.t_schedule_mode == "linear"
    assert request.sway_coeff == pytest.approx(-1.0)

    with pytest.raises(ValidationError, match="text"):
        SynthesisRequest(text="")

    with pytest.raises(ValidationError, match="text"):
        SynthesisRequest(text="   ")

    with pytest.raises(ValidationError, match="num_candidates"):
        SynthesisRequest(
            text="こんにちは",
            num_candidates=0,
        )
    with pytest.raises(ValidationError, match="num_candidates"):
        SynthesisRequest(
            text="こんにちは",
            num_candidates=MAX_NUM_CANDIDATES + 1,
        )
    with pytest.raises(ValidationError, match="num_steps"):
        SynthesisRequest(text="こんにちは", num_steps=MAX_NUM_STEPS + 1)
    with pytest.raises(ValidationError, match="style"):
        SynthesisRequest(text="こんにちは", style="dramatic")  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="cfg_scale_caption"):
        SynthesisRequest(text="こんにちは", cfg_scale_caption=0)


@pytest.mark.parametrize(
    "field",
    [
        "cfg_scale_text",
        "cfg_scale_caption",
        "cfg_scale_speaker",
        "duration_scale",
        "sway_coeff",
    ],
)
def test_synthesis_request_rejects_non_finite_sampling_values(field: str) -> None:
    payload = f'{{"text":"こんにちは","{field}":1e309}}'

    with pytest.raises(ValidationError, match=field):
        SynthesisRequest.model_validate_json(payload)


@pytest.mark.parametrize(
    ("style", "expected"),
    [
        ("neutral", None),
        ("calm", "穏やかで優しい女性の声で、自然に話す。"),
        ("cheerful", "明るく親しみやすい女性の声で、自然に話す。"),
        ("clear", "子どもに伝わるように、ゆっくり明瞭な女性の声で話す。"),
    ],
)
def test_style_caption_maps_public_style_to_fixed_caption(
    style: str,
    expected: str | None,
) -> None:
    assert style_caption(style) == expected  # type: ignore[arg-type]


def test_synthesis_request_normalizes_optional_identifiers() -> None:
    request = SynthesisRequest(
        text="こんにちは",
        speaker="  ミカ  ",
    )

    assert request.speaker == "ミカ"


def test_synthesis_request_accepts_legacy_speaker_or_versioned_voice_pair() -> None:
    legacy = SynthesisRequest(text="こんにちは", speaker="  legacy-speaker  ")
    versioned = SynthesisRequest(
        text="こんにちは",
        voice_id="  fixture-voice  ",
        if_generation="  fixture-generation  ",
    )

    assert legacy.speaker == "legacy-speaker"
    assert legacy.voice_id is None
    assert legacy.if_generation is None
    assert versioned.speaker is None
    assert versioned.voice_id == "fixture-voice"
    assert versioned.if_generation == "fixture-generation"


@pytest.mark.parametrize(
    "payload",
    [
        {"voice_id": "fixture-voice"},
        {"if_generation": "fixture-generation"},
        {
            "speaker": "legacy-speaker",
            "voice_id": "fixture-voice",
            "if_generation": "fixture-generation",
        },
    ],
)
def test_synthesis_request_rejects_incomplete_or_ambiguous_voice_selection(
    payload: dict[str, str],
) -> None:
    with pytest.raises(ValidationError, match="voice_id"):
        SynthesisRequest.model_validate({"text": "こんにちは", **payload})


def test_synthesis_segment_applies_versioned_voice_validation() -> None:
    with pytest.raises(ValidationError, match="voice_id"):
        SynthesisSegment(segment_index=0, text="こんにちは", voice_id="fixture-voice")


def test_synthesis_request_does_not_publish_freeform_caption() -> None:
    with pytest.raises(ValidationError, match="caption"):
        SynthesisRequest.model_validate(
            {
                "text": "こんにちは",
                "caption": "arbitrary delivery instruction",
            }
        )


def test_synthesis_request_does_not_publish_internal_ref_embed() -> None:
    with pytest.raises(ValidationError, match="ref_embed"):
        SynthesisRequest.model_validate(
            {
                "text": "こんにちは",
                "ref_embed": "speakers/client-local.speaker.safetensors",
            }
        )


def test_contracts_round_trip_through_json() -> None:
    request = BatchSynthesisRequest(
        segments=[
            SynthesisSegment(
                segment_index=0,
                text="地の文です。",
            ),
            SynthesisSegment(
                segment_index=1,
                text="台詞です。",
            ),
        ],
    )
    result = BatchSynthesisRequest.model_validate_json(request.model_dump_json())

    assert result == request

    health = HealthResponse(status="ok", model_loaded=True)
    voice = VoiceProfileResponse(
        name="Narrator",
    )
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


def test_stream_header_serializes_terminal_error_code() -> None:
    header = StreamChunkHeader(
        segment_index=1,
        byte_length=0,
        final=True,
        error_code="backend_unavailable",
    )

    wire = header.to_bytes()
    assert b"backend_unavailable" in wire
    assert StreamChunkHeader.from_bytes(wire) == header

    normal_header = StreamChunkHeader(segment_index=1, byte_length=4)
    assert b"error_code" not in normal_header.to_bytes()


@pytest.mark.parametrize(
    "case",
    [
        (False, 0, "final"),
        (True, 1, "byte_length"),
    ],
)
def test_stream_header_rejects_malformed_terminal_error_frame(
    case: tuple[bool, int, str],
) -> None:
    final, byte_length, expected_error = case

    with pytest.raises(ValidationError, match=expected_error):
        StreamChunkHeader(
            segment_index=1,
            byte_length=byte_length,
            final=final,
            error_code="backend_unavailable",
        )


def test_stream_header_accepts_terminal_error_frame_shape() -> None:
    header = StreamChunkHeader(
        segment_index=1,
        byte_length=0,
        final=True,
        error_code="backpressure",
    )

    assert header.error_code == "backpressure"
    assert header.final is True
    assert header.byte_length == 0


@pytest.mark.parametrize(
    "case",
    [
        (False, 0),
        (False, 16),
        (True, 0),
        (True, 16),
    ],
)
def test_stream_header_without_error_code_allows_regular_chunk_shapes(
    case: tuple[bool, int],
) -> None:
    final, byte_length = case

    header = StreamChunkHeader(segment_index=1, byte_length=byte_length, final=final)

    assert header.error_code is None
    assert header.final is final
    assert header.byte_length == byte_length


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


def test_v4_stream_wire_version_uses_compact_v1_aliases() -> None:
    assert STREAM_HEADER_VERSION == 1
    handshake = StreamHandshakeHeader(max_chunk_size=1024).model_dump(mode="json", by_alias=True)
    chunk = StreamChunkHeader(
        segment_index=0,
        byte_length=WIRE_TEST_CHUNK_BYTES,
        final=True,
    ).model_dump(
        mode="json",
        by_alias=True,
    )

    assert handshake == {"kind": "handshake", "v": 1, "max_chunk_size": 1024}
    assert chunk["kind"] == "chunk"
    assert chunk["v"] == 1
    assert chunk["index"] == 0
    assert chunk["nbytes"] == WIRE_TEST_CHUNK_BYTES
    assert chunk["final"] is True
    assert "header_version" not in chunk
    assert "segment_index" not in chunk
    assert "byte_length" not in chunk


def test_stream_handshake_header_rejects_out_of_range_max_chunk_size() -> None:
    with pytest.raises(ValidationError, match="max_chunk_size"):
        StreamHandshakeHeader(max_chunk_size=0)
    with pytest.raises(ValidationError, match="max_chunk_size"):
        StreamHandshakeHeader(max_chunk_size=MAX_CHUNK_SIZE_BYTES + 1)


def test_voice_profile_aliases_validation() -> None:
    profile = VoiceProfileResponse(
        name="Narrator",
        aliases=("Narrator-JP", "  Narrator-JP  ", "語り手"),
    )
    assert profile.aliases == ("Narrator-JP", "語り手")

    with pytest.raises(ValidationError, match="aliases"):
        VoiceProfileResponse(
            name="X",
            aliases=("   ",),
        )
