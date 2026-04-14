from __future__ import annotations

from io import BytesIO

import pytest
from pydantic import ValidationError

from irodori_tts_infra.contracts import (
    BatchSynthesisRequest,
    BatchSynthesisResult,
    ErrorPayload,
    HealthResponse,
    StreamChunkHeader,
    SynthesisRequest,
    SynthesisResult,
    SynthesisSegment,
    VoiceProfileResponse,
)

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


def test_stream_header_serialization_reconstructs_byte_exact_chunks() -> None:
    payloads = [b"RIFF\x00\x00first-wav", b"RIFF\x00\x01second-wav"]
    stream = b"".join(
        StreamChunkHeader(
            segment_index=index,
            byte_length=len(payload),
            elapsed_seconds=0.123 + index,
        ).to_bytes()
        + payload
        for index, payload in enumerate(payloads)
    )

    reader = BytesIO(stream)
    reconstructed: list[tuple[int, bytes]] = []
    while header_line := reader.readline():
        header = StreamChunkHeader.from_bytes(header_line)
        reconstructed.append((header.segment_index, reader.read(header.byte_length)))

    assert reconstructed == list(enumerate(payloads))
