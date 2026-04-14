from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient
from starlette import status

from irodori_tts_infra.contracts import (
    MAX_CHUNK_SIZE_BYTES,
    BatchSynthesisResult,
    StreamChunkHeader,
    StreamHandshakeHeader,
    SynthesisResult,
)
from irodori_tts_infra.engine.backends.fake import FakeSynthesizer, FakeSynthResponse
from irodori_tts_infra.engine.models import PipelineConfig, SynthesizedAudio
from irodori_tts_infra.server.app import create_app

if TYPE_CHECKING:
    from collections.abc import Callable

    from httpx import Response

    from irodori_tts_infra.contracts import SynthesisRequest
    from irodori_tts_infra.engine.pipeline import SynthesisPipeline

pytestmark = pytest.mark.unit

STREAM_MAX_CHUNK_SIZE = 64
SPLIT_STREAM_CHUNK_SIZE = 4
TEST_EVENT_TIMEOUT = 5.0


class BlockingSynthesizer:
    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()

    def synthesize(self, request: SynthesisRequest) -> SynthesizedAudio:
        del request
        self.entered.set()
        assert _wait_for_event(self.release)
        return SynthesizedAudio(wav_bytes=b"RIFFblocked", sample_rate=24_000)


def test_synthesize_returns_synthesis_result(
    pipeline_factory: Callable[..., SynthesisPipeline],
) -> None:
    app = create_app(
        pipeline_factory(
            FakeSynthesizer(
                responses=[
                    FakeSynthResponse(
                        audio=SynthesizedAudio(wav_bytes=b"RIFFsingle", sample_rate=24_000),
                    ),
                ],
            ),
        ),
    )

    with TestClient(app) as client:
        response = client.post(
            "/synthesize",
            json={"text": "本文", "caption": "声の説明。"},
        )

    assert response.status_code == status.HTTP_200_OK
    result = SynthesisResult.model_validate_json(response.text)
    assert result.segment_index == 0
    assert result.wav_bytes == b"RIFFsingle"
    assert result.content_type == "audio/wav"


def test_synthesize_returns_200_for_empty_wav_bytes(
    pipeline_factory: Callable[..., SynthesisPipeline],
) -> None:
    app = create_app(
        pipeline_factory(
            FakeSynthesizer(
                responses=[
                    FakeSynthResponse(
                        audio=SynthesizedAudio(wav_bytes=b"", sample_rate=24_000),
                    ),
                ],
            ),
        ),
    )

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/synthesize",
            json={"text": "本文", "caption": "声の説明。"},
        )

    assert response.status_code == status.HTTP_200_OK
    payload = response.json()
    assert payload["segment_index"] == 0
    assert "wav_bytes" in payload
    assert payload["wav_bytes"] is not None
    assert isinstance(payload["wav_bytes"], str)
    assert not payload["wav_bytes"]
    assert payload["content_type"] == "audio/wav"
    result = SynthesisResult.model_validate_json(response.text)
    assert result.segment_index == 0
    assert result.wav_bytes == b""
    assert result.elapsed_seconds >= 0.0


def test_synthesize_validation_error_returns_422(
    pipeline_factory: Callable[..., SynthesisPipeline],
) -> None:
    app = create_app(pipeline_factory())

    with TestClient(app) as client:
        response = client.post(
            "/synthesize",
            json={"text": " ", "caption": "声の説明。"},
        )

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
    _assert_validation_error_mentions(response, "text", "blank")


def test_synthesize_maps_backend_unavailable_to_503(
    pipeline_factory: Callable[..., SynthesisPipeline],
) -> None:
    app = create_app(
        pipeline_factory(
            FakeSynthesizer(
                responses=[FakeSynthResponse(exception=RuntimeError("backend offline"))],
            ),
        ),
    )

    with TestClient(app) as client:
        response = client.post(
            "/synthesize",
            json={"text": "本文", "caption": "声の説明。"},
        )

    assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    assert response.json()["code"] == "backend_unavailable"


def test_synthesize_maps_backpressure_to_429(
    pipeline_factory: Callable[..., SynthesisPipeline],
) -> None:
    synthesizer = BlockingSynthesizer()
    app = create_app(
        pipeline_factory(
            synthesizer,
            config=PipelineConfig(acquire_timeout_seconds=0),
        ),
    )

    with TestClient(app) as client:
        first_response: dict[str, Response] = {}

        def post_first() -> None:
            first_response["response"] = client.post(
                "/synthesize",
                json={"text": "一つ目", "caption": "声の説明。"},
            )

        worker = threading.Thread(target=post_first)
        worker.start()
        assert _wait_for_event(synthesizer.entered)

        second = client.post(
            "/synthesize",
            json={"text": "二つ目", "caption": "声の説明。"},
        )

        synthesizer.release.set()
        worker.join(timeout=TEST_EVENT_TIMEOUT)
        assert not worker.is_alive()

    assert second.status_code == status.HTTP_429_TOO_MANY_REQUESTS
    assert second.json()["code"] == "backpressure"
    assert first_response["response"].status_code == status.HTTP_200_OK


def test_synthesize_batch_returns_ordered_results(
    pipeline_factory: Callable[..., SynthesisPipeline],
) -> None:
    app = create_app(
        pipeline_factory(
            FakeSynthesizer(
                responses=[
                    FakeSynthResponse(
                        audio=SynthesizedAudio(wav_bytes=b"RIFFzero", sample_rate=24_000),
                    ),
                    FakeSynthResponse(
                        audio=SynthesizedAudio(wav_bytes=b"RIFFone", sample_rate=24_000),
                    ),
                ],
            ),
        ),
    )

    with TestClient(app) as client:
        response = client.post(
            "/synthesize_batch",
            json={
                "segments": [
                    {"segment_index": 0, "text": "一つ目", "caption": "声の説明。"},
                    {"segment_index": 1, "text": "二つ目", "caption": "別の声。"},
                ],
            },
        )

    assert response.status_code == status.HTTP_200_OK
    result = BatchSynthesisResult.model_validate_json(response.text)
    assert [item.segment_index for item in result.results] == [0, 1]
    assert [item.wav_bytes for item in result.results] == [b"RIFFzero", b"RIFFone"]


def test_synthesize_batch_returns_200_for_empty_wav_bytes(
    pipeline_factory: Callable[..., SynthesisPipeline],
) -> None:
    app = create_app(
        pipeline_factory(
            FakeSynthesizer(
                responses=[
                    FakeSynthResponse(
                        audio=SynthesizedAudio(wav_bytes=b"", sample_rate=24_000),
                    ),
                ],
            ),
        ),
    )

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/synthesize_batch",
            json={
                "segments": [
                    {"segment_index": 0, "text": "一つ目", "caption": "声の説明。"},
                ],
            },
        )

    assert response.status_code == status.HTTP_200_OK
    payload = response.json()
    assert payload["results"][0]["segment_index"] == 0
    assert "wav_bytes" in payload["results"][0]
    assert payload["results"][0]["wav_bytes"] is not None
    assert isinstance(payload["results"][0]["wav_bytes"], str)
    assert not payload["results"][0]["wav_bytes"]
    assert payload["results"][0]["content_type"] == "audio/wav"
    result = BatchSynthesisResult.model_validate_json(response.text)
    assert len(result.results) == 1
    assert result.results[0].segment_index == 0
    assert result.results[0].wav_bytes == b""
    assert result.results[0].elapsed_seconds >= 0.0


def test_synthesize_batch_rejects_unordered_segment_indices(
    pipeline_factory: Callable[..., SynthesisPipeline],
) -> None:
    app = create_app(pipeline_factory())

    with TestClient(app) as client:
        response = client.post(
            "/synthesize_batch",
            json={
                "segments": [
                    {"segment_index": 1, "text": "二つ目", "caption": "声の説明。"},
                    {"segment_index": 0, "text": "一つ目", "caption": "声の説明。"},
                ],
            },
        )

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
    assert response.json()["detail"] == "segments must be ordered by segment_index starting at 0"


def test_synthesize_batch_validation_error_returns_422(
    pipeline_factory: Callable[..., SynthesisPipeline],
) -> None:
    app = create_app(pipeline_factory())

    with TestClient(app) as client:
        response = client.post(
            "/synthesize_batch",
            json={
                "segments": [
                    {"segment_index": 0, "text": " ", "caption": "声の説明。"},
                ],
            },
        )

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
    _assert_validation_error_mentions(response, "text", "blank")


def test_synthesize_returns_500_when_pipeline_missing(
    pipeline_factory: Callable[..., SynthesisPipeline],
) -> None:
    app = create_app(pipeline_factory())
    app.state.pipeline = None

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/synthesize",
            json={"text": "本文", "caption": "声の説明。"},
        )

    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert response.json() == {"detail": "Synthesis pipeline is not configured"}


def test_synthesize_stream_preserves_bytes_and_segment_order(
    pipeline_factory: Callable[..., SynthesisPipeline],
) -> None:
    app = create_app(
        pipeline_factory(
            FakeSynthesizer(
                responses=[
                    FakeSynthResponse(
                        audio=SynthesizedAudio(wav_bytes=b"RIFFzero", sample_rate=24_000),
                    ),
                    FakeSynthResponse(
                        audio=SynthesizedAudio(wav_bytes=b"RIFFone", sample_rate=24_000),
                    ),
                ],
            ),
        ),
    )
    app.state.max_chunk_size = SPLIT_STREAM_CHUNK_SIZE

    with TestClient(app) as client:
        response = client.post(
            "/synthesize_stream",
            json={
                "segments": [
                    {"segment_index": 0, "text": "一つ目", "caption": "声の説明。"},
                    {"segment_index": 1, "text": "二つ目", "caption": "別の声。"},
                ],
            },
        )

    assert response.status_code == status.HTTP_200_OK
    handshake, chunks = _parse_stream(response.content)
    assert handshake.max_chunk_size == SPLIT_STREAM_CHUNK_SIZE
    assert [header.segment_index for header, _ in chunks] == [0, 0, 1, 1]
    assert [payload for _, payload in chunks] == [b"RIFF", b"zero", b"RIFF", b"one"]
    assert [header.final for header, _ in chunks] == [False, True, False, True]
    assert b"".join(payload for _, payload in chunks) == b"RIFFzeroRIFFone"


def test_synthesize_stream_emits_terminal_header_for_empty_wav_bytes(
    pipeline_factory: Callable[..., SynthesisPipeline],
) -> None:
    app = create_app(
        pipeline_factory(
            FakeSynthesizer(
                responses=[
                    FakeSynthResponse(
                        audio=SynthesizedAudio(wav_bytes=b"", sample_rate=24_000),
                    ),
                ],
            ),
        ),
    )

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/synthesize_stream",
            json={
                "segments": [
                    {"segment_index": 0, "text": "一つ目", "caption": "声の説明。"},
                ],
            },
        )

    assert response.status_code == status.HTTP_200_OK
    _, chunks = _parse_stream(response.content)
    assert len(chunks) == 1
    header, payload = chunks[0]
    assert header.segment_index == 0
    assert header.byte_length == 0
    assert header.final is True
    assert header.elapsed_seconds >= 0.0
    assert payload == b""


@pytest.mark.parametrize(
    ("max_chunk_size", "expected_detail"),
    [
        (0, "max_chunk_size must be a positive integer"),
        (
            MAX_CHUNK_SIZE_BYTES + 1,
            f"max_chunk_size must be less than or equal to {MAX_CHUNK_SIZE_BYTES} bytes",
        ),
    ],
)
def test_synthesize_stream_rejects_invalid_max_chunk_size(
    pipeline_factory: Callable[..., SynthesisPipeline],
    max_chunk_size: int,
    expected_detail: str,
) -> None:
    app = create_app(pipeline_factory())
    app.state.max_chunk_size = max_chunk_size

    with TestClient(app) as client:
        response = client.post(
            "/synthesize_stream",
            json={
                "segments": [
                    {"segment_index": 0, "text": "一つ目", "caption": "声の説明。"},
                ],
            },
        )

    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert response.json()["detail"] == expected_detail


def test_synthesize_stream_invalid_unordered_segment_index(
    pipeline_factory: Callable[..., SynthesisPipeline],
) -> None:
    app = create_app(pipeline_factory())

    with TestClient(app) as client:
        response = client.post(
            "/synthesize_stream",
            json={
                "segments": [
                    {"segment_index": 1, "text": "二つ目", "caption": "声の説明。"},
                    {"segment_index": 0, "text": "一つ目", "caption": "声の説明。"},
                ],
            },
        )

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
    assert response.json()["detail"] == "segments must be ordered by segment_index starting at 0"


def test_synthesize_stream_invalid_blank_text(
    pipeline_factory: Callable[..., SynthesisPipeline],
) -> None:
    app = create_app(pipeline_factory())

    with TestClient(app) as client:
        response = client.post(
            "/synthesize_stream",
            json={
                "segments": [
                    {"segment_index": 0, "text": " ", "caption": "声の説明。"},
                ],
            },
        )

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
    _assert_validation_error_mentions(response, "text", "text fields must not be blank")


def _parse_stream(
    data: bytes,
) -> tuple[StreamHandshakeHeader, list[tuple[StreamChunkHeader, bytes]]]:
    position = 0
    line_end = data.index(b"\n", position) + 1
    handshake = StreamHandshakeHeader.from_bytes(data[position:line_end])
    position = line_end
    chunks: list[tuple[StreamChunkHeader, bytes]] = []

    while position < len(data):
        line_end = data.index(b"\n", position) + 1
        header = StreamChunkHeader.from_bytes(data[position:line_end])
        position = line_end
        payload = data[position : position + header.byte_length]
        assert len(payload) == header.byte_length
        chunks.append((header, payload))
        position += header.byte_length

    return handshake, chunks


def _wait_for_event(event: threading.Event) -> bool:
    return event.wait(timeout=TEST_EVENT_TIMEOUT)


def _assert_validation_error_mentions(response: Response, field: str, message: str) -> None:
    detail = response.json()["detail"]
    assert isinstance(detail, list)
    assert any(field in entry["loc"] for entry in detail)
    assert any(message in entry["msg"].lower() for entry in detail)
