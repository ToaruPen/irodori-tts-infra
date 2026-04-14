from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient
from starlette import status

from irodori_tts_infra.contracts import (
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


class BlockingSynthesizer:
    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()

    def synthesize(self, request: SynthesisRequest) -> SynthesizedAudio:
        del request
        self.entered.set()
        assert self.release.wait(timeout=1.0)
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
        assert synthesizer.entered.wait(timeout=1.0)

        second = client.post(
            "/synthesize",
            json={"text": "二つ目", "caption": "声の説明。"},
        )

        synthesizer.release.set()
        worker.join(timeout=1.0)
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
    app.state.max_chunk_size = STREAM_MAX_CHUNK_SIZE

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
    assert handshake.max_chunk_size == STREAM_MAX_CHUNK_SIZE
    assert [header.segment_index for header, _ in chunks] == [0, 1]
    assert [payload for _, payload in chunks] == [b"RIFFzero", b"RIFFone"]
    assert [header.final for header, _ in chunks] == [False, True]
    assert b"".join(payload for _, payload in chunks) == b"RIFFzeroRIFFone"


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
