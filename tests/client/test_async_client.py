from __future__ import annotations

from typing import TYPE_CHECKING

import httpx
import pytest

from irodori_tts_infra.client.async_ import AsyncIrodoriClient
from irodori_tts_infra.client.errors import (
    ClientBackpressureError,
    ClientError,
    ClientTimeoutError,
    ClientUnavailableError,
)
from irodori_tts_infra.contracts import (
    BatchSynthesisRequest,
    BatchSynthesisResult,
    ErrorPayload,
    HealthResponse,
    StreamChunkHeader,
    StreamHandshakeHeader,
    SynthesisRequest,
    SynthesisResult,
    SynthesisSegment,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from pydantic import BaseModel

pytestmark = pytest.mark.unit

BASE_URL = "http://irodori.test"
MAX_TEST_CHUNK_SIZE = 4
SERVER_ERROR_STATUS = 500


def _client(handler: httpx.MockTransport) -> AsyncIrodoriClient:
    return AsyncIrodoriClient(base_url=BASE_URL, transport=handler)


def _json_response(model: BaseModel, status_code: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code,
        json=model.model_dump(mode="json"),
    )


def _framed(payloads: list[bytes], *, handshake: bool = True) -> bytes:
    frames = []
    if handshake:
        frames.append(StreamHandshakeHeader(max_chunk_size=MAX_TEST_CHUNK_SIZE).to_bytes())
    for index, payload in enumerate(payloads):
        frames.extend(
            (
                StreamChunkHeader(
                    segment_index=index,
                    byte_length=len(payload),
                    final=index == len(payloads) - 1,
                ).to_bytes(),
                payload,
            )
        )
    return b"".join(frames)


async def _collect(chunks: AsyncIterator[bytes]) -> list[bytes]:
    return [chunk async for chunk in chunks]


@pytest.mark.asyncio
async def test_health_returns_contract_from_get_health() -> None:
    health = HealthResponse(status="ok", model_loaded=True, max_chunk_size=MAX_TEST_CHUNK_SIZE)

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/health"
        return _json_response(health)

    assert await _client(httpx.MockTransport(handler)).health() == health


@pytest.mark.asyncio
async def test_synthesize_posts_request_and_returns_result() -> None:
    synthesis_request = SynthesisRequest(text="こんにちは", caption="女性が話している。")
    synthesis_result = SynthesisResult(
        segment_index=0,
        wav_bytes=b"RIFF-single",
        elapsed_seconds=0.25,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/synthesize"
        assert SynthesisRequest.model_validate_json(request.content) == synthesis_request
        return _json_response(synthesis_result)

    result = await _client(httpx.MockTransport(handler)).synthesize(synthesis_request)

    assert result == synthesis_result


@pytest.mark.asyncio
async def test_synthesize_batch_posts_segments_and_returns_ordered_results() -> None:
    batch_request = BatchSynthesisRequest(
        segments=[
            SynthesisSegment(segment_index=0, text="地の文です。", caption="女性が読んでいる。"),
            SynthesisSegment(segment_index=1, text="台詞です。", caption="男性が話している。"),
        ]
    )
    batch_result = BatchSynthesisResult(
        results=[
            SynthesisResult(segment_index=0, wav_bytes=b"RIFF-first", elapsed_seconds=0.1),
            SynthesisResult(segment_index=1, wav_bytes=b"RIFF-second", elapsed_seconds=0.2),
        ],
        total_elapsed_seconds=0.3,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/synthesize_batch"
        assert BatchSynthesisRequest.model_validate_json(request.content) == batch_request
        return _json_response(batch_result)

    result = await _client(httpx.MockTransport(handler)).synthesize_batch(batch_request)

    assert result == batch_result


@pytest.mark.asyncio
async def test_synthesize_stream_reconstructs_byte_exact_payload_across_three_chunks() -> None:
    synthesis_request = SynthesisRequest(text="長い本文です。", caption="女性が読んでいる。")
    payloads = [b"RI", b"FF", b"-wav"]
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        if request.url.path == "/health":
            return _json_response(HealthResponse(max_chunk_size=MAX_TEST_CHUNK_SIZE))
        assert SynthesisRequest.model_validate_json(request.content) == synthesis_request
        return httpx.Response(200, content=_framed(payloads))

    stream = _client(httpx.MockTransport(handler)).synthesize_stream(synthesis_request)
    chunks = await _collect(stream)

    assert paths == ["/health", "/synthesize_stream"]
    assert chunks == payloads
    assert b"".join(chunks) == b"RIFF-wav"


@pytest.mark.asyncio
async def test_synthesize_stream_accepts_missing_handshake_and_boundary_lengths() -> None:
    synthesis_request = SynthesisRequest(text="境界値です。", caption="女性が読んでいる。")
    payloads = [b"", b"abcd"]

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return _json_response(HealthResponse(max_chunk_size=MAX_TEST_CHUNK_SIZE))
        return httpx.Response(200, content=_framed(payloads, handshake=False))

    stream = _client(httpx.MockTransport(handler)).synthesize_stream(synthesis_request)
    chunks = await _collect(stream)

    assert chunks == payloads
    assert b"".join(chunks) == b"abcd"


@pytest.mark.parametrize(
    ("stream", "match"),
    [
        (
            StreamHandshakeHeader(max_chunk_size=MAX_TEST_CHUNK_SIZE).to_bytes()
            + StreamHandshakeHeader(max_chunk_size=MAX_TEST_CHUNK_SIZE).to_bytes(),
            "duplicate handshake",
        ),
        (
            StreamChunkHeader(segment_index=0, byte_length=0, final=True).to_bytes()
            + StreamChunkHeader(segment_index=1, byte_length=0, final=True).to_bytes(),
            "frame after final",
        ),
        (
            StreamChunkHeader(segment_index=0, byte_length=0).to_bytes()
            + StreamHandshakeHeader(max_chunk_size=MAX_TEST_CHUNK_SIZE).to_bytes(),
            "handshake after payload",
        ),
        (
            StreamHandshakeHeader(max_chunk_size=MAX_TEST_CHUNK_SIZE + 1).to_bytes(),
            "exceeds health",
        ),
        (
            StreamChunkHeader(segment_index=0, byte_length=MAX_TEST_CHUNK_SIZE + 1).to_bytes()
            + b"abcde",
            "exceeds stream cap",
        ),
        (
            StreamChunkHeader(segment_index=1, byte_length=0).to_bytes(),
            "segment_index",
        ),
    ],
)
@pytest.mark.asyncio
async def test_synthesize_stream_rejects_protocol_errors(stream: bytes, match: str) -> None:
    synthesis_request = SynthesisRequest(text="異常系です。", caption="女性が読んでいる。")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return _json_response(HealthResponse(max_chunk_size=MAX_TEST_CHUNK_SIZE))
        return httpx.Response(200, content=stream)

    with pytest.raises(ClientError, match=match):
        await _collect(_client(httpx.MockTransport(handler)).synthesize_stream(synthesis_request))


@pytest.mark.parametrize(
    ("status_code", "expected_error"),
    [
        (400, ClientError),
        (408, ClientTimeoutError),
        (429, ClientBackpressureError),
        (503, ClientUnavailableError),
        (SERVER_ERROR_STATUS, ClientUnavailableError),
    ],
)
@pytest.mark.asyncio
async def test_error_responses_map_to_typed_client_errors(
    status_code: int,
    expected_error: type[ClientError],
) -> None:
    error_payload = ErrorPayload(
        code="server_busy",
        message="server cannot accept work",
        details={"retry_after": 2},
    )

    def handler(_request: httpx.Request) -> httpx.Response:
        return _json_response(error_payload, status_code=status_code)

    with pytest.raises(expected_error) as raised:
        await _client(httpx.MockTransport(handler)).health()

    assert raised.value.status_code == status_code
    assert raised.value.code == error_payload.code
    assert raised.value.details == error_payload.details


@pytest.mark.asyncio
async def test_transport_timeout_maps_to_client_timeout_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        message = "timed out"
        raise httpx.TimeoutException(message, request=request)

    with pytest.raises(ClientTimeoutError, match="timed out"):
        await _client(httpx.MockTransport(handler)).health()


@pytest.mark.asyncio
async def test_transport_error_maps_to_client_unavailable_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        message = "connection failed"
        raise httpx.ConnectError(message, request=request)

    with pytest.raises(ClientUnavailableError, match="connection failed"):
        await _client(httpx.MockTransport(handler)).health()


@pytest.mark.asyncio
async def test_async_client_closes_owned_httpx_client() -> None:
    transport = httpx.MockTransport(lambda _request: _json_response(HealthResponse()))

    async with _client(transport) as client:
        assert (await client.health()).status == "ok"

    with pytest.raises(RuntimeError, match="closed"):
        await client.health()
