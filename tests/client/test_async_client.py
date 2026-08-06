from __future__ import annotations

import asyncio
import gzip
import zlib
from typing import TYPE_CHECKING, Literal, cast

import httpx
import pytest
from typing_extensions import override

from irodori_tts_infra.client import async_ as async_client
from irodori_tts_infra.client.async_ import AsyncIrodoriClient
from irodori_tts_infra.client.errors import (
    ClientBackpressureError,
    ClientError,
    ClientTimeoutError,
    ClientUnavailableError,
)
from irodori_tts_infra.config import ClientSettings
from irodori_tts_infra.contracts import (
    BatchSynthesisRequest,
    BatchSynthesisResult,
    CapabilitiesResponse,
    ErrorPayload,
    HealthResponse,
    StreamChunkHeader,
    StreamHandshakeHeader,
    SynthesisRequest,
    SynthesisResult,
    SynthesisSegment,
    VoiceCapability,
)
from irodori_tts_infra.engine.backends.fake import FakeSynthesizer, FakeSynthResponse
from irodori_tts_infra.engine.models import SynthesizedAudio
from irodori_tts_infra.engine.pipeline import SynthesisPipeline
from irodori_tts_infra.server.app import create_app
from irodori_tts_infra.voice_bank import CharacterVoice, SpeakerEmbeddingProfile, VoiceProfile

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable

    from pydantic import BaseModel

pytestmark = pytest.mark.unit

BASE_URL = "http://irodori.test"
MAX_TEST_CHUNK_SIZE = 4
MAX_TEST_HEADER_BYTES = 4_096
MAX_ONE_BYTE_RESPONSE_ENCODED_BYTES = 65
RAW_RESPONSE_TEST_CHUNK_BYTES = 64 * 1024
LARGE_GZIP_EXTRA_BYTES = 65_500
SERVER_ERROR_STATUS = 500
TERMINAL_STREAM_ERROR_CASES = [
    (
        "backend_unavailable",
        ClientUnavailableError,
        503,
        "synthesis backend is unavailable",
    ),
    (
        "backpressure",
        ClientBackpressureError,
        429,
        "synthesis request was rejected by backpressure",
    ),
    ("voice_not_found", ClientError, 404, "requested voice was not found"),
    (
        "runtime_generation_mismatch",
        ClientError,
        409,
        "runtime generation does not match request",
    ),
]


def _client(handler: httpx.MockTransport) -> AsyncIrodoriClient:
    return AsyncIrodoriClient(base_url=BASE_URL, transport=handler)


def _json_response(model: BaseModel, status_code: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code,
        json=model.model_dump(mode="json"),
    )


def _framed(
    payloads: list[bytes],
    *,
    handshake: bool = True,
    max_chunk_size: int = MAX_TEST_CHUNK_SIZE,
) -> bytes:
    frames = []
    if handshake:
        frames.append(StreamHandshakeHeader(max_chunk_size=max_chunk_size).to_bytes())
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


def _terminal_error_framed(
    error_code: Literal[
        "backend_unavailable",
        "backpressure",
        "voice_not_found",
        "runtime_generation_mismatch",
    ],
) -> bytes:
    return (
        StreamHandshakeHeader(max_chunk_size=MAX_TEST_CHUNK_SIZE).to_bytes()
        + StreamChunkHeader(
            segment_index=0,
            byte_length=0,
            final=True,
            error_code=error_code,
        ).to_bytes()
    )


def _raw_deflate_with_zlib_header_collision(decoded: bytes) -> bytes:
    first_block = decoded[:29]
    final_block = decoded[29:]

    def stored_block(header: int, payload: bytes) -> bytes:
        length = len(payload)
        return b"".join(
            (
                bytes([header]),
                length.to_bytes(2, "little"),
                (length ^ 0xFFFF).to_bytes(2, "little"),
                payload,
            )
        )

    encoded = stored_block(0x08, first_block) + stored_block(0x01, final_block)
    assert encoded.startswith(b"\x08\x1d")
    return encoded


def _empty_raw_deflate_blocks(count: int) -> bytes:
    assert count > 0
    nonfinal = b"\x00\x00\x00\xff\xff"
    final = b"\x01\x00\x00\xff\xff"
    return nonfinal * (count - 1) + final


def _gzip_member(payload: bytes = b"", *, extra: bytes = b"") -> bytes:
    encoded = gzip.compress(payload)
    if not extra:
        return encoded
    header = encoded[:3] + b"\x04" + encoded[4:10]
    return header + len(extra).to_bytes(2, "little") + extra + encoded[10:]


async def _read_bounded_response_for_test(
    response: httpx.Response,
    *,
    max_bytes: int,
) -> bytes:
    reader = cast(
        "Callable[..., Awaitable[bytes]]",
        vars(async_client)["_read_bounded_response"],
    )
    return await reader(response, max_bytes=max_bytes, endpoint="/health")


async def _collect(chunks: AsyncIterator[bytes]) -> list[bytes]:
    return [chunk async for chunk in chunks]


async def _collect_into(target: list[bytes], chunks: AsyncIterator[bytes]) -> None:
    target.extend([chunk async for chunk in chunks])


class _GatedAsyncByteStream(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks
        self._gates = [asyncio.Event() for _chunk in chunks]
        self.yielded = 0

    def release(self, index: int) -> None:
        self._gates[index].set()

    @override
    async def __aiter__(self) -> AsyncIterator[bytes]:
        for index, chunk in enumerate(self._chunks):
            await self._gates[index].wait()
            self.yielded += 1
            yield chunk


class _ChunkedAsyncByteStream(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks
        self.yielded = 0

    @override
    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            self.yielded += 1
            yield chunk


class _FailingAsyncByteStream(httpx.AsyncByteStream):
    @override
    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield b'{"status"'
        message = "response read failed"
        raise httpx.ReadError(message)


class _CloseTrackingAsyncByteStream(_ChunkedAsyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        super().__init__(chunks)
        self.closed = False

    @override
    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_health_returns_contract_from_get_health() -> None:
    health = HealthResponse(status="ok", model_loaded=True, max_chunk_size=MAX_TEST_CHUNK_SIZE)

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/health"
        return _json_response(health)

    assert await _client(httpx.MockTransport(handler)).health() == health


@pytest.mark.parametrize("count", [0, 3])
@pytest.mark.asyncio
async def test_capabilities_returns_strict_contract_from_get_capabilities(count: int) -> None:
    capabilities = CapabilitiesResponse(
        generation="fixture-generation",
        ready=True,
        readiness="ready",
        voices=tuple(
            VoiceCapability(id=f"fixture-{index}", label=f"Fixture {index}")
            for index in range(count)
        ),
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/capabilities"
        return _json_response(capabilities)

    result = await _client(httpx.MockTransport(handler)).capabilities()

    assert result == capabilities


@pytest.mark.asyncio
async def test_default_base_url_uses_client_settings() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        settings = ClientSettings()
        assert request.url == httpx.URL(f"http://{settings.host}:{settings.port}/health")
        return _json_response(HealthResponse())

    client = AsyncIrodoriClient(transport=httpx.MockTransport(handler))

    assert (await client.health()).status == "ok"


@pytest.mark.asyncio
async def test_synthesize_posts_request_and_returns_result() -> None:
    synthesis_request = SynthesisRequest(text="こんにちは")
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
async def test_nonstreaming_response_rejects_oversized_content_length() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-length": "5"},
            stream=_ChunkedAsyncByteStream([b"12345"]),
        )

    client = AsyncIrodoriClient(
        base_url=BASE_URL,
        transport=httpx.MockTransport(handler),
        max_response_bytes=4,
    )

    with pytest.raises(ClientError, match="response") as raised:
        await client.health()

    assert raised.value.code == "response_too_large"


@pytest.mark.asyncio
async def test_nonstreaming_response_rejects_oversized_chunked_body() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            stream=_ChunkedAsyncByteStream([b"12", b"345"]),
        )

    client = AsyncIrodoriClient(
        base_url=BASE_URL,
        transport=httpx.MockTransport(handler),
        max_response_bytes=4,
    )

    with pytest.raises(ClientError, match="response") as raised:
        await client.health()

    assert raised.value.code == "response_too_large"


@pytest.mark.asyncio
async def test_nonstreaming_response_preserves_decoded_compressed_content() -> None:
    health = HealthResponse(status="ok", model_loaded=True)
    decoded = health.model_dump_json().encode()
    compressed = gzip.compress(decoded)
    assert len(compressed) > len(decoded)

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-encoding": "gzip"},
            stream=_ChunkedAsyncByteStream([compressed]),
        )

    client = AsyncIrodoriClient(
        base_url=BASE_URL,
        transport=httpx.MockTransport(handler),
        max_response_bytes=len(decoded),
    )

    assert await client.health() == health


@pytest.mark.asyncio
async def test_nonstreaming_response_accepts_raw_deflate_with_zlib_header_collision() -> None:
    health = HealthResponse(status="ok", model_loaded=True)
    decoded = health.model_dump_json().encode()
    encoded = _raw_deflate_with_zlib_header_collision(decoded)

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-encoding": "deflate"},
            stream=_ChunkedAsyncByteStream([encoded]),
        )

    client = AsyncIrodoriClient(
        base_url=BASE_URL,
        transport=httpx.MockTransport(handler),
        max_response_bytes=len(decoded),
    )

    assert await client.health() == health


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "encoded",
    [
        gzip.compress(b"{}")[:-1],
        gzip.compress(b"{}") + b"trailing",
        gzip.compress(b"{}") + gzip.compress(b"{}")[:-1],
    ],
    ids=["truncated", "trailing", "truncated-second-member"],
)
async def test_nonstreaming_response_rejects_invalid_gzip_framing(encoded: bytes) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-encoding": "gzip"},
            stream=_ChunkedAsyncByteStream([encoded]),
        )

    client = AsyncIrodoriClient(base_url=BASE_URL, transport=httpx.MockTransport(handler))

    with pytest.raises(ClientError, match="invalid compressed content") as raised:
        await client.health()

    assert raised.value.code == "protocol_error"
    assert raised.value.endpoint == "/health"


@pytest.mark.asyncio
async def test_nonstreaming_response_accepts_chunked_multiple_member_gzip() -> None:
    health = HealthResponse(status="ok", model_loaded=True)
    decoded = health.model_dump_json().encode()
    split = len(decoded) // 2
    first_member = gzip.compress(decoded[:split])
    second_member = gzip.compress(decoded[split:])
    encoded = first_member + second_member

    def handler(_request: httpx.Request) -> httpx.Response:
        boundary = len(first_member) + 3
        return httpx.Response(
            200,
            headers={"content-encoding": "gzip"},
            stream=_ChunkedAsyncByteStream([encoded[:boundary], encoded[boundary:]]),
        )

    client = AsyncIrodoriClient(
        base_url=BASE_URL,
        transport=httpx.MockTransport(handler),
        max_response_bytes=len(decoded),
    )

    assert await client.health() == health


@pytest.mark.asyncio
async def test_nonstreaming_response_incrementally_decodes_multiple_member_gzip() -> None:
    health = HealthResponse(status="ok", model_loaded=True)
    decoded = health.model_dump_json().encode()
    split = len(decoded) // 2
    encoded = _gzip_member(
        decoded[:split],
        extra=b"x" * LARGE_GZIP_EXTRA_BYTES,
    ) + _gzip_member(decoded[split:])
    assert len(encoded) > RAW_RESPONSE_TEST_CHUNK_BYTES

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-encoding": "gzip"},
            stream=_ChunkedAsyncByteStream([encoded]),
        )

    client = AsyncIrodoriClient(base_url=BASE_URL, transport=httpx.MockTransport(handler))

    assert await client.health() == health


@pytest.mark.asyncio
async def test_nonstreaming_multiple_member_gzip_enforces_total_decoded_limit() -> None:
    encoded = gzip.compress(b"x" * 600) + gzip.compress(b"x" * 600)

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-encoding": "gzip"},
            stream=_ChunkedAsyncByteStream([encoded]),
        )

    client = AsyncIrodoriClient(
        base_url=BASE_URL,
        transport=httpx.MockTransport(handler),
        max_response_bytes=1_024,
    )

    with pytest.raises(ClientError, match="response") as raised:
        await client.health()

    assert raised.value.code == "response_too_large"
    assert raised.value.endpoint == "/health"


@pytest.mark.asyncio
async def test_nonstreaming_response_maps_iteration_read_error_to_transport_error() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=_FailingAsyncByteStream())

    client = AsyncIrodoriClient(base_url=BASE_URL, transport=httpx.MockTransport(handler))

    with pytest.raises(ClientUnavailableError) as raised:
        await client.health()

    assert raised.value.code == "transport_error"
    assert raised.value.endpoint == "/health"


@pytest.mark.asyncio
async def test_nonstreaming_response_accepts_preconsumed_decoded_gzip_content() -> None:
    health = HealthResponse(status="ok", model_loaded=True)
    compressed = gzip.compress(health.model_dump_json().encode())

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-encoding": "gzip"},
            content=compressed,
        )

    client = AsyncIrodoriClient(base_url=BASE_URL, transport=httpx.MockTransport(handler))

    assert await client.health() == health


@pytest.mark.asyncio
async def test_preconsumed_decoded_gzip_error_preserves_typed_server_error() -> None:
    error_payload = ErrorPayload(
        code="model_not_loaded",
        message="モデルがロードされていません",
        details={"generation": "v4"},
    )
    compressed = gzip.compress(error_payload.model_dump_json().encode())

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            SERVER_ERROR_STATUS,
            headers={"content-encoding": "gzip"},
            content=compressed,
        )

    client = AsyncIrodoriClient(base_url=BASE_URL, transport=httpx.MockTransport(handler))

    with pytest.raises(ClientUnavailableError) as raised:
        await client.health()

    assert raised.value.code == error_payload.code
    assert raised.value.details == error_payload.details
    assert raised.value.endpoint == "/health"


@pytest.mark.asyncio
async def test_preconsumed_decoded_gzip_content_enforces_limit() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-encoding": "gzip"},
            content=gzip.compress(b"x" * (128 * 1024)),
        )

    client = AsyncIrodoriClient(
        base_url=BASE_URL,
        transport=httpx.MockTransport(handler),
        max_response_bytes=1_024,
    )

    with pytest.raises(ClientError, match="response") as raised:
        await client.health()

    assert raised.value.code == "response_too_large"
    assert raised.value.endpoint == "/health"


@pytest.mark.asyncio
@pytest.mark.parametrize("wbits", [zlib.MAX_WBITS, -zlib.MAX_WBITS])
async def test_nonstreaming_response_preserves_bounded_deflate_content(wbits: int) -> None:
    health = HealthResponse(status="ok", model_loaded=True)
    decoded = health.model_dump_json().encode()
    compressor = zlib.compressobj(wbits=wbits)
    compressed = compressor.compress(decoded) + compressor.flush()

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-encoding": "deflate"},
            stream=_ChunkedAsyncByteStream([compressed]),
        )

    client = AsyncIrodoriClient(
        base_url=BASE_URL,
        transport=httpx.MockTransport(handler),
        max_response_bytes=len(decoded),
    )

    assert await client.health() == health


@pytest.mark.asyncio
async def test_nonstreaming_compressed_decode_uses_bounded_output_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decoded = b"x" * (128 * 1024)
    compressed = gzip.compress(decoded)
    max_response_bytes = 1_024
    decode_limits: list[int] = []
    original_decompressobj = zlib.decompressobj

    class RecordingDecompressor:
        def __init__(self, wbits: int) -> None:
            self._wrapped = original_decompressobj(wbits)

        @property
        def eof(self) -> bool:
            return self._wrapped.eof

        @property
        def unconsumed_tail(self) -> bytes:
            return self._wrapped.unconsumed_tail

        @property
        def unused_data(self) -> bytes:
            return self._wrapped.unused_data

        def decompress(self, data: bytes, max_length: int = 0) -> bytes:
            decode_limits.append(max_length)
            return self._wrapped.decompress(data, max_length)

        def flush(self) -> bytes:
            return self._wrapped.flush()

    def recording_decompressobj(wbits: int = zlib.MAX_WBITS) -> RecordingDecompressor:
        return RecordingDecompressor(wbits)

    monkeypatch.setattr(zlib, "decompressobj", recording_decompressobj)

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-encoding": "gzip"},
            stream=_ChunkedAsyncByteStream([compressed]),
        )

    client = AsyncIrodoriClient(
        base_url=BASE_URL,
        transport=httpx.MockTransport(handler),
        max_response_bytes=max_response_bytes,
    )

    with pytest.raises(ClientError, match="response") as raised:
        await client.health()

    assert raised.value.code == "response_too_large"
    assert decode_limits
    assert all(0 < limit <= max_response_bytes + 1 for limit in decode_limits)


@pytest.mark.asyncio
async def test_nonstreaming_compressed_content_length_rejects_encoded_body_before_reading() -> None:
    stream = _ChunkedAsyncByteStream([_empty_raw_deflate_blocks(14)])
    response = httpx.Response(
        200,
        headers={"content-encoding": "deflate", "content-length": "70"},
        stream=stream,
        request=httpx.Request("GET", f"{BASE_URL}/health"),
    )

    with pytest.raises(ClientError, match="response") as raised:
        await _read_bounded_response_for_test(response, max_bytes=1)

    assert raised.value.code == "response_too_large"
    assert raised.value.endpoint == "/health"
    assert stream.yielded == 0


@pytest.mark.asyncio
async def test_nonstreaming_deflate_encoded_limit_accepts_exact_chunked_boundary() -> None:
    encoded = _empty_raw_deflate_blocks(13)
    response = httpx.Response(
        200,
        headers={"content-encoding": "deflate"},
        stream=_ChunkedAsyncByteStream([encoded[:64], encoded[64:]]),
        request=httpx.Request("GET", f"{BASE_URL}/health"),
    )

    assert await _read_bounded_response_for_test(response, max_bytes=1) == b""


@pytest.mark.asyncio
async def test_nonstreaming_deflate_encoded_limit_rejects_chunked_empty_block_overflow() -> None:
    encoded = _empty_raw_deflate_blocks(14)
    response = httpx.Response(
        200,
        headers={"content-encoding": "deflate"},
        stream=_ChunkedAsyncByteStream([encoded[:65], encoded[65:]]),
        request=httpx.Request("GET", f"{BASE_URL}/health"),
    )

    with pytest.raises(ClientError, match="response") as raised:
        await _read_bounded_response_for_test(response, max_bytes=1)

    assert raised.value.code == "response_too_large"
    assert raised.value.endpoint == "/health"


@pytest.mark.asyncio
async def test_nonstreaming_multiple_member_gzip_accepts_exact_encoded_boundary() -> None:
    encoded = _gzip_member() * 2 + _gzip_member(extra=b"abc")
    assert len(encoded) == MAX_ONE_BYTE_RESPONSE_ENCODED_BYTES
    response = httpx.Response(
        200,
        headers={"content-encoding": "gzip"},
        stream=_ChunkedAsyncByteStream([encoded[:64], encoded[64:]]),
        request=httpx.Request("GET", f"{BASE_URL}/health"),
    )

    assert await _read_bounded_response_for_test(response, max_bytes=1) == b""


@pytest.mark.asyncio
async def test_nonstreaming_multiple_member_gzip_rejects_encoded_overflow() -> None:
    encoded = _gzip_member() * 2 + _gzip_member(extra=b"abc")
    response = httpx.Response(
        200,
        headers={"content-encoding": "gzip"},
        stream=_ChunkedAsyncByteStream([encoded, _gzip_member()]),
        request=httpx.Request("GET", f"{BASE_URL}/health"),
    )

    with pytest.raises(ClientError, match="response") as raised:
        await _read_bounded_response_for_test(response, max_bytes=1)

    assert raised.value.code == "response_too_large"
    assert raised.value.endpoint == "/health"


@pytest.mark.asyncio
@pytest.mark.parametrize("content_encoding", ["br", "gzip, deflate"])
async def test_nonstreaming_response_rejects_unsupported_content_encoding(
    content_encoding: str,
) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-encoding": content_encoding},
            content=b"encoded",
        )

    client = AsyncIrodoriClient(base_url=BASE_URL, transport=httpx.MockTransport(handler))

    with pytest.raises(ClientError, match="content encoding") as raised:
        await client.health()

    assert raised.value.code == "protocol_error"
    assert raised.value.endpoint == "/health"


@pytest.mark.parametrize("max_response_bytes", [0, -1, True])
def test_nonstreaming_response_limit_must_be_a_positive_integer(
    max_response_bytes: object,
) -> None:
    with pytest.raises(ValueError, match="max_response_bytes"):
        AsyncIrodoriClient(
            base_url=BASE_URL,
            max_response_bytes=max_response_bytes,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("max_stream_frames", [0, -1, True])
def test_stream_frame_limit_must_be_a_positive_integer(max_stream_frames: object) -> None:
    with pytest.raises(ValueError, match="max_stream_frames"):
        AsyncIrodoriClient(
            base_url=BASE_URL,
            max_stream_frames=max_stream_frames,  # type: ignore[arg-type]
        )


@pytest.mark.asyncio
async def test_synthesize_batch_posts_segments_and_returns_ordered_results() -> None:
    batch_request = BatchSynthesisRequest(
        segments=[
            SynthesisSegment(
                segment_index=0,
                text="地の文です。",
            ),
            SynthesisSegment(
                segment_index=1,
                text="台詞です。",
            ),
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
    synthesis_request = SynthesisRequest(text="長い本文です。")
    payloads = [b"RI", b"FF", b"-wav"]
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        if request.url.path == "/health":
            assert request.headers["accept-encoding"] == "gzip, deflate"
            return _json_response(HealthResponse(max_chunk_size=MAX_TEST_CHUNK_SIZE))
        assert request.headers["accept-encoding"] == "identity"
        assert SynthesisRequest.model_validate_json(request.content) == synthesis_request
        return httpx.Response(200, content=_framed(payloads))

    stream = _client(httpx.MockTransport(handler)).synthesize_stream(synthesis_request)
    chunks = await _collect(stream)

    assert paths == ["/health", "/synthesize_stream"]
    assert chunks == payloads
    assert b"".join(chunks) == b"RIFF-wav"


@pytest.mark.asyncio
async def test_synthesize_stream_posts_single_request_to_asgi_server() -> None:
    synthesizer = FakeSynthesizer(
        responses=[
            FakeSynthResponse(
                audio=SynthesizedAudio(wav_bytes=b"RIFFstream", sample_rate=24_000),
            ),
        ],
    )
    app = create_app(
        SynthesisPipeline(
            synthesizer,
            VoiceProfile(
                narrator=SpeakerEmbeddingProfile(
                    "speakers/narrator.speaker.safetensors",  # type: ignore[arg-type]
                ),
                characters={
                    "ミカ": CharacterVoice(
                        name="ミカ",
                        speaker=SpeakerEmbeddingProfile(
                            "speakers/mika.speaker.safetensors",  # type: ignore[arg-type]
                        ),
                    ),
                },
            ),
        ),
    )
    transport = httpx.ASGITransport(app=app)

    async with AsyncIrodoriClient(base_url="http://testserver", transport=transport) as client:
        chunks = await _collect(
            client.synthesize_stream(SynthesisRequest(text="本文", speaker="ミカ")),
        )

    assert b"".join(chunks) == b"RIFFstream"
    assert len(synthesizer.calls) == 1
    assert synthesizer.calls[0].ref_embed == "speakers/mika.speaker.safetensors"


@pytest.mark.asyncio
async def test_synthesize_stream_reconstructs_batch_stream_from_asgi_server() -> None:
    synthesizer = FakeSynthesizer(
        responses=[
            FakeSynthResponse(
                audio=SynthesizedAudio(wav_bytes=b"RIFFzero", sample_rate=24_000),
            ),
            FakeSynthResponse(
                audio=SynthesizedAudio(wav_bytes=b"RIFFone", sample_rate=24_000),
            ),
        ],
    )
    app = create_app(
        SynthesisPipeline(
            synthesizer,
            VoiceProfile(
                narrator=SpeakerEmbeddingProfile(
                    "speakers/narrator.speaker.safetensors",  # type: ignore[arg-type]
                ),
                characters={
                    "ミカ": CharacterVoice(
                        name="ミカ",
                        speaker=SpeakerEmbeddingProfile(
                            "speakers/mika.speaker.safetensors",  # type: ignore[arg-type]
                        ),
                    ),
                },
            ),
        ),
    )
    app.state.max_chunk_size = 4
    transport = httpx.ASGITransport(app=app)
    request = BatchSynthesisRequest(
        segments=[
            SynthesisSegment(segment_index=0, text="一つ目"),
            SynthesisSegment(segment_index=1, text="二つ目", speaker="ミカ"),
        ],
    )

    async with AsyncIrodoriClient(base_url="http://testserver", transport=transport) as client:
        chunks = await _collect(client.synthesize_stream(request))

    assert chunks == [b"RIFF", b"zero", b"RIFF", b"one"]
    assert [call.ref_embed for call in synthesizer.calls] == [
        "speakers/narrator.speaker.safetensors",
        "speakers/mika.speaker.safetensors",
    ]


@pytest.mark.asyncio
async def test_synthesize_stream_yields_payload_before_response_completes() -> None:
    synthesis_request = SynthesisRequest(text="長い本文です。")
    first_frame = (
        StreamHandshakeHeader(max_chunk_size=MAX_TEST_CHUNK_SIZE).to_bytes()
        + StreamChunkHeader(segment_index=0, byte_length=2, final=False).to_bytes()
        + b"RI"
    )
    final_frame = StreamChunkHeader(segment_index=1, byte_length=2, final=True).to_bytes() + b"FF"
    stream = _GatedAsyncByteStream([first_frame, final_frame])
    stream.release(0)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return _json_response(HealthResponse(max_chunk_size=MAX_TEST_CHUNK_SIZE))
        return httpx.Response(200, stream=stream)

    chunks = _client(httpx.MockTransport(handler)).synthesize_stream(synthesis_request)

    first_chunk = await asyncio.wait_for(anext(chunks), timeout=0.1)

    assert first_chunk == b"RI"
    assert stream.yielded == 1

    stream.release(1)
    assert await _collect(chunks) == [b"FF"]


@pytest.mark.asyncio
async def test_synthesize_stream_accepts_payload_at_total_byte_boundary() -> None:
    synthesis_request = SynthesisRequest(text="境界値です。")
    health = HealthResponse(max_chunk_size=MAX_TEST_CHUNK_SIZE)
    max_response_bytes = len(health.model_dump_json().encode())
    payloads = [b"x" * MAX_TEST_CHUNK_SIZE] * (max_response_bytes // MAX_TEST_CHUNK_SIZE)
    payloads.append(b"x" * (max_response_bytes % MAX_TEST_CHUNK_SIZE))

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return _json_response(health)
        return httpx.Response(200, content=_framed(payloads))

    client = AsyncIrodoriClient(
        base_url=BASE_URL,
        transport=httpx.MockTransport(handler),
        max_response_bytes=max_response_bytes,
    )
    stream = client.synthesize_stream(synthesis_request)
    chunks = await _collect(stream)

    assert chunks == payloads
    assert len(b"".join(chunks)) == max_response_bytes


@pytest.mark.asyncio
async def test_synthesize_stream_rejects_total_payload_over_limit() -> None:
    synthesis_request = SynthesisRequest(text="上限超過です。")
    health = HealthResponse(max_chunk_size=MAX_TEST_CHUNK_SIZE)
    max_response_bytes = len(health.model_dump_json().encode())
    payloads = [b"x" * MAX_TEST_CHUNK_SIZE] * (max_response_bytes // MAX_TEST_CHUNK_SIZE)
    payloads.append(b"x" * (max_response_bytes % MAX_TEST_CHUNK_SIZE + 1))

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return _json_response(health)
        return httpx.Response(200, content=_framed(payloads))

    client = AsyncIrodoriClient(
        base_url=BASE_URL,
        transport=httpx.MockTransport(handler),
        max_response_bytes=max_response_bytes,
    )
    with pytest.raises(ClientError, match="response") as raised:
        await _collect(client.synthesize_stream(synthesis_request))

    assert raised.value.code == "response_too_large"
    assert raised.value.endpoint == "/synthesize_stream"


@pytest.mark.asyncio
async def test_synthesize_stream_rejects_frame_count_over_limit() -> None:
    synthesis_request = SynthesisRequest(text="空フレーム攻撃です。")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return _json_response(HealthResponse(max_chunk_size=MAX_TEST_CHUNK_SIZE))
        return httpx.Response(200, content=_framed([b"", b"", b""]))

    client = AsyncIrodoriClient(
        base_url=BASE_URL,
        transport=httpx.MockTransport(handler),
        max_stream_frames=2,
    )
    with pytest.raises(ClientError, match="response") as raised:
        await _collect(client.synthesize_stream(synthesis_request))

    assert raised.value.code == "response_too_large"
    assert raised.value.endpoint == "/synthesize_stream"


@pytest.mark.asyncio
async def test_synthesize_stream_accepts_configured_frame_count_at_limit() -> None:
    synthesis_request = SynthesisRequest(text="空フレーム境界です。")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return _json_response(HealthResponse(max_chunk_size=1))
        return httpx.Response(
            200,
            content=_framed([b"x", b"x", b"x"], max_chunk_size=1),
        )

    client = AsyncIrodoriClient(
        base_url=BASE_URL,
        transport=httpx.MockTransport(handler),
        max_stream_frames=3,
    )
    assert await _collect(client.synthesize_stream(synthesis_request)) == [b"x", b"x", b"x"]


@pytest.mark.parametrize(
    ("error_code", "expected_error", "status_code", "message"),
    TERMINAL_STREAM_ERROR_CASES,
)
@pytest.mark.asyncio
async def test_synthesize_stream_raises_typed_terminal_frame_error(
    error_code: Literal[
        "backend_unavailable",
        "backpressure",
        "voice_not_found",
        "runtime_generation_mismatch",
    ],
    expected_error: type[ClientError],
    status_code: int,
    message: str,
) -> None:
    synthesis_request = SynthesisRequest(text="終端エラーです。")
    stream = _CloseTrackingAsyncByteStream([_terminal_error_framed(error_code)])
    yielded: list[bytes] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return _json_response(HealthResponse(max_chunk_size=MAX_TEST_CHUNK_SIZE))
        return httpx.Response(200, stream=stream)

    with pytest.raises(expected_error, match=message) as raised:
        await _collect_into(
            yielded,
            _client(httpx.MockTransport(handler)).synthesize_stream(synthesis_request),
        )

    assert yielded == []
    assert raised.value.status_code == status_code
    assert raised.value.code == error_code
    assert raised.value.details == {}
    assert raised.value.endpoint == "/synthesize_stream"
    assert stream.closed is True


@pytest.mark.asyncio
async def test_synthesize_stream_counts_terminal_error_frame_toward_frame_limit() -> None:
    synthesis_request = SynthesisRequest(text="終端エラー境界です。")
    stream_bytes = b"".join(
        (
            StreamHandshakeHeader(max_chunk_size=MAX_TEST_CHUNK_SIZE).to_bytes(),
            StreamChunkHeader(segment_index=0, byte_length=0).to_bytes(),
            StreamChunkHeader(segment_index=1, byte_length=0).to_bytes(),
            StreamChunkHeader(
                segment_index=2,
                byte_length=0,
                final=True,
                error_code="backend_unavailable",
            ).to_bytes(),
        ),
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return _json_response(HealthResponse(max_chunk_size=MAX_TEST_CHUNK_SIZE))
        return httpx.Response(200, content=stream_bytes)

    client = AsyncIrodoriClient(
        base_url=BASE_URL,
        transport=httpx.MockTransport(handler),
        max_stream_frames=2,
    )
    with pytest.raises(ClientError, match="response") as raised:
        await _collect(client.synthesize_stream(synthesis_request))

    assert raised.value.code == "response_too_large"
    assert raised.value.endpoint == "/synthesize_stream"


@pytest.mark.parametrize(
    ("stream_bytes", "match"),
    [
        (
            StreamChunkHeader(segment_index=0, byte_length=0, final=True).to_bytes(),
            "handshake",
        ),
        (
            StreamHandshakeHeader(max_chunk_size=MAX_TEST_CHUNK_SIZE).to_bytes()
            + StreamChunkHeader(segment_index=0, byte_length=0, final=False).to_bytes(),
            "final",
        ),
    ],
)
@pytest.mark.asyncio
async def test_synthesize_stream_requires_handshake_and_terminal_final(
    stream_bytes: bytes,
    match: str,
) -> None:
    synthesis_request = SynthesisRequest(text="終端検証です。")
    stream = _CloseTrackingAsyncByteStream([stream_bytes])

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return _json_response(HealthResponse(max_chunk_size=MAX_TEST_CHUNK_SIZE))
        return httpx.Response(200, stream=stream)

    with pytest.raises(ClientError, match=match) as raised:
        await _collect(_client(httpx.MockTransport(handler)).synthesize_stream(synthesis_request))

    assert raised.value.code == "protocol_error"
    assert raised.value.endpoint == "/synthesize_stream"
    assert stream.closed is True


@pytest.mark.parametrize(
    ("stream", "match"),
    [
        (
            StreamHandshakeHeader(max_chunk_size=MAX_TEST_CHUNK_SIZE).to_bytes()
            + StreamHandshakeHeader(max_chunk_size=MAX_TEST_CHUNK_SIZE).to_bytes(),
            "duplicate handshake",
        ),
        (
            StreamHandshakeHeader(max_chunk_size=MAX_TEST_CHUNK_SIZE).to_bytes()
            + StreamChunkHeader(segment_index=0, byte_length=0, final=True).to_bytes()
            + StreamChunkHeader(segment_index=1, byte_length=0, final=True).to_bytes(),
            "frame after final",
        ),
        (
            StreamHandshakeHeader(max_chunk_size=MAX_TEST_CHUNK_SIZE).to_bytes()
            + StreamChunkHeader(segment_index=0, byte_length=0).to_bytes()
            + StreamHandshakeHeader(max_chunk_size=MAX_TEST_CHUNK_SIZE).to_bytes(),
            "handshake after payload",
        ),
        (
            StreamHandshakeHeader(max_chunk_size=MAX_TEST_CHUNK_SIZE + 1).to_bytes(),
            "exceeds health",
        ),
        (
            StreamHandshakeHeader(max_chunk_size=MAX_TEST_CHUNK_SIZE).to_bytes()
            + StreamChunkHeader(
                segment_index=0,
                byte_length=MAX_TEST_CHUNK_SIZE + 1,
            ).to_bytes()
            + b"abcde",
            "exceeds stream cap",
        ),
        (
            StreamHandshakeHeader(max_chunk_size=MAX_TEST_CHUNK_SIZE).to_bytes()
            + StreamChunkHeader(segment_index=1, byte_length=0).to_bytes(),
            "segment_index",
        ),
    ],
)
@pytest.mark.asyncio
async def test_synthesize_stream_rejects_protocol_errors(stream: bytes, match: str) -> None:
    synthesis_request = SynthesisRequest(text="異常系です。")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return _json_response(HealthResponse(max_chunk_size=MAX_TEST_CHUNK_SIZE))
        return httpx.Response(200, content=stream)

    with pytest.raises(ClientError, match=match):
        await _collect(_client(httpx.MockTransport(handler)).synthesize_stream(synthesis_request))


@pytest.mark.parametrize(
    ("stream", "match"),
    [
        (b'{"kind":"chunk","v":1,"index":0,"nbytes":0,"final":true}', "separator"),
        (b"{not-json}\n", "malformed header JSON"),
        (b"[]\n", "JSON object"),
        (b'{"kind":"unknown","v":1}\n', "unknown stream header kind"),
        (b'{"v":1}\n', "unknown stream header kind"),
        (b'{"kind":"handshake","v":1,"max_chunk_size":0}\n', "invalid handshake"),
        (
            StreamHandshakeHeader(max_chunk_size=MAX_TEST_CHUNK_SIZE).to_bytes()
            + b'{"kind":"chunk","v":1,"index":0,"nbytes":-1}\n',
            "invalid chunk",
        ),
        (
            StreamHandshakeHeader(max_chunk_size=MAX_TEST_CHUNK_SIZE).to_bytes()
            + b'{"kind":"chunk","v":2,"index":0,"nbytes":0}\n',
            "unknown stream header version",
        ),
        (
            StreamHandshakeHeader(max_chunk_size=MAX_TEST_CHUNK_SIZE).to_bytes()
            + b'{"kind":"chunk","v":1,"index":0,"nbytes":4}\nab',
            "truncated",
        ),
        (b"x" * (MAX_TEST_HEADER_BYTES + 1), "header exceeds"),
        (b"x" * (MAX_TEST_HEADER_BYTES + 1) + b"\n", "header exceeds"),
    ],
)
@pytest.mark.asyncio
async def test_synthesize_stream_rejects_malformed_frames(stream: bytes, match: str) -> None:
    synthesis_request = SynthesisRequest(text="壊れたフレームです。")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return _json_response(HealthResponse(max_chunk_size=MAX_TEST_CHUNK_SIZE))
        return httpx.Response(200, content=stream)

    with pytest.raises(ClientError, match=match):
        await _collect(_client(httpx.MockTransport(handler)).synthesize_stream(synthesis_request))


@pytest.mark.asyncio
async def test_synthesize_stream_rejects_compressed_success_response() -> None:
    synthesis_request = SynthesisRequest(text="圧縮された成功応答です。")
    compressed_stream = gzip.compress(_framed([b"test"]))

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return _json_response(HealthResponse(max_chunk_size=MAX_TEST_CHUNK_SIZE))
        return httpx.Response(
            200,
            headers={"content-encoding": "gzip"},
            content=compressed_stream,
        )

    with pytest.raises(ClientError, match="content encoding"):
        await _collect(_client(httpx.MockTransport(handler)).synthesize_stream(synthesis_request))


@pytest.mark.parametrize(
    ("status_code", "expected_error"),
    [
        (400, ClientError),
        (503, ClientUnavailableError),
    ],
)
@pytest.mark.asyncio
async def test_synthesize_stream_error_responses_map_to_typed_client_errors(
    status_code: int,
    expected_error: type[ClientError],
) -> None:
    synthesis_request = SynthesisRequest(text="異常系です。")
    error_payload = ErrorPayload(
        code="server_busy",
        message="server cannot accept work",
        details={"retry_after": 2},
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return _json_response(HealthResponse(max_chunk_size=MAX_TEST_CHUNK_SIZE))
        return _json_response(error_payload, status_code=status_code)

    with pytest.raises(expected_error, match=error_payload.message) as raised:
        await _collect(_client(httpx.MockTransport(handler)).synthesize_stream(synthesis_request))

    assert raised.value.status_code == status_code
    assert raised.value.code == error_payload.code
    assert raised.value.details == error_payload.details


@pytest.mark.asyncio
@pytest.mark.parametrize("body_mode", ["declared", "chunked", "compressed"])
async def test_synthesize_stream_rejects_oversized_error_responses(body_mode: str) -> None:
    synthesis_request = SynthesisRequest(text="異常系です。")
    oversized_body = b"x" * 1_024

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return _json_response(HealthResponse(max_chunk_size=MAX_TEST_CHUNK_SIZE))
        if body_mode == "chunked":
            return httpx.Response(
                SERVER_ERROR_STATUS,
                stream=_ChunkedAsyncByteStream([b"x" * 64, b"x" * 65]),
            )
        if body_mode == "compressed":
            return httpx.Response(
                SERVER_ERROR_STATUS,
                headers={"content-encoding": "gzip"},
                stream=_ChunkedAsyncByteStream([gzip.compress(oversized_body)]),
            )
        return httpx.Response(SERVER_ERROR_STATUS, content=oversized_body)

    client = AsyncIrodoriClient(
        base_url=BASE_URL,
        transport=httpx.MockTransport(handler),
        max_response_bytes=128,
    )

    with pytest.raises(ClientError, match="response") as raised:
        await _collect(client.synthesize_stream(synthesis_request))

    assert raised.value.code == "response_too_large"
    assert raised.value.endpoint == "/synthesize_stream"


@pytest.mark.parametrize(
    ("error_type", "message", "expected_error"),
    [
        (httpx.TimeoutException, "stream timed out", ClientTimeoutError),
        (httpx.ConnectError, "connection failed", ClientUnavailableError),
    ],
)
@pytest.mark.asyncio
async def test_synthesize_stream_open_failures_map_to_typed_client_errors(
    error_type: type[httpx.TimeoutException | httpx.ConnectError],
    message: str,
    expected_error: type[ClientError],
) -> None:
    synthesis_request = SynthesisRequest(text="異常系です。")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return _json_response(HealthResponse(max_chunk_size=MAX_TEST_CHUNK_SIZE))
        raise error_type(message, request=request)

    with pytest.raises(expected_error, match=message):
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
