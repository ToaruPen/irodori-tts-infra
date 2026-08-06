from __future__ import annotations

import gzip
import zlib
from typing import TYPE_CHECKING, Literal

import httpx
import pytest
from typing_extensions import override

from irodori_tts_infra.client._stream import (  # noqa: PLC2701 - parser boundary tests
    MAX_STREAM_HEADER_BYTES,
    _header_kind,
)
from irodori_tts_infra.client.errors import (
    ClientBackpressureError,
    ClientError,
    ClientTimeoutError,
    ClientUnavailableError,
)
from irodori_tts_infra.client.sync import (
    SyncIrodoriClient,
    _read_bounded_response,  # noqa: PLC2701 - bounded reader unit test
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
from tests.client.helpers import (
    MAX_TEST_CHUNK_SIZE,
    TERMINAL_STREAM_ERROR_CASES,
    empty_raw_deflate_blocks,
    gzip_member,
    raw_deflate_with_zlib_header_collision,
    terminal_error_framed,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pydantic import BaseModel

pytestmark = pytest.mark.unit

BASE_URL = "http://irodori.test"
MAX_ONE_BYTE_RESPONSE_ENCODED_BYTES = 65
RAW_RESPONSE_TEST_CHUNK_BYTES = 64 * 1024
LARGE_GZIP_EXTRA_BYTES = 65_500
SERVER_ERROR_STATUS = 500


def _client(handler: httpx.MockTransport) -> SyncIrodoriClient:
    return SyncIrodoriClient(base_url=BASE_URL, transport=handler)


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


def _read_bounded_response_for_test(response: httpx.Response, *, max_bytes: int) -> bytes:
    return _read_bounded_response(response, max_bytes=max_bytes, endpoint="/health")


def _collect_into(target: list[bytes], chunks: Iterator[bytes]) -> None:
    target.extend(chunks)


class _CountingByteStream(httpx.SyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks
        self.yielded = 0

    @override
    def __iter__(self) -> Iterator[bytes]:
        for chunk in self._chunks:
            self.yielded += 1
            yield chunk


class _FailingByteStream(httpx.SyncByteStream):
    @override
    def __iter__(self) -> Iterator[bytes]:
        yield b'{"status"'
        message = "response read failed"
        raise httpx.ReadError(message)


class _CloseTrackingByteStream(_CountingByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        super().__init__(chunks)
        self.closed = False

    @override
    def close(self) -> None:
        self.closed = True


def test_health_returns_contract_from_get_health() -> None:
    health = HealthResponse(status="ok", model_loaded=True, max_chunk_size=MAX_TEST_CHUNK_SIZE)

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/health"
        return _json_response(health)

    assert _client(httpx.MockTransport(handler)).health() == health


@pytest.mark.parametrize("count", [0, 3])
def test_capabilities_returns_strict_contract_from_get_capabilities(count: int) -> None:
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

    assert _client(httpx.MockTransport(handler)).capabilities() == capabilities


def test_synthesize_posts_request_and_returns_result() -> None:
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

    assert _client(httpx.MockTransport(handler)).synthesize(synthesis_request) == synthesis_result


@pytest.mark.parametrize(
    ("content_encoding", "wbits"),
    [("gzip", zlib.MAX_WBITS | 16), ("deflate", zlib.MAX_WBITS), ("deflate", -zlib.MAX_WBITS)],
)
def test_nonstreaming_response_preserves_bounded_compressed_content(
    content_encoding: str,
    wbits: int,
) -> None:
    health = HealthResponse(status="ok", model_loaded=True)
    decoded = health.model_dump_json().encode()
    compressor = zlib.compressobj(wbits=wbits)
    compressed = compressor.compress(decoded) + compressor.flush()

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-encoding": content_encoding},
            stream=_CountingByteStream([compressed]),
        )

    client = SyncIrodoriClient(
        base_url=BASE_URL,
        transport=httpx.MockTransport(handler),
        max_response_bytes=len(decoded),
    )

    assert client.health() == health


def test_nonstreaming_response_accepts_raw_deflate_with_zlib_header_collision() -> None:
    health = HealthResponse(status="ok", model_loaded=True)
    decoded = health.model_dump_json().encode()
    encoded = raw_deflate_with_zlib_header_collision(decoded)

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-encoding": "deflate"},
            stream=_CountingByteStream([encoded]),
        )

    client = SyncIrodoriClient(
        base_url=BASE_URL,
        transport=httpx.MockTransport(handler),
        max_response_bytes=len(decoded),
    )

    assert client.health() == health


@pytest.mark.parametrize(
    "encoded",
    [
        gzip.compress(b"{}")[:-1],
        gzip.compress(b"{}") + b"trailing",
        gzip.compress(b"{}") + gzip.compress(b"{}")[:-1],
    ],
    ids=["truncated", "trailing", "truncated-second-member"],
)
def test_nonstreaming_response_rejects_invalid_gzip_framing(encoded: bytes) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-encoding": "gzip"},
            stream=_CountingByteStream([encoded]),
        )

    client = SyncIrodoriClient(base_url=BASE_URL, transport=httpx.MockTransport(handler))

    with pytest.raises(ClientError, match="invalid compressed content") as raised:
        client.health()

    assert raised.value.code == "protocol_error"
    assert raised.value.endpoint == "/health"


def test_nonstreaming_response_accepts_chunked_multiple_member_gzip() -> None:
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
            stream=_CountingByteStream([encoded[:boundary], encoded[boundary:]]),
        )

    client = SyncIrodoriClient(
        base_url=BASE_URL,
        transport=httpx.MockTransport(handler),
        max_response_bytes=len(decoded),
    )

    assert client.health() == health


def test_nonstreaming_response_incrementally_decodes_multiple_member_gzip() -> None:
    health = HealthResponse(status="ok", model_loaded=True)
    decoded = health.model_dump_json().encode()
    split = len(decoded) // 2
    encoded = gzip_member(
        decoded[:split],
        extra=b"x" * LARGE_GZIP_EXTRA_BYTES,
    ) + gzip_member(decoded[split:])
    assert len(encoded) > RAW_RESPONSE_TEST_CHUNK_BYTES

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-encoding": "gzip"},
            stream=_CountingByteStream([encoded]),
        )

    client = SyncIrodoriClient(base_url=BASE_URL, transport=httpx.MockTransport(handler))

    assert client.health() == health


def test_nonstreaming_multiple_member_gzip_enforces_total_decoded_limit() -> None:
    encoded = gzip.compress(b"x" * 600) + gzip.compress(b"x" * 600)

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-encoding": "gzip"},
            stream=_CountingByteStream([encoded]),
        )

    client = SyncIrodoriClient(
        base_url=BASE_URL,
        transport=httpx.MockTransport(handler),
        max_response_bytes=1_024,
    )

    with pytest.raises(ClientError, match="response") as raised:
        client.health()

    assert raised.value.code == "response_too_large"
    assert raised.value.endpoint == "/health"


def test_nonstreaming_response_maps_iteration_read_error_to_transport_error() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=_FailingByteStream())

    client = SyncIrodoriClient(base_url=BASE_URL, transport=httpx.MockTransport(handler))

    with pytest.raises(ClientUnavailableError) as raised:
        client.health()

    assert raised.value.code == "transport_error"
    assert raised.value.endpoint == "/health"


def test_nonstreaming_response_accepts_preconsumed_decoded_gzip_content() -> None:
    health = HealthResponse(status="ok", model_loaded=True)
    compressed = gzip.compress(health.model_dump_json().encode())

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-encoding": "gzip"},
            content=compressed,
        )

    client = SyncIrodoriClient(base_url=BASE_URL, transport=httpx.MockTransport(handler))

    assert client.health() == health


def test_preconsumed_decoded_gzip_error_preserves_typed_server_error() -> None:
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

    client = SyncIrodoriClient(base_url=BASE_URL, transport=httpx.MockTransport(handler))

    with pytest.raises(ClientUnavailableError) as raised:
        client.health()

    assert raised.value.code == error_payload.code
    assert raised.value.details == error_payload.details
    assert raised.value.endpoint == "/health"


def test_preconsumed_decoded_gzip_content_enforces_limit() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-encoding": "gzip"},
            content=gzip.compress(b"x" * (128 * 1024)),
        )

    client = SyncIrodoriClient(
        base_url=BASE_URL,
        transport=httpx.MockTransport(handler),
        max_response_bytes=1_024,
    )

    with pytest.raises(ClientError, match="response") as raised:
        client.health()

    assert raised.value.code == "response_too_large"
    assert raised.value.endpoint == "/health"


def test_nonstreaming_response_rejects_high_expansion_gzip_body() -> None:
    decoded = b"x" * (128 * 1024)

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-encoding": "gzip"},
            stream=_CountingByteStream([gzip.compress(decoded)]),
        )

    client = SyncIrodoriClient(
        base_url=BASE_URL,
        transport=httpx.MockTransport(handler),
        max_response_bytes=1_024,
    )

    with pytest.raises(ClientError, match="response") as raised:
        client.health()

    assert raised.value.code == "response_too_large"
    assert raised.value.endpoint == "/health"


def test_nonstreaming_compressed_content_length_rejects_encoded_body_before_reading() -> None:
    stream = _CountingByteStream([empty_raw_deflate_blocks(14)])
    response = httpx.Response(
        200,
        headers={"content-encoding": "deflate", "content-length": "70"},
        stream=stream,
        request=httpx.Request("GET", f"{BASE_URL}/health"),
    )

    with pytest.raises(ClientError, match="response") as raised:
        _read_bounded_response_for_test(response, max_bytes=1)

    assert raised.value.code == "response_too_large"
    assert raised.value.endpoint == "/health"
    assert stream.yielded == 0


def test_nonstreaming_deflate_encoded_limit_accepts_exact_chunked_boundary() -> None:
    encoded = empty_raw_deflate_blocks(13)
    response = httpx.Response(
        200,
        headers={"content-encoding": "deflate"},
        stream=_CountingByteStream([encoded[:64], encoded[64:]]),
        request=httpx.Request("GET", f"{BASE_URL}/health"),
    )

    assert _read_bounded_response_for_test(response, max_bytes=1) == b""


def test_nonstreaming_deflate_encoded_limit_rejects_chunked_empty_block_overflow() -> None:
    encoded = empty_raw_deflate_blocks(14)
    response = httpx.Response(
        200,
        headers={"content-encoding": "deflate"},
        stream=_CountingByteStream([encoded[:65], encoded[65:]]),
        request=httpx.Request("GET", f"{BASE_URL}/health"),
    )

    with pytest.raises(ClientError, match="response") as raised:
        _read_bounded_response_for_test(response, max_bytes=1)

    assert raised.value.code == "response_too_large"
    assert raised.value.endpoint == "/health"


def test_nonstreaming_multiple_member_gzip_accepts_exact_encoded_boundary() -> None:
    encoded = gzip_member() * 2 + gzip_member(extra=b"abc")
    assert len(encoded) == MAX_ONE_BYTE_RESPONSE_ENCODED_BYTES
    response = httpx.Response(
        200,
        headers={"content-encoding": "gzip"},
        stream=_CountingByteStream([encoded[:64], encoded[64:]]),
        request=httpx.Request("GET", f"{BASE_URL}/health"),
    )

    assert _read_bounded_response_for_test(response, max_bytes=1) == b""


def test_nonstreaming_multiple_member_gzip_rejects_encoded_overflow() -> None:
    encoded = gzip_member() * 2 + gzip_member(extra=b"abc")
    response = httpx.Response(
        200,
        headers={"content-encoding": "gzip"},
        stream=_CountingByteStream([encoded, gzip_member()]),
        request=httpx.Request("GET", f"{BASE_URL}/health"),
    )

    with pytest.raises(ClientError, match="response") as raised:
        _read_bounded_response_for_test(response, max_bytes=1)

    assert raised.value.code == "response_too_large"
    assert raised.value.endpoint == "/health"


@pytest.mark.parametrize("content_encoding", ["br", "gzip, deflate"])
def test_nonstreaming_response_rejects_unsupported_content_encoding(
    content_encoding: str,
) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-encoding": content_encoding},
            content=b"encoded",
        )

    client = SyncIrodoriClient(base_url=BASE_URL, transport=httpx.MockTransport(handler))

    with pytest.raises(ClientError, match="content encoding") as raised:
        client.health()

    assert raised.value.code == "protocol_error"
    assert raised.value.endpoint == "/health"


@pytest.mark.parametrize("max_response_bytes", [0, -1, True])
def test_nonstreaming_response_limit_must_be_a_positive_integer(
    max_response_bytes: object,
) -> None:
    with pytest.raises(ValueError, match="max_response_bytes"):
        SyncIrodoriClient(
            base_url=BASE_URL,
            max_response_bytes=max_response_bytes,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("max_stream_frames", [0, -1, True])
def test_stream_frame_limit_must_be_a_positive_integer(max_stream_frames: object) -> None:
    with pytest.raises(ValueError, match="max_stream_frames"):
        SyncIrodoriClient(
            base_url=BASE_URL,
            max_stream_frames=max_stream_frames,  # type: ignore[arg-type]
        )


def test_synthesize_batch_posts_segments_and_returns_ordered_results() -> None:
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

    assert _client(httpx.MockTransport(handler)).synthesize_batch(batch_request) == batch_result


def test_synthesize_stream_reconstructs_byte_exact_payload_across_three_chunks() -> None:
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

    chunks = list(_client(httpx.MockTransport(handler)).synthesize_stream(synthesis_request))

    assert paths == ["/health", "/synthesize_stream"]
    assert chunks == payloads
    assert b"".join(chunks) == b"RIFF-wav"


def test_synthesize_stream_yields_payload_before_response_completes() -> None:
    synthesis_request = SynthesisRequest(text="長い本文です。")
    first_frame = (
        StreamHandshakeHeader(max_chunk_size=MAX_TEST_CHUNK_SIZE).to_bytes()
        + StreamChunkHeader(segment_index=0, byte_length=2, final=False).to_bytes()
        + b"RI"
    )
    final_frame = StreamChunkHeader(segment_index=1, byte_length=2, final=True).to_bytes() + b"FF"
    stream = _CountingByteStream([first_frame, final_frame])

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return _json_response(HealthResponse(max_chunk_size=MAX_TEST_CHUNK_SIZE))
        return httpx.Response(200, stream=stream)

    chunks = _client(httpx.MockTransport(handler)).synthesize_stream(synthesis_request)

    assert next(chunks) == b"RI"
    assert stream.yielded == 1
    assert list(chunks) == [b"FF"]


def test_synthesize_stream_accepts_payload_at_total_byte_boundary() -> None:
    synthesis_request = SynthesisRequest(text="境界値です。")
    health = HealthResponse(max_chunk_size=MAX_TEST_CHUNK_SIZE)
    max_response_bytes = len(health.model_dump_json().encode())
    payloads = [b"x" * MAX_TEST_CHUNK_SIZE] * (max_response_bytes // MAX_TEST_CHUNK_SIZE)
    payloads.append(b"x" * (max_response_bytes % MAX_TEST_CHUNK_SIZE))

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return _json_response(health)
        return httpx.Response(200, content=_framed(payloads))

    client = SyncIrodoriClient(
        base_url=BASE_URL,
        transport=httpx.MockTransport(handler),
        max_response_bytes=max_response_bytes,
    )
    chunks = list(client.synthesize_stream(synthesis_request))

    assert chunks == payloads
    assert len(b"".join(chunks)) == max_response_bytes


def test_synthesize_stream_rejects_total_payload_over_limit() -> None:
    synthesis_request = SynthesisRequest(text="上限超過です。")
    health = HealthResponse(max_chunk_size=MAX_TEST_CHUNK_SIZE)
    max_response_bytes = len(health.model_dump_json().encode())
    payloads = [b"x" * MAX_TEST_CHUNK_SIZE] * (max_response_bytes // MAX_TEST_CHUNK_SIZE)
    payloads.append(b"x" * (max_response_bytes % MAX_TEST_CHUNK_SIZE + 1))

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return _json_response(health)
        return httpx.Response(200, content=_framed(payloads))

    client = SyncIrodoriClient(
        base_url=BASE_URL,
        transport=httpx.MockTransport(handler),
        max_response_bytes=max_response_bytes,
    )
    with pytest.raises(ClientError, match="response") as raised:
        list(client.synthesize_stream(synthesis_request))

    assert raised.value.code == "response_too_large"
    assert raised.value.endpoint == "/synthesize_stream"


def test_synthesize_stream_rejects_frame_count_over_limit() -> None:
    synthesis_request = SynthesisRequest(text="空フレーム攻撃です。")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return _json_response(HealthResponse(max_chunk_size=MAX_TEST_CHUNK_SIZE))
        return httpx.Response(200, content=_framed([b"", b"", b""]))

    client = SyncIrodoriClient(
        base_url=BASE_URL,
        transport=httpx.MockTransport(handler),
        max_stream_frames=2,
    )
    with pytest.raises(ClientError, match="response") as raised:
        list(client.synthesize_stream(synthesis_request))

    assert raised.value.code == "response_too_large"
    assert raised.value.endpoint == "/synthesize_stream"


def test_synthesize_stream_accepts_configured_frame_count_at_limit() -> None:
    synthesis_request = SynthesisRequest(text="空フレーム境界です。")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return _json_response(HealthResponse(max_chunk_size=1))
        return httpx.Response(
            200,
            content=_framed([b"x", b"x", b"x"], max_chunk_size=1),
        )

    client = SyncIrodoriClient(
        base_url=BASE_URL,
        transport=httpx.MockTransport(handler),
        max_stream_frames=3,
    )
    assert list(client.synthesize_stream(synthesis_request)) == [b"x", b"x", b"x"]


@pytest.mark.parametrize(
    ("error_code", "expected_error", "status_code", "message"),
    TERMINAL_STREAM_ERROR_CASES,
)
def test_synthesize_stream_raises_typed_terminal_frame_error(
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
    stream = _CloseTrackingByteStream([terminal_error_framed(error_code)])
    yielded: list[bytes] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return _json_response(HealthResponse(max_chunk_size=MAX_TEST_CHUNK_SIZE))
        return httpx.Response(200, stream=stream)

    with pytest.raises(expected_error, match=message) as raised:
        _collect_into(
            yielded,
            _client(httpx.MockTransport(handler)).synthesize_stream(synthesis_request),
        )

    assert yielded == []
    assert raised.value.status_code == status_code
    assert raised.value.code == error_code
    assert raised.value.details == {}
    assert raised.value.endpoint == "/synthesize_stream"
    assert stream.closed is True


def test_synthesize_stream_counts_terminal_error_frame_toward_frame_limit() -> None:
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

    client = SyncIrodoriClient(
        base_url=BASE_URL,
        transport=httpx.MockTransport(handler),
        max_stream_frames=2,
    )
    with pytest.raises(ClientError, match="response") as raised:
        list(client.synthesize_stream(synthesis_request))

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
def test_synthesize_stream_requires_handshake_and_terminal_final(
    stream_bytes: bytes,
    match: str,
) -> None:
    synthesis_request = SynthesisRequest(text="終端検証です。")
    stream = _CloseTrackingByteStream([stream_bytes])

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return _json_response(HealthResponse(max_chunk_size=MAX_TEST_CHUNK_SIZE))
        return httpx.Response(200, stream=stream)

    with pytest.raises(ClientError, match=match) as raised:
        list(_client(httpx.MockTransport(handler)).synthesize_stream(synthesis_request))

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
def test_synthesize_stream_rejects_protocol_errors(stream: bytes, match: str) -> None:
    synthesis_request = SynthesisRequest(text="異常系です。")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return _json_response(HealthResponse(max_chunk_size=MAX_TEST_CHUNK_SIZE))
        return httpx.Response(200, content=stream)

    with pytest.raises(ClientError, match=match):
        list(_client(httpx.MockTransport(handler)).synthesize_stream(synthesis_request))


@pytest.mark.parametrize("header", [b'{"kind":1}\n', b'{"kind":false}\n', b'{"kind":[]}\n'])
def test_header_kind_ignores_non_string_kind(header: bytes) -> None:
    assert _header_kind(header) is None


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
        (b"x" * (MAX_STREAM_HEADER_BYTES + 1), "header exceeds"),
        (b"x" * (MAX_STREAM_HEADER_BYTES + 1) + b"\n", "header exceeds"),
    ],
)
def test_synthesize_stream_rejects_malformed_frames(stream: bytes, match: str) -> None:
    synthesis_request = SynthesisRequest(text="壊れたフレームです。")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return _json_response(HealthResponse(max_chunk_size=MAX_TEST_CHUNK_SIZE))
        return httpx.Response(200, content=stream)

    with pytest.raises(ClientError, match=match):
        list(_client(httpx.MockTransport(handler)).synthesize_stream(synthesis_request))


def test_synthesize_stream_rejects_compressed_success_response() -> None:
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
        list(_client(httpx.MockTransport(handler)).synthesize_stream(synthesis_request))


@pytest.mark.parametrize(
    ("status_code", "expected_error"),
    [
        (400, ClientError),
        (503, ClientUnavailableError),
    ],
)
def test_synthesize_stream_error_responses_map_to_typed_client_errors(
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
        list(_client(httpx.MockTransport(handler)).synthesize_stream(synthesis_request))

    assert raised.value.status_code == status_code
    assert raised.value.code == error_payload.code
    assert raised.value.details == error_payload.details


@pytest.mark.parametrize(
    ("error_type", "message", "expected_error"),
    [
        (httpx.TimeoutException, "stream timed out", ClientTimeoutError),
        (httpx.ConnectError, "connection failed", ClientUnavailableError),
    ],
)
def test_synthesize_stream_open_failures_map_to_typed_client_errors(
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
        list(_client(httpx.MockTransport(handler)).synthesize_stream(synthesis_request))


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
def test_error_responses_map_to_typed_client_errors(
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
        _client(httpx.MockTransport(handler)).health()

    assert raised.value.status_code == status_code
    assert raised.value.code == error_payload.code
    assert raised.value.details == error_payload.details


@pytest.mark.parametrize(
    ("response", "match"),
    [
        (httpx.Response(400, content=b"not-json"), "not-json"),
        (httpx.Response(422, json={"detail": "bad request"}), "bad request"),
    ],
)
def test_error_mapping_preserves_non_contract_response_context(
    response: httpx.Response,
    match: str,
) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return response

    with pytest.raises(ClientError, match=match) as raised:
        _client(httpx.MockTransport(handler)).health()

    assert raised.value.status_code == response.status_code
    assert raised.value.code == "http_error"
    assert raised.value.details == {"status_code": response.status_code}


def test_transport_timeout_maps_to_client_timeout_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        message = "timed out"
        raise httpx.TimeoutException(message, request=request)

    with pytest.raises(ClientTimeoutError, match="timed out"):
        _client(httpx.MockTransport(handler)).health()


def test_transport_error_maps_to_client_unavailable_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        message = "connection failed"
        raise httpx.ConnectError(message, request=request)

    with pytest.raises(ClientUnavailableError, match="connection failed"):
        _client(httpx.MockTransport(handler)).health()


def test_default_base_url_uses_client_settings() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        settings = ClientSettings()
        assert request.url == httpx.URL(f"http://{settings.host}:{settings.port}/health")
        return _json_response(HealthResponse())

    assert SyncIrodoriClient(transport=httpx.MockTransport(handler)).health().status == "ok"


def test_sync_client_closes_owned_httpx_client() -> None:
    transport = httpx.MockTransport(lambda _request: _json_response(HealthResponse()))

    with _client(transport) as client:
        assert client.health().status == "ok"

    with pytest.raises(RuntimeError, match="closed"):
        client.health()


def test_sync_client_closes_owned_httpx_client_when_stream_health_fails() -> None:
    synthesis_request = SynthesisRequest(text="本文です。")
    error_payload = ErrorPayload(code="server_busy", message="server cannot accept work")
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        assert request.url.path == "/health"
        return _json_response(error_payload, status_code=SERVER_ERROR_STATUS)

    with (
        pytest.raises(ClientUnavailableError, match="server cannot accept work"),
        _client(httpx.MockTransport(handler)) as client,
    ):
        list(client.synthesize_stream(synthesis_request))

    assert paths == ["/health"]
    with pytest.raises(RuntimeError, match="closed"):
        client.health()
