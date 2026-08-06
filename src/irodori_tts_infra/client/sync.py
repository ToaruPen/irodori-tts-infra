from __future__ import annotations

import json
from typing import TYPE_CHECKING, NoReturn, Self, cast

import httpx
from pydantic import BaseModel, ValidationError

from irodori_tts_infra.client._response import (
    _ACCEPTED_RESPONSE_ENCODINGS,
    _RAW_RESPONSE_CHUNK_BYTES,
    _STREAM_RESPONSE_ENCODING,
    DEFAULT_MAX_RESPONSE_BYTES,
    DEFAULT_MAX_STREAM_FRAMES,
    _BoundedResponseDecoder,
    _buffered_response,
    _iter_bounded_decoded_content,
    _response_too_large_error,
    _validate_content_encoding,
    _validate_max_response_bytes,
    _validate_max_stream_frames,
)
from irodori_tts_infra.client.errors import (
    ClientError,
    build_response_error,
    build_stream_error,
    build_timeout_error,
    build_transport_error,
)
from irodori_tts_infra.config import ClientSettings
from irodori_tts_infra.contracts import (
    STREAM_HEADER_VERSION,
    BatchSynthesisRequest,
    BatchSynthesisResult,
    CapabilitiesResponse,
    HealthResponse,
    StreamChunkHeader,
    StreamHandshakeHeader,
    SynthesisRequest,
    SynthesisResult,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from types import TracebackType

_StreamPayloadUpdate = tuple[bytes | ClientError | None, int, bool, bool, bool, int]
_MAX_STREAM_HEADER_BYTES = 4_096


class SyncIrodoriClient:
    def __init__(
        self,
        *,
        base_url: str | None = None,
        timeout: float | httpx.Timeout | None = 30.0,
        transport: httpx.BaseTransport | None = None,
        max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
        max_stream_frames: int = DEFAULT_MAX_STREAM_FRAMES,
    ) -> None:
        _validate_max_response_bytes(max_response_bytes)
        _validate_max_stream_frames(max_stream_frames)
        self._max_response_bytes = max_response_bytes
        self._max_stream_frames = max_stream_frames
        self._client = httpx.Client(
            base_url=base_url or _default_base_url(),
            headers={"accept-encoding": _ACCEPTED_RESPONSE_ENCODINGS},
            timeout=timeout,
            transport=transport,
        )

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def health(self) -> HealthResponse:
        response = self._request("GET", "/health")
        return HealthResponse.model_validate_json(response.content)

    def capabilities(self) -> CapabilitiesResponse:
        response = self._request("GET", "/capabilities")
        return CapabilitiesResponse.model_validate_json(response.content)

    def synthesize(self, request: SynthesisRequest) -> SynthesisResult:
        response = self._request("POST", "/synthesize", json=_json_body(request))
        return SynthesisResult.model_validate_json(response.content)

    def synthesize_batch(self, request: BatchSynthesisRequest) -> BatchSynthesisResult:
        response = self._request("POST", "/synthesize_batch", json=_json_body(request))
        return BatchSynthesisResult.model_validate_json(response.content)

    def synthesize_stream(
        self,
        request: SynthesisRequest | BatchSynthesisRequest,
    ) -> Iterator[bytes]:
        health = self.health()
        try:
            with self._client.stream(
                "POST",
                "/synthesize_stream",
                headers={"accept-encoding": _STREAM_RESPONSE_ENCODING},
                json=_json_body(request),
            ) as response:
                if response.is_error:
                    endpoint = "/synthesize_stream"
                    content = _read_bounded_response(
                        response,
                        max_bytes=self._max_response_bytes,
                        endpoint=endpoint,
                    )
                    raise build_response_error(
                        _buffered_response(response, content),
                        endpoint=endpoint,
                    )
                _require_identity_stream(response)
                yield from _iter_stream_payloads(
                    response.iter_bytes(chunk_size=health.max_chunk_size),
                    health.max_chunk_size,
                    self._max_response_bytes,
                    self._max_stream_frames,
                )
        except httpx.TimeoutException as exc:
            raise build_timeout_error(exc, endpoint="/synthesize_stream") from exc
        except httpx.TransportError as exc:
            raise build_transport_error(exc, endpoint="/synthesize_stream") from exc

    def _request(self, method: str, path: str, *, json: object | None = None) -> httpx.Response:
        try:
            with self._client.stream(method, path, json=json) as streaming_response:
                content = _read_bounded_response(
                    streaming_response,
                    max_bytes=self._max_response_bytes,
                    endpoint=path,
                )
                response = _buffered_response(streaming_response, content)
        except httpx.TimeoutException as exc:
            raise build_timeout_error(exc, endpoint=path) from exc
        except httpx.TransportError as exc:
            raise build_transport_error(exc, endpoint=path) from exc
        if response.is_error:
            raise build_response_error(response, endpoint=path)
        return response


def _read_bounded_response(
    response: httpx.Response,
    *,
    max_bytes: int,
    endpoint: str,
) -> bytes:
    content = bytearray()
    if response.is_stream_consumed:
        _validate_content_encoding(response, endpoint=endpoint)
        for decoded in _iter_bounded_decoded_content(
            response.content,
            max_bytes=max_bytes,
            endpoint=endpoint,
        ):
            content.extend(decoded)
        return bytes(content)

    decoder = _BoundedResponseDecoder(response, max_bytes=max_bytes, endpoint=endpoint)
    for raw in response.iter_raw(chunk_size=_RAW_RESPONSE_CHUNK_BYTES):
        for decoded in decoder.decode(raw):
            content.extend(decoded)
    for decoded in decoder.finish():
        content.extend(decoded)
    return bytes(content)


def _default_base_url() -> str:
    settings = ClientSettings()
    return f"http://{settings.host}:{settings.port}"


def _json_body(model: BaseModel) -> dict[str, object]:
    return model.model_dump(mode="json")


def _iter_stream_payloads(
    chunks: Iterator[bytes],
    health_max_chunk_size: int,
    max_response_bytes: int,
    max_stream_frames: int,
) -> Iterator[bytes]:
    buffer = bytearray()
    expected_index = 0
    effective_max_chunk_size = health_max_chunk_size
    handshake_seen = False
    payload_seen = False
    final_seen = False
    payload_bytes_received = 0
    frame_count = 0

    for chunk in chunks:
        buffer.extend(chunk)
        while True:
            if not buffer:
                break
            payload = _next_stream_payload(
                buffer,
                health_max_chunk_size=health_max_chunk_size,
                effective_max_chunk_size=effective_max_chunk_size,
                handshake_seen=handshake_seen,
                payload_seen=payload_seen,
                final_seen=final_seen,
                expected_index=expected_index,
                stream_done=False,
            )
            if payload is None:
                break
            (
                payload_item,
                effective_max_chunk_size,
                handshake_seen,
                payload_seen,
                final_seen,
                expected_index,
            ) = payload
            if payload_item is not None:
                payload_bytes, payload_bytes_received, frame_count = _accept_stream_item(
                    payload_item,
                    payload_bytes_received=payload_bytes_received,
                    frame_count=frame_count,
                    max_response_bytes=max_response_bytes,
                    max_stream_frames=max_stream_frames,
                )
                yield payload_bytes

    while buffer:
        payload = _next_stream_payload(
            buffer,
            health_max_chunk_size=health_max_chunk_size,
            effective_max_chunk_size=effective_max_chunk_size,
            handshake_seen=handshake_seen,
            payload_seen=payload_seen,
            final_seen=final_seen,
            expected_index=expected_index,
            stream_done=True,
        )
        if payload is None:
            break
        (
            payload_item,
            effective_max_chunk_size,
            handshake_seen,
            payload_seen,
            final_seen,
            expected_index,
        ) = payload
        if payload_item is not None:
            payload_bytes, payload_bytes_received, frame_count = _accept_stream_item(
                payload_item,
                payload_bytes_received=payload_bytes_received,
                frame_count=frame_count,
                max_response_bytes=max_response_bytes,
                max_stream_frames=max_stream_frames,
            )
            yield payload_bytes

    _validate_stream_completion(handshake_seen=handshake_seen, final_seen=final_seen)


def _accept_stream_payload(
    payload: bytes,
    *,
    payload_bytes_received: int,
    frame_count: int,
    max_response_bytes: int,
    max_stream_frames: int,
) -> tuple[int, int]:
    payload_bytes_received += len(payload)
    frame_count += 1
    if payload_bytes_received > max_response_bytes or frame_count > max_stream_frames:
        endpoint = "/synthesize_stream"
        raise _response_too_large_error(endpoint)
    return payload_bytes_received, frame_count


def _accept_stream_item(
    item: bytes | ClientError,
    *,
    payload_bytes_received: int,
    frame_count: int,
    max_response_bytes: int,
    max_stream_frames: int,
) -> tuple[bytes, int, int]:
    payload = b"" if isinstance(item, ClientError) else item
    payload_bytes_received, frame_count = _accept_stream_payload(
        payload,
        payload_bytes_received=payload_bytes_received,
        frame_count=frame_count,
        max_response_bytes=max_response_bytes,
        max_stream_frames=max_stream_frames,
    )
    if isinstance(item, ClientError):
        raise item
    return item, payload_bytes_received, frame_count


def _validate_stream_completion(*, handshake_seen: bool, final_seen: bool) -> None:
    if not handshake_seen:
        _raise_protocol_error("missing handshake")
    if not final_seen:
        _raise_protocol_error("missing final chunk")


def _next_stream_payload(
    buffer: bytearray,
    *,
    health_max_chunk_size: int,
    effective_max_chunk_size: int,
    handshake_seen: bool,
    payload_seen: bool,
    final_seen: bool,
    expected_index: int,
    stream_done: bool,
) -> _StreamPayloadUpdate | None:
    if final_seen:
        _raise_protocol_error("frame after final chunk")

    newline_index = buffer.find(b"\n")
    if newline_index < 0:
        if len(buffer) > _MAX_STREAM_HEADER_BYTES:
            _raise_protocol_error("stream header exceeds size limit")
        if stream_done:
            _raise_protocol_error("missing stream header separator")
        return None
    if newline_index + 1 > _MAX_STREAM_HEADER_BYTES:
        _raise_protocol_error("stream header exceeds size limit")

    header_line = bytes(buffer[: newline_index + 1])
    kind = _header_kind(header_line)

    if kind == "handshake":
        effective_max_chunk_size = _handle_handshake(
            header_line,
            handshake_seen=handshake_seen,
            payload_seen=payload_seen,
            health_max_chunk_size=health_max_chunk_size,
        )
        del buffer[: newline_index + 1]
        return None, effective_max_chunk_size, True, payload_seen, final_seen, expected_index

    header = _parse_required_chunk_header(
        header_line,
        kind=kind,
        handshake_seen=handshake_seen,
    )
    terminal_error = None
    if header.error_code is not None:
        endpoint = "/synthesize_stream"
        terminal_error = build_stream_error(header.error_code, endpoint=endpoint)
    if header.byte_length > effective_max_chunk_size:
        _raise_protocol_error("chunk byte_length exceeds stream cap")
    if header.segment_index != expected_index:
        _raise_protocol_error("unexpected segment_index")

    payload_start = newline_index + 1
    payload_end = payload_start + header.byte_length
    if payload_end > len(buffer):
        if stream_done:
            _raise_protocol_error("truncated stream payload")
        return None

    payload = bytes(buffer[payload_start:payload_end])
    del buffer[:payload_end]
    item = terminal_error if terminal_error is not None else payload
    return item, effective_max_chunk_size, handshake_seen, True, header.final, expected_index + 1


def _parse_required_chunk_header(
    header_line: bytes,
    *,
    kind: str | None,
    handshake_seen: bool,
) -> StreamChunkHeader:
    if kind != "chunk":
        _raise_protocol_error("unknown stream header kind")
    if not handshake_seen:
        _raise_protocol_error("missing handshake before payload")
    return _parse_chunk_header(header_line)


def _handle_handshake(
    header_line: bytes,
    *,
    handshake_seen: bool,
    payload_seen: bool,
    health_max_chunk_size: int,
) -> int:
    if payload_seen:
        _raise_protocol_error("handshake after payload")
    if handshake_seen:
        _raise_protocol_error("duplicate handshake")
    handshake = _parse_handshake(header_line)
    if handshake.max_chunk_size > health_max_chunk_size:
        _raise_protocol_error("handshake max_chunk_size exceeds health max_chunk_size")
    return handshake.max_chunk_size


def _header_kind(header_line: bytes) -> str | None:
    try:
        data = json.loads(header_line)
    except json.JSONDecodeError as exc:
        message = "stream protocol error: malformed header JSON"
        raise ClientError(
            message,
            code="protocol_error",
            endpoint="/synthesize_stream",
        ) from exc
    if not isinstance(data, dict):
        _raise_protocol_error("stream header must be a JSON object")
    data = cast("dict[str, object]", data)
    value = data.get("kind")
    return value if isinstance(value, str) else None


def _parse_handshake(header_line: bytes) -> StreamHandshakeHeader:
    try:
        handshake = StreamHandshakeHeader.from_bytes(header_line)
    except ValidationError as exc:
        message = "stream protocol error: invalid handshake header"
        raise ClientError(
            message,
            code="protocol_error",
            endpoint="/synthesize_stream",
        ) from exc
    _validate_header_version(handshake.header_version)
    return handshake


def _parse_chunk_header(header_line: bytes) -> StreamChunkHeader:
    try:
        header = StreamChunkHeader.from_bytes(header_line)
    except ValidationError as exc:
        message = "stream protocol error: invalid chunk header"
        raise ClientError(
            message,
            code="protocol_error",
            endpoint="/synthesize_stream",
        ) from exc
    _validate_header_version(header.header_version)
    return header


def _validate_header_version(header_version: int) -> None:
    if header_version != STREAM_HEADER_VERSION:
        _raise_protocol_error("unknown stream header version")


def _raise_protocol_error(message: str) -> NoReturn:
    error_message = f"stream protocol error: {message}"
    raise ClientError(
        error_message,
        code="protocol_error",
        endpoint="/synthesize_stream",
    )


def _require_identity_stream(response: httpx.Response) -> None:
    content_encoding = response.headers.get("content-encoding", "identity").strip().lower()
    if content_encoding not in {"", "identity"}:
        message = "stream protocol error: unsupported content encoding"
        raise ClientError(
            message,
            code="protocol_error",
            endpoint="/synthesize_stream",
        )
