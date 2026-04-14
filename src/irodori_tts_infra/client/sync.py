from __future__ import annotations

import json
from typing import TYPE_CHECKING, NoReturn, Self, cast

import httpx
from pydantic import BaseModel, ValidationError

from irodori_tts_infra.client.errors import (
    ClientError,
    build_response_error,
    build_timeout_error,
    build_transport_error,
)
from irodori_tts_infra.config import ClientSettings
from irodori_tts_infra.contracts import (
    STREAM_HEADER_VERSION,
    BatchSynthesisRequest,
    BatchSynthesisResult,
    HealthResponse,
    StreamChunkHeader,
    StreamHandshakeHeader,
    SynthesisRequest,
    SynthesisResult,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from types import TracebackType

_StreamPayloadUpdate = tuple[bytes | None, int, bool, bool, bool, int]


class SyncIrodoriClient:
    def __init__(
        self,
        *,
        base_url: str | None = None,
        timeout: float | httpx.Timeout | None = 30.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._client = httpx.Client(
            base_url=base_url or _default_base_url(),
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

    def synthesize(self, request: SynthesisRequest) -> SynthesisResult:
        response = self._request("POST", "/synthesize", json=_json_body(request))
        return SynthesisResult.model_validate_json(response.content)

    def synthesize_batch(self, request: BatchSynthesisRequest) -> BatchSynthesisResult:
        response = self._request("POST", "/synthesize_batch", json=_json_body(request))
        return BatchSynthesisResult.model_validate_json(response.content)

    def synthesize_stream(self, request: SynthesisRequest) -> Iterator[bytes]:
        health = self.health()
        try:
            with self._client.stream(
                "POST",
                "/synthesize_stream",
                json=_json_body(request),
            ) as response:
                if response.is_error:
                    response.read()
                    raise build_response_error(response, endpoint="/synthesize_stream")
                yield from _iter_stream_payloads(
                    response.iter_bytes(chunk_size=health.max_chunk_size),
                    health.max_chunk_size,
                )
        except httpx.TimeoutException as exc:
            raise build_timeout_error(exc, endpoint="/synthesize_stream") from exc
        except httpx.TransportError as exc:
            raise build_transport_error(exc, endpoint="/synthesize_stream") from exc

    def _request(self, method: str, path: str, *, json: object | None = None) -> httpx.Response:
        try:
            response = self._client.request(method, path, json=json)
        except httpx.TimeoutException as exc:
            raise build_timeout_error(exc, endpoint=path) from exc
        except httpx.TransportError as exc:
            raise build_transport_error(exc, endpoint=path) from exc
        if response.is_error:
            raise build_response_error(response, endpoint=path)
        return response


def _default_base_url() -> str:
    settings = ClientSettings()
    return f"http://{settings.host}:{settings.port}"


def _json_body(model: BaseModel) -> dict[str, object]:
    return model.model_dump(mode="json")


def _iter_stream_payloads(
    chunks: Iterator[bytes],
    health_max_chunk_size: int,
) -> Iterator[bytes]:
    buffer = bytearray()
    expected_index = 0
    effective_max_chunk_size = health_max_chunk_size
    handshake_seen = False
    payload_seen = False
    final_seen = False

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
                payload_bytes,
                effective_max_chunk_size,
                handshake_seen,
                payload_seen,
                final_seen,
                expected_index,
            ) = payload
            if payload_bytes is not None:
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
            payload_bytes,
            effective_max_chunk_size,
            handshake_seen,
            payload_seen,
            final_seen,
            expected_index,
        ) = payload
        if payload_bytes is not None:
            yield payload_bytes


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
        if stream_done:
            _raise_protocol_error("missing stream header separator")
        return None

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

    if kind != "chunk":
        _raise_protocol_error("unknown stream header kind")

    header = _parse_chunk_header(header_line)
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
    return payload, effective_max_chunk_size, handshake_seen, True, header.final, expected_index + 1


def _handle_handshake(
    header_line: bytes,
    *,
    handshake_seen: bool,
    payload_seen: bool,
    health_max_chunk_size: int,
) -> int:
    if handshake_seen:
        _raise_protocol_error("duplicate handshake")
    if payload_seen:
        _raise_protocol_error("handshake after payload")
    handshake = _parse_handshake(header_line)
    if handshake.max_chunk_size > health_max_chunk_size:
        _raise_protocol_error("handshake max_chunk_size exceeds health max_chunk_size")
    return handshake.max_chunk_size


def _header_kind(header_line: bytes) -> str | None:
    try:
        data = json.loads(header_line)
    except json.JSONDecodeError as exc:
        message = "stream protocol error: malformed header JSON"
        raise ClientError(message, code="protocol_error") from exc
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
        raise ClientError(message, code="protocol_error") from exc
    _validate_header_version(handshake.header_version)
    return handshake


def _parse_chunk_header(header_line: bytes) -> StreamChunkHeader:
    try:
        header = StreamChunkHeader.from_bytes(header_line)
    except ValidationError as exc:
        message = "stream protocol error: invalid chunk header"
        raise ClientError(message, code="protocol_error") from exc
    _validate_header_version(header.header_version)
    return header


def _validate_header_version(header_version: int) -> None:
    if header_version != STREAM_HEADER_VERSION:
        _raise_protocol_error("unknown stream header version")


def _raise_protocol_error(message: str) -> NoReturn:
    error_message = f"stream protocol error: {message}"
    raise ClientError(error_message, code="protocol_error")
