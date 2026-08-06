from __future__ import annotations

import json
from typing import TYPE_CHECKING, NamedTuple, NoReturn, cast

from pydantic import ValidationError

from irodori_tts_infra.client._response import _response_too_large_error
from irodori_tts_infra.client.errors import ClientError, build_stream_error
from irodori_tts_infra.contracts import (
    STREAM_HEADER_VERSION,
    StreamChunkHeader,
    StreamHandshakeHeader,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    import httpx

MAX_STREAM_HEADER_BYTES = 4_096


class StreamPayloadUpdate(NamedTuple):
    item: bytes | ClientError | None
    effective_max_chunk_size: int
    handshake_seen: bool
    payload_seen: bool
    final_seen: bool
    expected_index: int


class StreamPayloadParser:
    def __init__(
        self,
        *,
        health_max_chunk_size: int,
        max_response_bytes: int,
        max_stream_frames: int,
    ) -> None:
        self._health_max_chunk_size = health_max_chunk_size
        self._max_response_bytes = max_response_bytes
        self._max_stream_frames = max_stream_frames
        self._buffer = bytearray()
        self._effective_max_chunk_size = health_max_chunk_size
        self._expected_index = 0
        self._handshake_seen = False
        self._payload_seen = False
        self._final_seen = False
        self._payload_bytes_received = 0
        self._frame_count = 0

    def feed(self, chunk: bytes) -> Iterator[bytes]:
        self._buffer.extend(chunk)
        yield from self._drain(stream_done=False)

    def finish(self) -> Iterator[bytes]:
        yield from self._drain(stream_done=True)
        _validate_stream_completion(
            handshake_seen=self._handshake_seen,
            final_seen=self._final_seen,
        )

    def _drain(self, *, stream_done: bool) -> Iterator[bytes]:
        while self._buffer:
            update = _next_stream_payload(
                self._buffer,
                health_max_chunk_size=self._health_max_chunk_size,
                effective_max_chunk_size=self._effective_max_chunk_size,
                handshake_seen=self._handshake_seen,
                payload_seen=self._payload_seen,
                final_seen=self._final_seen,
                expected_index=self._expected_index,
                stream_done=stream_done,
            )
            if update is None:
                return
            self._apply(update)
            if update.item is not None:
                yield self._accept(update.item)

    def _apply(self, update: StreamPayloadUpdate) -> None:
        self._effective_max_chunk_size = update.effective_max_chunk_size
        self._handshake_seen = update.handshake_seen
        self._payload_seen = update.payload_seen
        self._final_seen = update.final_seen
        self._expected_index = update.expected_index

    def _accept(self, item: bytes | ClientError) -> bytes:
        payload = b"" if isinstance(item, ClientError) else item
        self._payload_bytes_received += len(payload)
        self._frame_count += 1
        if (
            self._payload_bytes_received > self._max_response_bytes
            or self._frame_count > self._max_stream_frames
        ):
            endpoint = "/synthesize_stream"
            raise _response_too_large_error(endpoint)
        if isinstance(item, ClientError):
            raise item
        return item


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
) -> StreamPayloadUpdate | None:
    if final_seen:
        _raise_protocol_error("frame after final chunk")

    newline_index = buffer.find(b"\n")
    if newline_index < 0:
        if len(buffer) > MAX_STREAM_HEADER_BYTES:
            _raise_protocol_error("stream header exceeds size limit")
        if stream_done:
            _raise_protocol_error("missing stream header separator")
        return None
    if newline_index + 1 > MAX_STREAM_HEADER_BYTES:
        _raise_protocol_error("stream header exceeds size limit")

    header_line = bytes(buffer[: newline_index + 1])
    kind = _header_kind(header_line)

    if kind == "handshake":
        next_max_chunk_size = _handle_handshake(
            header_line,
            handshake_seen=handshake_seen,
            payload_seen=payload_seen,
            health_max_chunk_size=health_max_chunk_size,
        )
        del buffer[: newline_index + 1]
        return StreamPayloadUpdate(
            item=None,
            effective_max_chunk_size=next_max_chunk_size,
            handshake_seen=True,
            payload_seen=payload_seen,
            final_seen=final_seen,
            expected_index=expected_index,
        )

    header = _parse_required_chunk_header(
        header_line,
        kind=kind,
        handshake_seen=handshake_seen,
    )
    terminal_error = None
    if header.error_code is not None:
        terminal_error = build_stream_error(
            header.error_code,
            endpoint="/synthesize_stream",
        )
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
    return StreamPayloadUpdate(
        item=item,
        effective_max_chunk_size=effective_max_chunk_size,
        handshake_seen=handshake_seen,
        payload_seen=True,
        final_seen=header.final,
        expected_index=expected_index + 1,
    )


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
