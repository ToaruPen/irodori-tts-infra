from __future__ import annotations

from typing import TYPE_CHECKING, Self

import httpx

from irodori_tts_infra.client.errors import (
    build_response_error,
    build_timeout_error,
    build_transport_error,
)
from irodori_tts_infra.client.sync import (
    _default_base_url,
    _handle_handshake,
    _header_kind,
    _json_body,
    _parse_chunk_header,
    _raise_protocol_error,
)
from irodori_tts_infra.contracts import (
    BatchSynthesisRequest,
    BatchSynthesisResult,
    HealthResponse,
    SynthesisRequest,
    SynthesisResult,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from types import TracebackType

_StreamPayloadUpdate = tuple[bytes | None, int, bool, bool, bool, int]


class AsyncIrodoriClient:
    def __init__(
        self,
        *,
        base_url: str | None = None,
        timeout: float | httpx.Timeout | None = 30.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url or _default_base_url(),
            timeout=timeout,
            transport=transport,
        )

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def health(self) -> HealthResponse:
        response = await self._request("GET", "/health")
        return HealthResponse.model_validate_json(response.content)

    async def synthesize(self, request: SynthesisRequest) -> SynthesisResult:
        response = await self._request("POST", "/synthesize", json=_json_body(request))
        return SynthesisResult.model_validate_json(response.content)

    async def synthesize_batch(self, request: BatchSynthesisRequest) -> BatchSynthesisResult:
        response = await self._request("POST", "/synthesize_batch", json=_json_body(request))
        return BatchSynthesisResult.model_validate_json(response.content)

    async def synthesize_stream(self, request: SynthesisRequest) -> AsyncIterator[bytes]:
        health = await self.health()
        try:
            async with self._client.stream(
                "POST",
                "/synthesize_stream",
                json=_json_body(request),
            ) as response:
                if response.is_error:
                    await response.aread()
                    raise build_response_error(response, endpoint="/synthesize_stream")
                async for payload in _iter_stream_payloads(
                    response.aiter_bytes(),
                    health.max_chunk_size,
                ):
                    yield payload
        except httpx.TimeoutException as exc:
            raise build_timeout_error(exc, endpoint="/synthesize_stream") from exc
        except httpx.TransportError as exc:
            raise build_transport_error(exc, endpoint="/synthesize_stream") from exc

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: object | None = None,
    ) -> httpx.Response:
        try:
            response = await self._client.request(method, path, json=json)
        except httpx.TimeoutException as exc:
            raise build_timeout_error(exc, endpoint=path) from exc
        except httpx.TransportError as exc:
            raise build_transport_error(exc, endpoint=path) from exc
        if response.is_error:
            raise build_response_error(response, endpoint=path)
        return response


async def _iter_stream_payloads(
    chunks: AsyncIterator[bytes],
    health_max_chunk_size: int,
) -> AsyncIterator[bytes]:
    buffer = bytearray()
    expected_index = 0
    effective_max_chunk_size = health_max_chunk_size
    handshake_seen = False
    payload_seen = False
    final_seen = False

    async for chunk in chunks:
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
