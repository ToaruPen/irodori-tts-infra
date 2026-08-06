from __future__ import annotations

from typing import TYPE_CHECKING, Self

import httpx

from irodori_tts_infra.client._response import (
    _ACCEPTED_RESPONSE_ENCODINGS,
    _RAW_RESPONSE_CHUNK_BYTES,
    _STREAM_RESPONSE_ENCODING,
    DEFAULT_MAX_RESPONSE_BYTES,
    DEFAULT_MAX_STREAM_FRAMES,
    BoundedResponseDecoder,
    _buffered_response,
    _iter_bounded_decoded_content,
    _validate_content_encoding,
    _validate_max_response_bytes,
    _validate_max_stream_frames,
)
from irodori_tts_infra.client._stream import StreamPayloadParser, _require_identity_stream
from irodori_tts_infra.client.errors import (
    build_response_error,
    build_timeout_error,
    build_transport_error,
)
from irodori_tts_infra.client.sync import _default_base_url, _json_body
from irodori_tts_infra.contracts import (
    BatchSynthesisRequest,
    BatchSynthesisResult,
    CapabilitiesResponse,
    HealthResponse,
    SynthesisRequest,
    SynthesisResult,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from types import TracebackType


class AsyncIrodoriClient:
    def __init__(
        self,
        *,
        base_url: str | None = None,
        timeout: float | httpx.Timeout | None = 30.0,
        transport: httpx.AsyncBaseTransport | None = None,
        max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
        max_stream_frames: int = DEFAULT_MAX_STREAM_FRAMES,
    ) -> None:
        _validate_max_response_bytes(max_response_bytes)
        _validate_max_stream_frames(max_stream_frames)
        self._max_response_bytes = max_response_bytes
        self._max_stream_frames = max_stream_frames
        self._client = httpx.AsyncClient(
            base_url=base_url or _default_base_url(),
            headers={"accept-encoding": _ACCEPTED_RESPONSE_ENCODINGS},
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

    async def capabilities(self) -> CapabilitiesResponse:
        response = await self._request("GET", "/capabilities")
        return CapabilitiesResponse.model_validate_json(response.content)

    async def synthesize(self, request: SynthesisRequest) -> SynthesisResult:
        response = await self._request("POST", "/synthesize", json=_json_body(request))
        return SynthesisResult.model_validate_json(response.content)

    async def synthesize_batch(self, request: BatchSynthesisRequest) -> BatchSynthesisResult:
        response = await self._request("POST", "/synthesize_batch", json=_json_body(request))
        return BatchSynthesisResult.model_validate_json(response.content)

    async def synthesize_stream(
        self,
        request: SynthesisRequest | BatchSynthesisRequest,
    ) -> AsyncIterator[bytes]:
        health = await self.health()
        try:
            async with self._client.stream(
                "POST",
                "/synthesize_stream",
                headers={"accept-encoding": _STREAM_RESPONSE_ENCODING},
                json=_json_body(request),
            ) as response:
                if response.is_error:
                    endpoint = "/synthesize_stream"
                    content = await _read_bounded_response(
                        response,
                        max_bytes=self._max_response_bytes,
                        endpoint=endpoint,
                    )
                    raise build_response_error(
                        _buffered_response(response, content),
                        endpoint=endpoint,
                    )
                _require_identity_stream(response)
                async for payload in _iter_stream_payloads(
                    response.aiter_bytes(chunk_size=health.max_chunk_size),
                    health_max_chunk_size=health.max_chunk_size,
                    max_response_bytes=self._max_response_bytes,
                    max_stream_frames=self._max_stream_frames,
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
            async with self._client.stream(method, path, json=json) as streaming_response:
                content = await _read_bounded_response(
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


async def _read_bounded_response(
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

    decoder = BoundedResponseDecoder(response, max_bytes=max_bytes, endpoint=endpoint)
    async for raw in response.aiter_raw(chunk_size=_RAW_RESPONSE_CHUNK_BYTES):
        for decoded in decoder.decode(raw):
            content.extend(decoded)
    for decoded in decoder.finish():
        content.extend(decoded)
    return bytes(content)


async def _iter_stream_payloads(
    chunks: AsyncIterator[bytes],
    *,
    health_max_chunk_size: int,
    max_response_bytes: int,
    max_stream_frames: int,
) -> AsyncIterator[bytes]:
    parser = StreamPayloadParser(
        health_max_chunk_size=health_max_chunk_size,
        max_response_bytes=max_response_bytes,
        max_stream_frames=max_stream_frames,
    )
    async for chunk in chunks:
        for payload in parser.feed(chunk):
            yield payload
    for payload in parser.finish():
        yield payload
