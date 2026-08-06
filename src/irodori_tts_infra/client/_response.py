from __future__ import annotations

import zlib
from collections import deque
from typing import TYPE_CHECKING, Protocol

import httpx

from irodori_tts_infra.client.errors import ClientError

if TYPE_CHECKING:
    from collections.abc import Iterator

DEFAULT_MAX_RESPONSE_BYTES = 64 * 1024 * 1024
DEFAULT_MAX_STREAM_FRAMES = 65_536
_ACCEPTED_RESPONSE_ENCODINGS = "gzip, deflate"
_DECODE_CHUNK_BYTES = 64 * 1024
_RAW_RESPONSE_CHUNK_BYTES = 64 * 1024
_STREAM_RESPONSE_ENCODING = "identity"


class _Decompressor(Protocol):
    @property
    def eof(self) -> bool: ...

    @property
    def unconsumed_tail(self) -> bytes: ...

    @property
    def unused_data(self) -> bytes: ...

    def decompress(self, data: bytes, _max_length: int = 0, /) -> bytes: ...


class _DeflateCandidate:
    def __init__(self, *, wbits: int, max_bytes: int) -> None:
        self._decoder: _Decompressor = zlib.decompressobj(wbits)
        self._max_bytes = max_bytes
        self._decoded_bytes = 0
        self._chunks: deque[bytes] = deque()
        self.invalid = False
        self.trailing = False
        self.too_large = False

    def decode(self, raw: bytes) -> None:
        if self.invalid:
            return
        pending = raw
        while pending:
            if self._decoder.eof:
                self._invalidate(trailing=True)
                return
            try:
                decoded = self._decoder.decompress(pending, self._next_output_limit())
            except zlib.error:
                self._invalidate()
                return
            pending = self._decoder.unconsumed_tail
            if self._decoder.unused_data:
                self._invalidate(trailing=True)
                return
            self._record(decoded)

    def finish(self) -> None:
        if self.invalid:
            return
        while True:
            try:
                decoded = self._decoder.decompress(b"", self._next_output_limit())
            except zlib.error:
                self._invalidate()
                return
            if not decoded:
                break
            self._record(decoded)
        if not self._decoder.eof or self._decoder.unused_data:
            self._invalidate(trailing=bool(self._decoder.unused_data))

    def drain(self) -> Iterator[bytes]:
        while self._chunks:
            yield self._chunks.popleft()

    def discard(self) -> None:
        self._chunks.clear()

    def _record(self, decoded: bytes) -> None:
        if not decoded:
            return
        self._decoded_bytes += len(decoded)
        if self.too_large:
            return
        if self._decoded_bytes > self._max_bytes:
            self.too_large = True
            self._chunks.clear()
            return
        self._chunks.append(decoded)

    def _next_output_limit(self) -> int:
        if self.too_large:
            return min(_DECODE_CHUNK_BYTES, self._max_bytes + 1)
        return min(_DECODE_CHUNK_BYTES, self._max_bytes - self._decoded_bytes + 1)

    def _invalidate(self, *, trailing: bool = False) -> None:
        self.invalid = True
        self.trailing = trailing
        self._chunks.clear()


class DeflateResponseDecoder:
    def __init__(self, *, max_bytes: int, endpoint: str) -> None:
        self._endpoint = endpoint
        self._zlib = _DeflateCandidate(wbits=zlib.MAX_WBITS, max_bytes=max_bytes)
        self._raw = _DeflateCandidate(wbits=-zlib.MAX_WBITS, max_bytes=max_bytes)
        self._selected: _DeflateCandidate | None = None

    def decode(self, raw: bytes) -> Iterator[bytes]:
        if self._selected is not None:
            self._selected.decode(raw)
            yield from self._drain_selected()
            return

        self._zlib.decode(raw)
        self._raw.decode(raw)
        yield from self._resolve_candidates()

    def finish(self) -> Iterator[bytes]:
        if self._selected is not None:
            self._selected.finish()
            yield from self._drain_selected()
            return

        self._zlib.finish()
        self._raw.finish()
        if self._zlib.trailing:
            raise _invalid_compressed_content_error(self._endpoint)
        if not self._zlib.invalid:
            self._select(self._zlib)
        elif not self._raw.invalid:
            self._select(self._raw)
        else:
            raise _invalid_compressed_content_error(self._endpoint)
        yield from self._drain_selected()

    def _resolve_candidates(self) -> Iterator[bytes]:
        if self._zlib.trailing:
            raise _invalid_compressed_content_error(self._endpoint)
        if self._zlib.invalid and self._raw.invalid:
            raise _invalid_compressed_content_error(self._endpoint)
        if self._zlib.invalid:
            self._select(self._raw)
        elif self._raw.invalid:
            self._select(self._zlib)
        if self._selected is not None:
            yield from self._drain_selected()

    def _drain_selected(self) -> Iterator[bytes]:
        if self._selected is None:
            message = "deflate response decoder is not selected"
            raise AssertionError(message)
        if self._selected.invalid:
            raise _invalid_compressed_content_error(self._endpoint)
        if self._selected.too_large:
            raise _response_too_large_error(self._endpoint)
        yield from self._selected.drain()

    def _select(self, candidate: _DeflateCandidate) -> None:
        self._selected = candidate
        other = self._raw if candidate is self._zlib else self._zlib
        other.discard()


class BoundedResponseDecoder:
    def __init__(
        self,
        response: httpx.Response,
        *,
        max_bytes: int,
        endpoint: str,
    ) -> None:
        self._max_bytes = max_bytes
        self._endpoint = endpoint
        self._decoded_bytes = 0
        self._encoding = _content_encoding(response, endpoint=endpoint)
        self._encoded_bytes = 0
        self._max_encoded_bytes = (
            max_bytes if self._encoding == "identity" else _encoded_response_limit(max_bytes)
        )
        self._decoder: _Decompressor | None = None
        self._deflate_decoder: DeflateResponseDecoder | None = None

        content_length = response.headers.get("content-length")
        if content_length is not None:
            if not content_length.isascii() or not content_length.isdigit():
                raise _invalid_content_length_error(endpoint)
            normalized_length = content_length.lstrip("0") or "0"
            encoded_limit = str(self._max_encoded_bytes)
            if len(normalized_length) > len(encoded_limit) or (
                len(normalized_length) == len(encoded_limit) and normalized_length > encoded_limit
            ):
                raise _response_too_large_error(endpoint)

        if self._encoding == "gzip":
            self._decoder = zlib.decompressobj(zlib.MAX_WBITS | 16)
        elif self._encoding == "deflate":
            self._deflate_decoder = DeflateResponseDecoder(
                max_bytes=max_bytes,
                endpoint=endpoint,
            )

    def decode(self, raw: bytes) -> Iterator[bytes]:
        if not raw:
            return
        self._encoded_bytes += len(raw)
        if self._encoded_bytes > self._max_encoded_bytes:
            raise _response_too_large_error(self._endpoint)
        if self._encoding == "identity":
            yield from self._decode_identity(raw)
            return
        if self._encoding == "deflate":
            if self._deflate_decoder is None:
                message = "deflate response decoder is not initialized"
                raise AssertionError(message)
            for decoded in self._deflate_decoder.decode(raw):
                yield self._accept(decoded)
            return
        yield from self._decode_compressed(raw)

    def finish(self) -> Iterator[bytes]:
        if self._encoding == "identity":
            return
        if self._encoding == "deflate":
            if self._deflate_decoder is None:
                message = "deflate response decoder is not initialized"
                raise AssertionError(message)
            for decoded in self._deflate_decoder.finish():
                yield self._accept(decoded)
            return
        if self._decoder is None:
            raise _invalid_compressed_content_error(self._endpoint)

        while True:
            try:
                decoded = self._decoder.decompress(b"", self._next_output_limit())
            except zlib.error as exc:
                raise _invalid_compressed_content_error(self._endpoint) from exc
            if not decoded:
                break
            yield self._accept(decoded)

        if not self._decoder.eof or self._decoder.unused_data:
            raise _invalid_compressed_content_error(self._endpoint)

    def _decode_identity(self, raw: bytes) -> Iterator[bytes]:
        offset = 0
        while offset < len(raw):
            end = offset + self._next_output_limit()
            decoded = raw[offset:end]
            offset = end
            yield self._accept(decoded)

    def _decode_compressed(self, raw: bytes) -> Iterator[bytes]:
        if self._decoder is None:
            message = "compressed response decoder is not initialized"
            raise AssertionError(message)
        pending = raw
        while pending:
            if self._decoder.eof:
                self._decoder = zlib.decompressobj(zlib.MAX_WBITS | 16)
            try:
                decoded = self._decoder.decompress(pending, self._next_output_limit())
            except zlib.error as exc:
                raise _invalid_compressed_content_error(self._endpoint) from exc
            pending = self._decoder.unconsumed_tail
            if self._decoder.unused_data:
                pending = self._decoder.unused_data
            if decoded:
                yield self._accept(decoded)

    def _next_output_limit(self) -> int:
        return min(_DECODE_CHUNK_BYTES, self._max_bytes - self._decoded_bytes + 1)

    def _accept(self, decoded: bytes) -> bytes:
        if self._decoded_bytes + len(decoded) > self._max_bytes:
            raise _response_too_large_error(self._endpoint)
        self._decoded_bytes += len(decoded)
        return decoded


def _validate_max_response_bytes(max_response_bytes: int) -> None:
    if (
        isinstance(max_response_bytes, bool)
        or not isinstance(max_response_bytes, int)
        or max_response_bytes <= 0
    ):
        message = "max_response_bytes must be a positive integer"
        raise ValueError(message)


def _validate_max_stream_frames(max_stream_frames: int) -> None:
    if (
        isinstance(max_stream_frames, bool)
        or not isinstance(max_stream_frames, int)
        or max_stream_frames <= 0
    ):
        message = "max_stream_frames must be a positive integer"
        raise ValueError(message)


# Use zlib's documented worst-case growth terms plus headroom for gzip/zlib wrappers.
def _encoded_response_limit(max_bytes: int) -> int:
    return max_bytes + (max_bytes >> 12) + (max_bytes >> 14) + (max_bytes >> 25) + 64


def _iter_bounded_decoded_content(
    content: bytes,
    *,
    max_bytes: int,
    endpoint: str,
) -> Iterator[bytes]:
    if len(content) > max_bytes:
        raise _response_too_large_error(endpoint)
    for offset in range(0, len(content), _DECODE_CHUNK_BYTES):
        yield content[offset : offset + _DECODE_CHUNK_BYTES]


def _validate_content_encoding(response: httpx.Response, *, endpoint: str) -> None:
    _content_encoding(response, endpoint=endpoint)


def _buffered_response(response: httpx.Response, content: bytes) -> httpx.Response:
    headers = response.headers.copy()
    for decoded_header in ("content-encoding", "content-length"):
        if decoded_header in headers:
            del headers[decoded_header]
    return httpx.Response(
        response.status_code,
        headers=headers,
        content=content,
        request=response.request,
    )


def _content_encoding(response: httpx.Response, *, endpoint: str) -> str:
    content_encoding = response.headers.get("content-encoding", "identity").strip().lower()
    if content_encoding in {"", "identity"}:
        return "identity"
    if content_encoding == "gzip":
        return "gzip"
    if content_encoding == "deflate":
        return "deflate"
    message = "response protocol error: unsupported content encoding"
    raise ClientError(
        message,
        code="protocol_error",
        endpoint=endpoint,
    )


def _response_too_large_error(endpoint: str) -> ClientError:
    return ClientError(
        "response exceeds configured size limit",
        code="response_too_large",
        endpoint=endpoint,
    )


def _invalid_compressed_content_error(endpoint: str) -> ClientError:
    return ClientError(
        "response protocol error: invalid compressed content",
        code="protocol_error",
        endpoint=endpoint,
    )


def _invalid_content_length_error(endpoint: str) -> ClientError:
    return ClientError(
        "response protocol error: invalid content-length",
        code="protocol_error",
        endpoint=endpoint,
    )
