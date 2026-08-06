from __future__ import annotations

import zlib
from typing import TYPE_CHECKING, cast

import httpx
import pytest

from irodori_tts_infra.client._response import (  # noqa: PLC2701 - white-box decoder tests
    BoundedResponseDecoder,
    DeflateResponseDecoder,
)
from irodori_tts_infra.client._stream import (  # noqa: PLC2701 - parser boundary test
    MAX_STREAM_HEADER_BYTES,
    StreamPayloadParser,
)
from irodori_tts_infra.client.errors import ClientError, build_stream_error
from irodori_tts_infra.contracts import StreamChunkHeader, StreamHandshakeHeader

if TYPE_CHECKING:
    from irodori_tts_infra.contracts import StreamErrorCode

pytestmark = pytest.mark.unit

ZLIB_STREAM_WITH_RAW_COLLISION = bytes.fromhex(
    "785edba124b73bff6ac8bbcf0c85c5ad002d810696",
)
ZLIB_STREAM_WITH_RAW_COLLISION_PAYLOAD = bytes.fromhex("b8221ebb6fd554eef300717385")


@pytest.mark.parametrize("content_length", ["invalid", "-1", "1, 1"])
def test_bounded_response_decoder_rejects_invalid_content_length(
    content_length: str,
) -> None:
    response = httpx.Response(200, headers={"content-length": content_length})

    with pytest.raises(ClientError) as raised:
        BoundedResponseDecoder(
            response,
            max_bytes=1024,
            endpoint="/health",
        )

    assert raised.value.code == "protocol_error"
    assert raised.value.endpoint == "/health"


def test_bounded_response_decoder_rejects_oversized_numeric_content_length() -> None:
    response = httpx.Response(200, headers={"content-length": "9" * 4_301})

    with pytest.raises(ClientError) as raised:
        BoundedResponseDecoder(
            response,
            max_bytes=1024,
            endpoint="/health",
        )

    assert raised.value.code == "response_too_large"
    assert raised.value.endpoint == "/health"


def test_deflate_decoder_rejects_trailing_zlib_bytes() -> None:
    decoder = DeflateResponseDecoder(max_bytes=1024, endpoint="/health")

    with pytest.raises(ClientError) as raised:
        list(decoder.decode(zlib.compress(b"audio") + b"trailing"))

    assert raised.value.code == "protocol_error"


def test_deflate_decoder_rejects_when_both_formats_are_invalid() -> None:
    decoder = DeflateResponseDecoder(max_bytes=1024, endpoint="/health")

    with pytest.raises(ClientError) as raised:
        [*decoder.decode(b"\xff"), *decoder.finish()]

    assert raised.value.code == "protocol_error"


def test_deflate_decoder_selects_zlib_candidate_during_finish() -> None:
    # This valid zlib stream is also an incomplete raw-deflate prefix until EOF.
    decoder = DeflateResponseDecoder(max_bytes=1024, endpoint="/health")

    assert list(decoder.decode(ZLIB_STREAM_WITH_RAW_COLLISION)) == []
    assert b"".join(decoder.finish()) == ZLIB_STREAM_WITH_RAW_COLLISION_PAYLOAD


def test_deflate_decoder_defers_size_error_until_format_is_selected() -> None:
    decoder = DeflateResponseDecoder(max_bytes=1, endpoint="/health")

    assert list(decoder.decode(ZLIB_STREAM_WITH_RAW_COLLISION)) == []
    with pytest.raises(ClientError) as raised:
        list(decoder.finish())

    assert raised.value.code == "response_too_large"


def test_deflate_decoder_accepts_raw_stream_split_after_selection() -> None:
    # The first two bytes invalidate zlib while leaving a selected raw stream in progress.
    encoded = bytes.fromhex("4b4c4a4e494d0300")
    decoder = DeflateResponseDecoder(max_bytes=1024, endpoint="/health")

    decoded = [*decoder.decode(encoded[:2]), *decoder.decode(encoded[2:]), *decoder.finish()]

    assert b"".join(decoded) == b"abcdef"


def test_build_stream_error_rejects_unmapped_contract_code_without_key_error() -> None:
    unknown = cast("StreamErrorCode", "future_error")

    error = build_stream_error(unknown, endpoint="/synthesize_stream")

    assert error.code == "protocol_error"
    assert error.endpoint == "/synthesize_stream"


def test_stream_payload_parser_accepts_header_at_size_limit() -> None:
    handshake = StreamHandshakeHeader(max_chunk_size=4).to_bytes().rstrip(b"\n")
    padded_handshake = handshake.ljust(MAX_STREAM_HEADER_BYTES - 1, b" ") + b"\n"
    final = StreamChunkHeader(segment_index=0, byte_length=0, final=True).to_bytes()
    parser = StreamPayloadParser(
        health_max_chunk_size=4,
        max_response_bytes=1024,
        max_stream_frames=2,
    )

    payloads = [*parser.feed(padded_handshake + final), *parser.finish()]

    assert payloads == [b""]
