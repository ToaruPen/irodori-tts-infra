from __future__ import annotations

import gzip
from typing import TYPE_CHECKING

from irodori_tts_infra.client.errors import (
    ClientBackpressureError,
    ClientError,
    ClientUnavailableError,
)
from irodori_tts_infra.contracts import StreamChunkHeader, StreamHandshakeHeader

if TYPE_CHECKING:
    from irodori_tts_infra.contracts import StreamErrorCode

MAX_TEST_CHUNK_SIZE = 4
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


def terminal_error_framed(
    error_code: StreamErrorCode,
    *,
    max_chunk_size: int = MAX_TEST_CHUNK_SIZE,
) -> bytes:
    return (
        StreamHandshakeHeader(max_chunk_size=max_chunk_size).to_bytes()
        + StreamChunkHeader(
            segment_index=0,
            byte_length=0,
            final=True,
            error_code=error_code,
        ).to_bytes()
    )


# The 29-byte first stored block starts with the zlib-valid header bytes 08 1d.
def raw_deflate_with_zlib_header_collision(decoded: bytes) -> bytes:
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


# Each empty stored block is five bytes, enabling exact and one-block-over limits.
def empty_raw_deflate_blocks(count: int) -> bytes:
    assert count > 0
    nonfinal = b"\x00\x00\x00\xff\xff"
    final = b"\x01\x00\x00\xff\xff"
    return nonfinal * (count - 1) + final


# Rewriting FLG to FEXTRA permits deterministic gzip header-size boundary tests.
def gzip_member(payload: bytes = b"", *, extra: bytes = b"") -> bytes:
    encoded = gzip.compress(payload)
    if not extra:
        return encoded
    header = encoded[:3] + b"\x04" + encoded[4:10]
    return header + len(extra).to_bytes(2, "little") + extra + encoded[10:]
