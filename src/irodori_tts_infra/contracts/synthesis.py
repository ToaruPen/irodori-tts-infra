from __future__ import annotations

import json
from typing import Literal, Self

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)


class _ContractModel(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        ser_json_bytes="base64",
        val_json_bytes="base64",
    )


class SynthesisRequest(_ContractModel):
    text: str = Field(min_length=1)
    caption: str = Field(min_length=1)
    num_steps: int = Field(default=30, gt=0)
    cfg_scale_text: float = Field(default=3.0, gt=0.0)
    cfg_scale_caption: float = Field(default=3.5, gt=0.0)
    no_ref: bool = True

    @field_validator("text", "caption")
    @classmethod
    def _reject_blank_text(cls, value: str) -> str:
        if not value.strip():
            msg = "text fields must not be blank"
            raise ValueError(msg)
        return value


class SynthesisSegment(SynthesisRequest):
    segment_index: int = Field(ge=0)


class BatchSynthesisRequest(_ContractModel):
    segments: list[SynthesisSegment] = Field(min_length=1)


class SynthesisResult(_ContractModel):
    segment_index: int = Field(ge=0)
    wav_bytes: bytes
    elapsed_seconds: float = Field(ge=0.0)
    content_type: Literal["audio/wav"] = "audio/wav"


class BatchSynthesisResult(_ContractModel):
    results: list[SynthesisResult] = Field(min_length=1)
    total_elapsed_seconds: float = Field(ge=0.0)

    @model_validator(mode="after")
    def _validate_ordered_results(self) -> Self:
        actual = [result.segment_index for result in self.results]
        expected = list(range(len(self.results)))
        if actual != expected:
            msg = "batch results must be ordered by segment_index starting at 0"
            raise ValueError(msg)
        return self


STREAM_HEADER_VERSION = 1
MAX_CHUNK_SIZE_BYTES = 4 * 1024 * 1024
MAX_SEGMENT_INDEX = (2**32) - 1


def _encode_ndjson_line(payload: dict[str, object]) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


class StreamChunkHeader(_ContractModel):
    kind: Literal["chunk"] = "chunk"
    header_version: int = Field(
        default=STREAM_HEADER_VERSION,
        ge=1,
        serialization_alias="v",
        validation_alias=AliasChoices("header_version", "v"),
    )
    segment_index: int = Field(
        ge=0,
        le=MAX_SEGMENT_INDEX,
        serialization_alias="index",
        validation_alias=AliasChoices("segment_index", "index"),
    )
    byte_length: int = Field(
        ge=0,
        le=MAX_CHUNK_SIZE_BYTES,
        serialization_alias="nbytes",
        validation_alias=AliasChoices("byte_length", "nbytes"),
    )
    final: bool = Field(
        default=False,
        serialization_alias="final",
        validation_alias=AliasChoices("final"),
    )
    elapsed_seconds: float = Field(
        default=0.0,
        ge=0.0,
        serialization_alias="elapsed",
        validation_alias=AliasChoices("elapsed_seconds", "elapsed"),
    )
    error_code: Literal["backend_unavailable", "backpressure"] | None = None

    @field_serializer("elapsed_seconds", when_used="json")
    def _serialize_elapsed(self, value: float) -> float:  # noqa: PLR6301
        return round(value, 3)

    def to_bytes(self) -> bytes:
        return _encode_ndjson_line(self.model_dump(mode="json", by_alias=True, exclude_none=True))

    @classmethod
    def from_bytes(cls, data: bytes) -> Self:
        # Tolerate callers that pass the whole framed line including the trailing newline
        # as well as callers that pre-strip it; reconstruction must be deterministic either way.
        return cls.model_validate_json(data.rstrip(b"\n"))


class StreamHandshakeHeader(_ContractModel):
    kind: Literal["handshake"] = "handshake"
    header_version: int = Field(
        default=STREAM_HEADER_VERSION,
        ge=1,
        serialization_alias="v",
        validation_alias=AliasChoices("header_version", "v"),
    )
    max_chunk_size: int = Field(
        ge=1,
        le=MAX_CHUNK_SIZE_BYTES,
    )

    def to_bytes(self) -> bytes:
        return _encode_ndjson_line(self.model_dump(mode="json", by_alias=True))

    @classmethod
    def from_bytes(cls, data: bytes) -> Self:
        return cls.model_validate_json(data.rstrip(b"\n"))
