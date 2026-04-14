from __future__ import annotations

import time
from typing import TYPE_CHECKING, Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from irodori_tts_infra.contracts import (
    BatchSynthesisRequest,
    BatchSynthesisResult,
    StreamChunkHeader,
    StreamHandshakeHeader,
    SynthesisRequest,
    SynthesisResult,
    SynthesisSegment,
)
from irodori_tts_infra.engine.models import SynthesisJob
from irodori_tts_infra.engine.pipeline import SynthesisPipeline
from irodori_tts_infra.server.dependencies import get_max_chunk_size, get_pipeline

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

router = APIRouter()

PipelineDependency = Annotated[SynthesisPipeline, Depends(get_pipeline)]
MaxChunkSizeDependency = Annotated[int, Depends(get_max_chunk_size)]


@router.post("/synthesize", response_model=SynthesisResult)
def synthesize(request: SynthesisRequest, pipeline: PipelineDependency) -> SynthesisResult:
    return pipeline.synthesize_job(_job_from_request(request, segment_index=0))


@router.post("/synthesize_batch", response_model=BatchSynthesisResult)
def synthesize_batch(
    request: BatchSynthesisRequest,
    pipeline: PipelineDependency,
) -> BatchSynthesisResult:
    _validate_segment_order(request.segments)
    started = time.perf_counter()
    results = [pipeline.synthesize_job(_job_from_segment(segment)) for segment in request.segments]
    return BatchSynthesisResult(
        results=results,
        total_elapsed_seconds=round(time.perf_counter() - started, 3),
    )


@router.post("/synthesize_stream")
def synthesize_stream(
    request: BatchSynthesisRequest,
    pipeline: PipelineDependency,
    max_chunk_size: MaxChunkSizeDependency,
) -> StreamingResponse:
    _validate_segment_order(request.segments)
    return StreamingResponse(
        _frame_stream(request.segments, pipeline, max_chunk_size),
        media_type="application/octet-stream",
    )


def _frame_stream(
    segments: Sequence[SynthesisSegment],
    pipeline: SynthesisPipeline,
    max_chunk_size: int,
) -> Iterator[bytes]:
    yield StreamHandshakeHeader(max_chunk_size=max_chunk_size).to_bytes()

    for segment in segments:
        result = pipeline.synthesize_job(_job_from_segment(segment))
        chunks = _split_wav_bytes(result.wav_bytes, max_chunk_size)
        for chunk_offset, chunk in enumerate(chunks):
            is_final = chunk_offset == len(chunks) - 1
            yield StreamChunkHeader(
                segment_index=segment.segment_index,
                byte_length=len(chunk),
                final=is_final,
                elapsed_seconds=result.elapsed_seconds,
            ).to_bytes()
            yield chunk


def _split_wav_bytes(wav_bytes: bytes, max_chunk_size: int) -> list[bytes]:
    return [
        wav_bytes[index : index + max_chunk_size]
        for index in range(0, len(wav_bytes), max_chunk_size)
    ]


def _validate_segment_order(segments: Sequence[SynthesisSegment]) -> None:
    expected = list(range(len(segments)))
    actual = [segment.segment_index for segment in segments]
    if actual != expected:
        msg = "segments must be ordered by segment_index starting at 0"
        raise HTTPException(status_code=422, detail=msg)


def _job_from_request(request: SynthesisRequest, *, segment_index: int) -> SynthesisJob:
    return SynthesisJob(
        segment_index=segment_index,
        text=request.text,
        caption=request.caption,
        num_steps=request.num_steps,
        cfg_scale_text=request.cfg_scale_text,
        cfg_scale_caption=request.cfg_scale_caption,
        no_ref=request.no_ref,
    )


def _job_from_segment(segment: SynthesisSegment) -> SynthesisJob:
    return _job_from_request(segment, segment_index=segment.segment_index)
