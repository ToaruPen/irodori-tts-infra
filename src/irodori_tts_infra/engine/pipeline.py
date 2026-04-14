from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

from irodori_tts_infra.contracts.synthesis import BatchSynthesisResult, SynthesisResult
from irodori_tts_infra.engine.errors import (
    BackendUnavailableError,
    BackpressureError,
    EmptyBatchError,
    EngineError,
)
from irodori_tts_infra.engine.models import PipelineConfig, SynthesisJob
from irodori_tts_infra.text.models import SegmentKind
from irodori_tts_infra.voice_bank import resolve_segment_caption

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from irodori_tts_infra.engine.protocols import Synthesizer
    from irodori_tts_infra.text.models import Segment
    from irodori_tts_infra.voice_bank import VoiceProfile


class SynthesisPipeline:
    def __init__(
        self,
        synthesizer: Synthesizer,
        voice_profile: VoiceProfile,
        *,
        config: PipelineConfig | None = None,
    ) -> None:
        self._synthesizer = synthesizer
        self._voice_profile = voice_profile
        self._config = config or PipelineConfig()
        self._semaphore = threading.BoundedSemaphore(self._config.capacity)

    @property
    def backend(self) -> Synthesizer:
        return self._synthesizer

    def plan_segment(self, segment_index: int, segment: Segment) -> SynthesisJob:
        caption = (
            self._voice_profile.narrator_caption
            if segment.kind == SegmentKind.NARRATION
            else resolve_segment_caption(segment, self._voice_profile)
        )
        return SynthesisJob(
            segment_index=segment_index,
            text=segment.text,
            caption=caption,
        )

    def synthesize_job(self, job: SynthesisJob) -> SynthesisResult:
        if not self._acquire_slot():
            msg = "backend capacity unavailable"
            raise BackpressureError(msg)

        try:
            request = job.to_request()
            started = time.perf_counter()
            try:
                audio = self._synthesizer.synthesize(request)
                return SynthesisResult(
                    segment_index=job.segment_index,
                    wav_bytes=audio.wav_bytes,
                    elapsed_seconds=round(time.perf_counter() - started, 3),
                )
            except EngineError:
                raise
            except Exception as exc:
                msg = "Backend synthesize failed"
                raise BackendUnavailableError(msg) from exc
        finally:
            self._semaphore.release()

    def synthesize_batch(self, segments: Iterable[Segment]) -> BatchSynthesisResult:
        jobs = self._plan_segments(segments)
        started = time.perf_counter()
        results = [self.synthesize_job(job) for job in jobs]
        return BatchSynthesisResult(
            results=results,
            total_elapsed_seconds=round(time.perf_counter() - started, 3),
        )

    def synthesize_stream(self, segments: Iterable[Segment]) -> Iterator[SynthesisResult]:
        iterator = iter(segments)
        try:
            first_segment = next(iterator)
        except StopIteration as exc:
            msg = "No segments submitted to synthesis pipeline"
            raise EmptyBatchError(msg) from exc

        def _iter() -> Iterator[SynthesisResult]:
            yield self.synthesize_job(self.plan_segment(0, first_segment))
            for index, segment in enumerate(iterator, start=1):
                yield self.synthesize_job(self.plan_segment(index, segment))

        return _iter()

    def _plan_segments(self, segments: Iterable[Segment]) -> list[SynthesisJob]:
        jobs: list[SynthesisJob] = []
        for index, segment in enumerate(segments):
            jobs.append(self.plan_segment(index, segment))
        if not jobs:
            msg = "No segments submitted to synthesis pipeline"
            raise EmptyBatchError(msg)
        return jobs

    def _acquire_slot(self) -> bool:
        timeout = self._config.acquire_timeout_seconds
        if timeout is None:
            return self._semaphore.acquire(blocking=True)
        if timeout == 0:
            return self._semaphore.acquire(blocking=False)
        return self._semaphore.acquire(timeout=timeout)
