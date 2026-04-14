from __future__ import annotations

import subprocess  # noqa: S404
import sys
import threading
import time
from queue import Queue
from typing import TYPE_CHECKING

import pytest

from irodori_tts_infra.engine.backends.fake import FakeSynthesizer, FakeSynthResponse
from irodori_tts_infra.engine.errors import (
    BackendUnavailableError,
    BackpressureError,
    EmptyBatchError,
)
from irodori_tts_infra.engine.models import PipelineConfig, SynthesisJob, SynthesizedAudio
from irodori_tts_infra.engine.pipeline import SynthesisPipeline
from irodori_tts_infra.text.models import Segment, SegmentKind
from irodori_tts_infra.voice_bank import CharacterVoice, VoiceProfile, resolve_segment_caption

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from irodori_tts_infra.contracts.synthesis import SynthesisRequest, SynthesisResult
    from irodori_tts_infra.engine.protocols import Synthesizer

pytestmark = pytest.mark.unit

CONCURRENT_JOB_COUNT = 5
EXPECTED_THIRD_INDEX = 2
MIN_TIMEOUT_SECONDS = 0.04
MAX_TIMEOUT_SECONDS = 0.5
TIMING_DELAY_SECONDS = 0.01


class BlockingSynthesizer:
    def __init__(
        self,
        *,
        release_events: list[threading.Event] | None = None,
        wav_bytes: bytes = b"RIFFblocking",
    ) -> None:
        self.calls: list[SynthesisRequest] = []
        self.enter_events: list[threading.Event] = []
        self.max_in_flight = 0
        self._in_flight = 0
        self._lock = threading.Lock()
        self._release_events = release_events
        self._wav_bytes = wav_bytes

    def synthesize(self, request: SynthesisRequest) -> SynthesizedAudio:
        with self._lock:
            call_index = len(self.calls)
            self.calls.append(request)
            self.enter_events.append(threading.Event())
            self._in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self._in_flight)
            enter_event = self.enter_events[call_index]
            release_event = self._release_event(call_index)

        enter_event.set()
        release_event.wait()

        with self._lock:
            self._in_flight -= 1

        return SynthesizedAudio(wav_bytes=self._wav_bytes, sample_rate=24_000)

    def _release_event(self, call_index: int) -> threading.Event:
        if self._release_events is None:
            event = threading.Event()
            event.set()
            return event
        return self._release_events[call_index]


def profile() -> VoiceProfile:
    return VoiceProfile(
        characters={
            "ミカ": CharacterVoice(
                name="ミカ",
                caption="若い女性が、明るく楽しそうに話している。若々しい声。",
            ),
        },
        narrator_caption="落ち着いた大人の女性が読み上げている。",
        generic_dialogue_caption="若い人が自然な口調で話している。",
    )


def narration(text: str = "地の文です。") -> Segment:
    return Segment(kind=SegmentKind.NARRATION, text=text)


def dialogue(
    text: str = "台詞です。",
    *,
    speaker: str | None = "ミカ",
    direction: str = "",
) -> Segment:
    return Segment(kind=SegmentKind.DIALOGUE, text=text, speaker=speaker, direction=direction)


def make_pipeline(
    synthesizer: Synthesizer | None = None,
    *,
    config: PipelineConfig | None = None,
) -> SynthesisPipeline:
    return SynthesisPipeline(synthesizer or FakeSynthesizer(), profile(), config=config)


def make_job(segment_index: int = 0) -> SynthesisJob:
    return SynthesisJob(
        segment_index=segment_index,
        text=f"本文{segment_index}",
        caption="声の説明。",
    )


def wait_for(event: threading.Event, message: str) -> None:
    assert event.wait(timeout=1.0), message


def wait_for_call(fake: BlockingSynthesizer, call_index: int) -> threading.Event:
    deadline = time.monotonic() + 1.0
    while len(fake.enter_events) <= call_index and time.monotonic() < deadline:
        time.sleep(0.001)
    assert len(fake.enter_events) > call_index
    event = fake.enter_events[call_index]
    wait_for(event, f"call {call_index} did not enter backend")
    return event


def _run_in_thread(
    fn: Callable[[], object],
) -> tuple[threading.Thread, Queue[object]]:
    results: Queue[object] = Queue()

    def worker() -> None:
        try:
            results.put(fn())
        except BaseException as exc:  # noqa: BLE001
            results.put(exc)

    thread = threading.Thread(target=worker)
    thread.start()
    return thread, results


def _join_thread(
    thread: threading.Thread,
    results: Queue[object],
    *,
    timeout: float = 1.0,
) -> object:
    thread.join(timeout=timeout)
    assert not thread.is_alive()
    item = results.get_nowait()
    if isinstance(item, BaseException):
        raise item
    return item


def _run_synthesis_job(
    pipeline: SynthesisPipeline,
    segment_index: int,
) -> Callable[[], SynthesisResult]:
    def run() -> SynthesisResult:
        return pipeline.synthesize_job(make_job(segment_index))

    return run


def probe_available_capacity(pipeline: SynthesisPipeline) -> None:
    semaphore = pipeline._semaphore  # noqa: SLF001
    assert semaphore.acquire(blocking=False)
    semaphore.release()


def pull_next(
    stream: Iterator[SynthesisResult],
    results: Queue[SynthesisResult | Exception],
) -> None:
    try:
        results.put(next(stream))
    except Exception as exc:  # noqa: BLE001
        results.put(exc)


def test_empty_batch_raises_empty_batch_error() -> None:
    pipeline = make_pipeline()

    with pytest.raises(EmptyBatchError, match="No segments"):
        pipeline.synthesize_batch([])


def test_single_narration_segment_uses_narrator_caption() -> None:
    fake = FakeSynthesizer()
    pipeline = make_pipeline(fake)

    result = pipeline.synthesize_batch([narration()])

    assert result.results[0].segment_index == 0
    assert fake.calls[0].caption == profile().narrator_caption


def test_single_dialogue_segment_uses_known_speaker_caption() -> None:
    fake = FakeSynthesizer()
    pipeline = make_pipeline(fake)
    segment = dialogue()

    pipeline.synthesize_batch([segment])

    assert fake.calls[0].caption == resolve_segment_caption(segment, profile())


def test_unknown_speaker_uses_generic_dialogue_caption() -> None:
    fake = FakeSynthesizer()
    pipeline = make_pipeline(fake)

    pipeline.synthesize_batch([dialogue(speaker="不明")])

    assert fake.calls[0].caption == profile().generic_dialogue_caption


def test_directed_dialogue_injects_direction_into_caption() -> None:
    fake = FakeSynthesizer()
    pipeline = make_pipeline(fake)
    segment = dialogue(direction="小声で")

    pipeline.synthesize_batch([segment])

    assert fake.calls[0].caption == resolve_segment_caption(segment, profile())
    assert "小声で話している" in fake.calls[0].caption


def test_ordering_preserves_submission_indices() -> None:
    segments = [narration("一つ目"), dialogue("二つ目"), narration("三つ目")]

    result = make_pipeline().synthesize_batch(segments)

    assert [item.segment_index for item in result.results] == [0, 1, 2]


def test_elapsed_timing_is_measured_for_results_and_batch() -> None:
    pipeline = make_pipeline(
        FakeSynthesizer(responses=[FakeSynthResponse(delay_seconds=TIMING_DELAY_SECONDS)]),
    )

    result = pipeline.synthesize_batch([narration()])

    assert result.results[0].elapsed_seconds >= TIMING_DELAY_SECONDS
    assert result.total_elapsed_seconds >= result.results[0].elapsed_seconds


def test_backend_wraps_non_engine_exception() -> None:
    error = RuntimeError("runtime died")
    pipeline = make_pipeline(FakeSynthesizer(responses=[FakeSynthResponse(exception=error)]))

    with pytest.raises(BackendUnavailableError, match="Backend synthesize failed") as exc_info:
        pipeline.synthesize_job(make_job())

    assert exc_info.value.__cause__ is error


def test_backend_reraises_engine_error_unchanged() -> None:
    error = BackendUnavailableError("boom")
    pipeline = make_pipeline(FakeSynthesizer(responses=[FakeSynthResponse(exception=error)]))

    with pytest.raises(BackendUnavailableError, match="boom") as exc_info:
        pipeline.synthesize_job(make_job())

    assert exc_info.value is error
    assert exc_info.value.__cause__ is None


def test_capacity_one_serializes_concurrent_jobs_event_driven() -> None:
    release_event = threading.Event()
    fake = BlockingSynthesizer(release_events=[release_event] * CONCURRENT_JOB_COUNT)
    pipeline = make_pipeline(fake)
    workers = [
        _run_in_thread(_run_synthesis_job(pipeline, index)) for index in range(CONCURRENT_JOB_COUNT)
    ]

    wait_for_call(fake, 0)
    release_event.set()
    for thread, results in workers:
        _join_thread(thread, results)

    assert fake.max_in_flight == 1
    assert len(fake.calls) == CONCURRENT_JOB_COUNT


def test_semaphore_is_released_after_success() -> None:
    pipeline = make_pipeline(config=PipelineConfig(acquire_timeout_seconds=0))

    for index in range(3):
        pipeline.synthesize_job(make_job(index))
        probe_available_capacity(pipeline)


def test_semaphore_is_released_after_backend_exception() -> None:
    pipeline = make_pipeline(
        FakeSynthesizer(responses=[FakeSynthResponse(exception=RuntimeError("dead"))]),
        config=PipelineConfig(acquire_timeout_seconds=0),
    )

    with pytest.raises(BackendUnavailableError):
        pipeline.synthesize_job(make_job())
    probe_available_capacity(pipeline)


def test_backpressure_timeout_zero_rejects_immediately() -> None:
    release_event = threading.Event()
    fake = BlockingSynthesizer(release_events=[release_event])
    pipeline = make_pipeline(fake, config=PipelineConfig(acquire_timeout_seconds=0))
    holder, holder_results = _run_in_thread(lambda: pipeline.synthesize_job(make_job()))
    wait_for_call(fake, 0)

    try:
        with pytest.raises(BackpressureError, match="capacity"):
            pipeline.synthesize_job(make_job(1))
    finally:
        release_event.set()
        _join_thread(holder, holder_results)


def test_backpressure_bounded_timeout_raises_within_wide_window() -> None:
    release_event = threading.Event()
    fake = BlockingSynthesizer(release_events=[release_event])
    pipeline = make_pipeline(fake, config=PipelineConfig(acquire_timeout_seconds=0.05))
    holder, holder_results = _run_in_thread(lambda: pipeline.synthesize_job(make_job()))
    wait_for_call(fake, 0)

    started = time.perf_counter()
    try:
        with pytest.raises(BackpressureError):
            pipeline.synthesize_job(make_job(1))
    finally:
        release_event.set()
        _join_thread(holder, holder_results)
    elapsed = time.perf_counter() - started

    assert MIN_TIMEOUT_SECONDS <= elapsed <= MAX_TIMEOUT_SECONDS


def test_bounded_timeout_succeeds_when_slot_frees_before_timeout() -> None:
    first_release = threading.Event()
    second_release = threading.Event()
    second_release.set()
    fake = BlockingSynthesizer(release_events=[first_release, second_release])
    pipeline = make_pipeline(fake, config=PipelineConfig(acquire_timeout_seconds=0.5))
    holder = threading.Thread(target=pipeline.synthesize_job, args=(make_job(),))
    holder.start()
    wait_for_call(fake, 0)

    results: Queue[SynthesisResult | Exception] = Queue()
    claimant = threading.Thread(
        target=lambda: results.put(pipeline.synthesize_job(make_job(1))),
    )
    claimant.start()
    first_release.set()
    holder.join(timeout=1.0)
    claimant.join(timeout=1.0)

    item = results.get_nowait()
    assert not isinstance(item, Exception)
    assert item.segment_index == 1


def test_synthesize_stream_yields_results_in_order() -> None:
    pipeline = make_pipeline()

    results = list(pipeline.synthesize_stream([narration("一"), dialogue("二"), narration("三")]))

    assert [result.segment_index for result in results] == [0, 1, 2]


def test_synthesize_stream_consumes_input_incrementally() -> None:
    yielded_count = 0
    pipeline = make_pipeline()

    def segments() -> Iterator[Segment]:
        nonlocal yielded_count
        yielded_count += 1
        yield narration("一")
        yielded_count += 1
        yield dialogue("二")
        yielded_count += 1
        yield narration("三")

    stream = pipeline.synthesize_stream(segments())

    assert yielded_count == 1
    assert next(stream).segment_index == 0
    assert yielded_count == 1


def test_streaming_iterator_yields_incrementally() -> None:
    releases = [threading.Event() for _ in range(3)]
    fake = BlockingSynthesizer(release_events=releases)
    stream = make_pipeline(fake).synthesize_stream(
        [narration("一"), dialogue("二"), narration("三")],
    )
    results: Queue[SynthesisResult | Exception] = Queue()

    first = threading.Thread(target=pull_next, args=(stream, results))
    first.start()
    wait_for_call(fake, 0)
    assert len(fake.enter_events) == 1
    releases[0].set()
    first.join(timeout=1.0)
    first_result = results.get_nowait()
    assert not isinstance(first_result, Exception)
    assert first_result.segment_index == 0
    assert len(fake.enter_events) == 1

    second = threading.Thread(target=pull_next, args=(stream, results))
    second.start()
    wait_for_call(fake, 1)
    releases[1].set()
    second.join(timeout=1.0)
    second_result = results.get_nowait()
    assert not isinstance(second_result, Exception)
    assert second_result.segment_index == 1

    third = threading.Thread(target=pull_next, args=(stream, results))
    third.start()
    wait_for_call(fake, 2)
    releases[2].set()
    third.join(timeout=1.0)
    third_result = results.get_nowait()
    assert not isinstance(third_result, Exception)
    assert third_result.segment_index == EXPECTED_THIRD_INDEX


def test_plan_segment_is_deterministic() -> None:
    pipeline = make_pipeline()
    segment = dialogue(direction="慌てて")

    assert pipeline.plan_segment(0, segment) == pipeline.plan_segment(0, segment)


def test_engine_import_is_lightweight() -> None:
    code = (
        "import sys\n"
        "import irodori_tts_infra.engine\n"
        "import irodori_tts_infra.engine.backends.irodori\n"
        'blocked = {"irodori_tts", "huggingface_hub", "torch", "fastapi", "httpx", "uvicorn"}\n'
        "loaded = blocked & set(sys.modules)\n"
        "print(loaded)\n"
        'assert not loaded, f"heavy modules loaded: {loaded}"\n'
    )

    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_synthesis_job_maps_to_contract_request() -> None:
    job = SynthesisJob(
        segment_index=3,
        text="本文",
        caption="声の説明。",
        num_steps=24,
        cfg_scale_text=2.5,
        cfg_scale_caption=4.0,
        no_ref=False,
    )

    request = job.to_request()

    assert request.text == job.text
    assert request.caption == job.caption
    assert request.num_steps == job.num_steps
    assert request.cfg_scale_text == pytest.approx(job.cfg_scale_text)
    assert request.cfg_scale_caption == pytest.approx(job.cfg_scale_caption)
    assert request.no_ref is job.no_ref


def test_synthesize_job_maps_backend_audio_to_contract_result() -> None:
    pipeline = make_pipeline(
        FakeSynthesizer(
            responses=[
                FakeSynthResponse(
                    audio=SynthesizedAudio(wav_bytes=b"RIFFaudio", sample_rate=48_000),
                ),
            ],
        ),
    )

    result = pipeline.synthesize_job(make_job(EXPECTED_THIRD_INDEX))

    assert result.segment_index == EXPECTED_THIRD_INDEX
    assert result.wav_bytes == b"RIFFaudio"
    assert result.elapsed_seconds >= 0
    assert result.content_type == "audio/wav"
    assert not hasattr(SynthesizedAudio(wav_bytes=b"x", sample_rate=1), "elapsed_seconds")


def test_pipeline_config_validates_capacity_and_timeout() -> None:
    with pytest.raises(TypeError, match="capacity must be an int >= 1"):
        PipelineConfig(capacity=1.5)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="capacity must be an int >= 1"):
        PipelineConfig(capacity=True)

    with pytest.raises(ValueError, match="capacity must be >= 1"):
        PipelineConfig(capacity=0)

    with pytest.raises(ValueError, match="acquire_timeout_seconds must be None or >= 0"):
        PipelineConfig(acquire_timeout_seconds=-1)
