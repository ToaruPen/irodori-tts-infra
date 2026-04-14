from __future__ import annotations

import threading
import time
from queue import Queue

import pytest

from irodori_tts_infra.contracts.synthesis import SynthesisRequest
from irodori_tts_infra.engine.backends.fake import FakeSynthesizer, FakeSynthResponse
from irodori_tts_infra.engine.models import SynthesizedAudio

pytestmark = pytest.mark.unit

DEFAULT_SAMPLE_RATE = 24_000
DELAY_SECONDS = 0.05
THREAD_COUNT = 10


def make_request(text: str = "こんにちは") -> SynthesisRequest:
    return SynthesisRequest(text=text, caption="女性が自然な口調で話している。")


def test_default_response_returns_default_audio() -> None:
    synth = FakeSynthesizer()

    audio = synth.synthesize(make_request())

    assert audio.wav_bytes == b"RIFF\x00\x00\x00\x00WAVEfake"
    assert audio.sample_rate == DEFAULT_SAMPLE_RATE


def test_scripted_responses_are_consumed_in_order() -> None:
    first = SynthesizedAudio(wav_bytes=b"first", sample_rate=16_000)
    second = SynthesizedAudio(wav_bytes=b"second", sample_rate=22_050)
    synth = FakeSynthesizer(
        responses=[
            FakeSynthResponse(audio=first),
            FakeSynthResponse(audio=second),
        ],
    )

    assert synth.synthesize(make_request("一つ目")) == first
    assert synth.synthesize(make_request("二つ目")) == second


def test_scripted_exception_is_raised() -> None:
    error = RuntimeError("backend exploded")
    synth = FakeSynthesizer(responses=[FakeSynthResponse(exception=error)])

    with pytest.raises(RuntimeError, match="backend exploded") as exc_info:
        synth.synthesize(make_request())

    assert exc_info.value is error


def test_call_recording_keeps_input_requests() -> None:
    first = make_request("一つ目")
    second = make_request("二つ目")
    synth = FakeSynthesizer()

    synth.synthesize(first)
    synth.synthesize(second)

    assert synth.calls == [first, second]


def test_delay_is_honored_before_response() -> None:
    synth = FakeSynthesizer(responses=[FakeSynthResponse(delay_seconds=DELAY_SECONDS)])

    started = time.perf_counter()
    synth.synthesize(make_request())
    elapsed = time.perf_counter() - started

    assert elapsed >= DELAY_SECONDS


def test_call_recording_is_thread_safe() -> None:
    expected = [f"audio-{index}".encode() for index in range(THREAD_COUNT)]
    synth = FakeSynthesizer(
        responses=[
            FakeSynthResponse(
                audio=SynthesizedAudio(wav_bytes=wav_bytes, sample_rate=DEFAULT_SAMPLE_RATE),
            )
            for wav_bytes in expected
        ],
    )
    barrier = threading.Barrier(THREAD_COUNT)
    results: Queue[bytes | Exception] = Queue()

    def worker(index: int) -> None:
        try:
            barrier.wait(timeout=1.0)
            results.put(synth.synthesize(make_request(f"本文{index}")).wav_bytes)
        except Exception as exc:  # noqa: BLE001
            results.put(exc)

    threads = [threading.Thread(target=worker, args=(index,)) for index in range(THREAD_COUNT)]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=1.0)
        assert not thread.is_alive()

    collected: list[bytes] = []
    for _ in range(THREAD_COUNT):
        item = results.get_nowait()
        if isinstance(item, Exception):
            raise item
        collected.append(item)

    assert len(synth.calls) == THREAD_COUNT
    assert sorted(collected) == sorted(expected)
    assert sorted(request.text for request in synth.calls) == [
        f"本文{i}" for i in range(THREAD_COUNT)
    ]
