from __future__ import annotations

import io
import wave
from typing import Literal

import numpy as np
import pytest

from irodori_tts_infra.engine.backends.fake import FakeSynthesizer, FakeSynthResponse
from irodori_tts_infra.engine.beep_guard import (
    DEFAULT_MAX_ATTEMPTS,
    BeepGuardedSynthesizer,
    BeepGuardExhaustedError,
)
from irodori_tts_infra.engine.errors import BackendUnavailableError
from irodori_tts_infra.engine.models import ResolvedSynthesisRequest, SynthesizedAudio

pytestmark = pytest.mark.unit

SAMPLE_RATE = 48_000
FIXED_SEED = 7


def _wav(*, beep: bool) -> bytes:
    time = np.arange(SAMPLE_RATE) / SAMPLE_RATE
    samples = 0.05 * np.sin(2 * np.pi * 220.0 * time) + 0.05 * np.sin(2 * np.pi * 447.0 * time)
    if beep:
        samples[12_000:18_000] = 0.3 * np.sin(2 * np.pi * 1000.6 * time[:6_000])
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(SAMPLE_RATE)
        target.writeframes((samples * 32767).astype("<i2").tobytes())
    return buffer.getvalue()


def _responses(*kinds: Literal["beep", "clean"]) -> list[FakeSynthResponse]:
    return [
        FakeSynthResponse(
            audio=SynthesizedAudio(wav_bytes=_wav(beep=kind == "beep"), sample_rate=SAMPLE_RATE)
        )
        for kind in kinds
    ]


def _request(seed: int | None) -> ResolvedSynthesisRequest:
    return ResolvedSynthesisRequest(text="テスト", ref_embed="voice.speaker.safetensors", seed=seed)


def test_clean_audio_is_returned_after_one_attempt() -> None:
    backend = FakeSynthesizer(responses=_responses("clean"))

    audio = BeepGuardedSynthesizer(backend).synthesize(_request(FIXED_SEED))

    assert audio.wav_bytes == _wav(beep=False)
    assert [call.seed for call in backend.calls] == [FIXED_SEED]


def test_beep_triggers_regeneration_with_the_next_seed() -> None:
    backend = FakeSynthesizer(responses=_responses("beep", "beep", "clean"))

    audio = BeepGuardedSynthesizer(backend).synthesize(_request(FIXED_SEED))

    assert audio.wav_bytes == _wav(beep=False)
    assert [call.seed for call in backend.calls] == [FIXED_SEED, FIXED_SEED + 1, FIXED_SEED + 2]


def test_unseeded_request_stays_unseeded_when_regenerating() -> None:
    backend = FakeSynthesizer(responses=_responses("beep", "clean"))

    BeepGuardedSynthesizer(backend).synthesize(_request(None))

    assert [call.seed for call in backend.calls] == [None, None]


def test_exhausted_attempts_fail_closed_as_backend_unavailable() -> None:
    always_beeping: list[Literal["beep", "clean"]] = ["beep"] * (DEFAULT_MAX_ATTEMPTS + 1)
    backend = FakeSynthesizer(responses=_responses(*always_beeping))

    with pytest.raises(BeepGuardExhaustedError) as raised:
        BeepGuardedSynthesizer(backend).synthesize(_request(FIXED_SEED))

    assert isinstance(raised.value, BackendUnavailableError)
    assert len(backend.calls) == DEFAULT_MAX_ATTEMPTS


def test_undecodable_audio_fails_closed() -> None:
    backend = FakeSynthesizer()

    with pytest.raises(BackendUnavailableError, match="beep inspection"):
        BeepGuardedSynthesizer(backend).synthesize(_request(FIXED_SEED))


def test_max_attempts_must_be_positive() -> None:
    with pytest.raises(ValueError, match="max_attempts"):
        BeepGuardedSynthesizer(FakeSynthesizer(), max_attempts=0)


def test_warm_up_and_close_are_delegated() -> None:
    class Backend(FakeSynthesizer):
        def __init__(self) -> None:
            super().__init__()
            self.events: list[str] = []

        def warm_up(self, *, ref_embed: str | None = None) -> None:
            self.events.append(f"warm_up:{ref_embed}")

        def close(self) -> None:
            self.events.append("close")

    backend = Backend()
    guarded = BeepGuardedSynthesizer(backend)

    guarded.warm_up(ref_embed="narrator.speaker.safetensors")
    guarded.close()

    assert backend.events == ["warm_up:narrator.speaker.safetensors", "close"]


def test_backends_without_lifecycle_hooks_are_accepted() -> None:
    guarded = BeepGuardedSynthesizer(FakeSynthesizer())

    guarded.warm_up(ref_embed="narrator.speaker.safetensors")
    guarded.close()
