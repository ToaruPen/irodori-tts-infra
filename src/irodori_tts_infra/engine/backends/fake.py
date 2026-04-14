from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from irodori_tts_infra.engine.models import SynthesizedAudio

if TYPE_CHECKING:
    from irodori_tts_infra.contracts.synthesis import SynthesisRequest


@dataclass(frozen=True, slots=True)
class FakeSynthResponse:
    audio: SynthesizedAudio | None = None
    exception: Exception | None = None
    delay_seconds: float = 0.0


class FakeSynthesizer:
    def __init__(
        self,
        *,
        default_wav: bytes = b"RIFF\x00\x00\x00\x00WAVEfake",
        default_sample_rate: int = 24_000,
        responses: list[FakeSynthResponse] | None = None,
    ) -> None:
        self.calls: list[SynthesisRequest] = []
        self._default_wav = default_wav
        self._default_sample_rate = default_sample_rate
        self._responses = list(responses or [])
        self._lock = threading.Lock()

    def synthesize(self, request: SynthesisRequest) -> SynthesizedAudio:
        with self._lock:
            self.calls.append(request)
            response = self._responses.pop(0) if self._responses else None

        if response is not None:
            if response.delay_seconds > 0:
                time.sleep(response.delay_seconds)
            if response.exception is not None:
                raise response.exception
            if response.audio is not None:
                return response.audio

        return SynthesizedAudio(
            wav_bytes=self._default_wav,
            sample_rate=self._default_sample_rate,
        )
