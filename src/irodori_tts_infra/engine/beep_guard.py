from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from irodori_tts_infra.engine.errors import BackendUnavailableError
from irodori_tts_infra.metrics.beep import find_censor_beeps_in_wav

if TYPE_CHECKING:
    from irodori_tts_infra.engine.models import ResolvedSynthesisRequest, SynthesizedAudio
    from irodori_tts_infra.engine.protocols import Synthesizer

DEFAULT_MAX_ATTEMPTS = 4

_logger = structlog.get_logger()


class BeepGuardExhaustedError(BackendUnavailableError):
    pass


class BeepGuardedSynthesizer:
    """Regenerate audio that contains a censor beep; fail closed when every attempt beeps."""

    def __init__(
        self, synthesizer: Synthesizer, *, max_attempts: int = DEFAULT_MAX_ATTEMPTS
    ) -> None:
        if max_attempts < 1:
            msg = "max_attempts must be >= 1"
            raise ValueError(msg)
        self._synthesizer = synthesizer
        self._max_attempts = max_attempts

    def synthesize(self, request: ResolvedSynthesisRequest) -> SynthesizedAudio:
        for attempt in range(self._max_attempts):
            # A fixed seed reproduces the same beep, so every retry moves to the next seed.
            seed = None if request.seed is None else request.seed + attempt
            audio = self._synthesizer.synthesize(request.model_copy(update={"seed": seed}))
            try:
                beeps = find_censor_beeps_in_wav(audio.wav_bytes)
            except ValueError as exc:
                msg = "synthesized audio could not be decoded for beep inspection"
                raise BackendUnavailableError(msg) from exc
            if not beeps:
                return audio
            _logger.warning(
                "censor_beep_detected",
                attempt=attempt + 1,
                max_attempts=self._max_attempts,
                beep_count=len(beeps),
                frequency_hz=round(beeps[0].frequency_hz, 1),
                start_seconds=round(beeps[0].start_seconds, 2),
                end_seconds=round(beeps[0].end_seconds, 2),
            )
        msg = f"censor beep remained after {self._max_attempts} synthesis attempts"
        raise BeepGuardExhaustedError(msg)

    def warm_up(self, *, ref_embed: str | None = None) -> None:
        warm_up = getattr(self._synthesizer, "warm_up", None)
        if callable(warm_up):
            warm_up(ref_embed=ref_embed)

    def close(self) -> None:
        close = getattr(self._synthesizer, "close", None)
        if callable(close):
            close()
