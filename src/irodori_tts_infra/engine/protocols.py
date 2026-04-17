from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from irodori_tts_infra.contracts.synthesis import SynthesisRequest
    from irodori_tts_infra.engine.models import SynthesizedAudio
    from irodori_tts_infra.voice_bank.models import RVCProfile


@runtime_checkable
class Synthesizer(Protocol):
    def synthesize(self, request: SynthesisRequest) -> SynthesizedAudio: ...


@runtime_checkable
class VoiceConverter(Protocol):
    def convert(self, audio: SynthesizedAudio, *, profile: RVCProfile) -> SynthesizedAudio: ...
