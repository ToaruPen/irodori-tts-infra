from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from irodori_tts_infra.engine.models import ResolvedSynthesisRequest, SynthesizedAudio


@runtime_checkable
class Synthesizer(Protocol):
    def synthesize(self, request: ResolvedSynthesisRequest) -> SynthesizedAudio: ...
