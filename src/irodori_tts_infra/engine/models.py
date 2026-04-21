from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from irodori_tts_infra.contracts.synthesis import SynthesisRequest

if TYPE_CHECKING:
    from irodori_tts_infra.voice_bank.models import RVCProfile


@dataclass(frozen=True, slots=True)
class SynthesizedAudio:
    wav_bytes: bytes
    sample_rate: int


@dataclass(frozen=True, slots=True)
class SynthesisJob:
    segment_index: int
    text: str
    caption: str
    num_steps: int = 30
    cfg_scale_text: float = 3.0
    cfg_scale_caption: float = 3.5
    no_ref: bool = True
    rvc: RVCProfile | None = None

    def to_request(self) -> SynthesisRequest:
        return SynthesisRequest(
            text=self.text,
            caption=self.caption,
            num_steps=self.num_steps,
            cfg_scale_text=self.cfg_scale_text,
            cfg_scale_caption=self.cfg_scale_caption,
            no_ref=self.no_ref,
        )


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    capacity: int = 1
    acquire_timeout_seconds: float | None = None

    def __post_init__(self) -> None:
        if isinstance(self.capacity, bool) or not isinstance(self.capacity, int):
            msg = "capacity must be an int >= 1"
            raise TypeError(msg)
        if self.capacity < 1:
            msg = "capacity must be >= 1"
            raise ValueError(msg)
        if self.acquire_timeout_seconds is not None and self.acquire_timeout_seconds < 0:
            msg = "acquire_timeout_seconds must be None or >= 0"
            raise ValueError(msg)
