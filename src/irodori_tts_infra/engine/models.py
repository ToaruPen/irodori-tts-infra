from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import Field, field_validator

from irodori_tts_infra.contracts.synthesis import IrodoriStyle, SynthesisRequest


@dataclass(frozen=True, slots=True)
class SynthesizedAudio:
    wav_bytes: bytes
    sample_rate: int


class ResolvedSynthesisRequest(SynthesisRequest):
    ref_embed: str = Field(min_length=1)

    @field_validator("ref_embed")
    @classmethod
    def _normalize_ref_embed(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            msg = "ref_embed must not be blank"
            raise ValueError(msg)
        return stripped


@dataclass(frozen=True, slots=True)
class SynthesisJob:
    segment_index: int
    text: str
    speaker: str | None = None
    voice_id: str | None = None
    if_generation: str | None = None
    ref_embed: str | None = None
    require_speaker: bool = False
    num_steps: int = 40
    cfg_scale_text: float = 3.0
    cfg_scale_caption: float = 3.0
    cfg_scale_speaker: float = 5.0
    style: IrodoriStyle = "neutral"
    seed: int | None = None
    duration_scale: float = 1.0
    num_candidates: int = 1
    t_schedule_mode: Literal["linear", "sway"] = "linear"
    sway_coeff: float = -1.0

    def to_request(self, *, ref_embed: str | None = None) -> ResolvedSynthesisRequest:
        resolved_ref_embed = ref_embed if ref_embed is not None else self.ref_embed
        if resolved_ref_embed is None:
            msg = "resolved ref_embed is required"
            raise ValueError(msg)
        return ResolvedSynthesisRequest(
            text=self.text,
            speaker=self.speaker,
            voice_id=self.voice_id,
            if_generation=self.if_generation,
            ref_embed=resolved_ref_embed,
            num_steps=self.num_steps,
            cfg_scale_text=self.cfg_scale_text,
            cfg_scale_caption=self.cfg_scale_caption,
            cfg_scale_speaker=self.cfg_scale_speaker,
            style=self.style,
            seed=self.seed,
            duration_scale=self.duration_scale,
            num_candidates=self.num_candidates,
            t_schedule_mode=self.t_schedule_mode,
            sway_coeff=self.sway_coeff,
        )


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    capacity: int = 1
    acquire_timeout_seconds: float | None = None
    generation: str = "unconfigured"

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
        if not isinstance(self.generation, str) or not self.generation.strip():
            msg = "generation must be a non-blank string"
            raise ValueError(msg)
        object.__setattr__(self, "generation", self.generation.strip())
