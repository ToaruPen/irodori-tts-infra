from __future__ import annotations

from pathlib import Path
from tempfile import gettempdir
from typing import Annotated, Literal, Self

from pydantic import Field, FiniteFloat, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from irodori_tts_infra.contracts.synthesis import IrodoriStyle  # noqa: TC001 - Pydantic resolves it

Port = Annotated[int, Field(ge=1, le=65_535)]
PositiveInt = Annotated[int, Field(gt=0)]
PositiveFloat = Annotated[float, Field(gt=0.0, allow_inf_nan=False)]
CommitRevision = Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
Sha256Hex = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
DeviceName = Literal["cuda", "cpu", "mps"]
PrecisionName = Literal["bf16", "fp32", "fp16"]
DecodeMode = Literal["batch", "sequential"]
ScheduleMode = Literal["linear", "sway"]
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})


class ClientSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="IRODORI_TTS_CLIENT_", extra="forbid")

    host: str = Field(default="127.0.0.1", min_length=1)
    port: Port = 8924


class ServerSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="IRODORI_TTS_SERVER_", extra="forbid")

    host: str = Field(default="127.0.0.1", min_length=1)
    port: Port = 8924

    @field_validator("host")
    @classmethod
    def _require_loopback_host(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in LOOPBACK_HOSTS:
            msg = "server host must be loopback"
            raise ValueError(msg)
        return normalized


class IrodoriRuntimeSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="IRODORI_TTS_RUNTIME_", extra="forbid")

    checkpoint: str = Field(
        default="Aratako/Irodori-TTS-v4-Small",
        min_length=1,
    )
    checkpoint_revision: CommitRevision = "e4aaac4df355ff560dcd35e0dae272c3a759317b"
    checkpoint_sha256: Sha256Hex = (
        "5863c986345d9f6d20b7d8748fee1af02079c5161cf0c9e52557da0a0c378593"
    )
    checkpoint_tokenizer_json_sha256: Sha256Hex | None = (
        "6a0734cf21c802169defaffe719bc2ef12bb9d0be37e54b61ed27aa89394723d"
    )
    checkpoint_tokenizer_config_sha256: Sha256Hex | None = (
        "d229a271c64de1a7939d20d3665498e873fa91d5ee2edf135d73ec752cb9c9d3"
    )
    num_steps: PositiveInt = 40
    cfg_scale_text: PositiveFloat = 3.0
    cfg_scale_caption: PositiveFloat = 3.0
    cfg_scale_speaker: PositiveFloat = 5.0
    seed: int | None = None
    duration_scale: PositiveFloat = 1.0
    num_candidates: PositiveInt = 1
    t_schedule_mode: ScheduleMode = "linear"
    sway_coeff: FiniteFloat = -1.0
    model_device: DeviceName = "cuda"
    model_precision: PrecisionName = "bf16"
    codec_device: DeviceName = "cuda"
    codec_precision: PrecisionName = "fp32"
    warmup_num_steps: PositiveInt = 40
    warmup_text: str = Field(default="テスト", min_length=1)
    warmup_style: IrodoriStyle = "calm"
    decode_mode: DecodeMode = "sequential"
    context_kv_cache: bool = True
    compile_model: bool = False
    public_generation: str = Field(default="unconfigured", min_length=1)
    emoji_conditioning_supported: bool = True

    @field_validator("checkpoint")
    @classmethod
    def _normalize_checkpoint(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            msg = "checkpoint must not be blank"
            raise ValueError(msg)
        return stripped

    @field_validator("public_generation")
    @classmethod
    def _normalize_public_generation(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            msg = "public_generation must not be blank"
            raise ValueError(msg)
        return stripped

    @model_validator(mode="after")
    def _require_complete_bundled_tokenizer_pins(self) -> Self:
        json_pin = self.checkpoint_tokenizer_json_sha256
        config_pin = self.checkpoint_tokenizer_config_sha256
        if (json_pin is None) != (config_pin is None):
            msg = "bundled tokenizer pins must be both set or both unset"
            raise ValueError(msg)
        return self


class PathSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="IRODORI_TTS_PATH_", extra="forbid")

    temp_wav_dir: Path = Field(default_factory=lambda: Path(gettempdir()) / "irodori-tts-wav")

    @field_validator("temp_wav_dir", mode="before")
    @classmethod
    def _reject_blank_path(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            msg = "temp_wav_dir must not be blank"
            raise ValueError(msg)
        return value
