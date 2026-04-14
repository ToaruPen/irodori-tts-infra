from __future__ import annotations

from pathlib import Path
from tempfile import gettempdir
from typing import Annotated, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Port = Annotated[int, Field(ge=1, le=65_535)]
PositiveInt = Annotated[int, Field(gt=0)]
PositiveFloat = Annotated[float, Field(gt=0.0)]
DeviceName = Literal["cuda", "cpu", "mps"]
PrecisionName = Literal["bf16", "fp32", "fp16"]
DecodeMode = Literal["batch", "sequential"]


class ClientSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="IRODORI_TTS_CLIENT_", extra="forbid")

    host: str = Field(default="127.0.0.1", min_length=1)
    port: Port = 8923


class ServerSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="IRODORI_TTS_SERVER_", extra="forbid")

    host: str = Field(default="0.0.0.0", min_length=1)  # noqa: S104
    port: Port = 8923


class IrodoriRuntimeSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="IRODORI_TTS_RUNTIME_", extra="forbid")

    checkpoint: str = Field(
        default="Aratako/Irodori-TTS-500M-v2-VoiceDesign",
        min_length=1,
    )
    num_steps: PositiveInt = 30
    cfg_scale_text: PositiveFloat = 3.0
    cfg_scale_caption: PositiveFloat = 3.5
    model_device: DeviceName = "cuda"
    model_precision: PrecisionName = "bf16"
    codec_device: DeviceName = "cuda"
    codec_precision: PrecisionName = "fp32"
    warmup_num_steps: PositiveInt = 30
    warmup_text: str = Field(default="テスト", min_length=1)
    warmup_caption: str = Field(default="女性が話している。", min_length=1)
    decode_mode: DecodeMode = "batch"
    context_kv_cache: bool = True
    compile_model: bool = False


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
