from __future__ import annotations

import os
import shutil
import subprocess  # noqa: S404
import sys
from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

from irodori_tts_infra.config import (
    ClientSettings,
    IrodoriRuntimeSettings,
    PathSettings,
    RVCSidecarSettings,
    ServerSettings,
)

if TYPE_CHECKING:
    from pathlib import Path

DEFAULT_PORT = 8923
DEFAULT_NUM_STEPS = 30
DEFAULT_CFG_SCALE_TEXT = 3.0
DEFAULT_CFG_SCALE_CAPTION = 3.5
DEFAULT_RVC_PORT = 7865
DEFAULT_RVC_CONNECT_TIMEOUT_SECONDS = 10.0
DEFAULT_RVC_CONVERT_TIMEOUT_SECONDS = 120.0
OVERRIDE_PORT = 9001
OVERRIDE_NUM_STEPS = 24
OVERRIDE_CFG_SCALE_CAPTION = 4.0
OVERRIDE_RVC_PORT = 8877
OVERRIDE_RVC_CONNECT_TIMEOUT_SECONDS = 5.5
OVERRIDE_RVC_CONVERT_TIMEOUT_SECONDS = 240.0


def test_settings_defaults_match_phase1_runtime_plan() -> None:
    client = ClientSettings()
    server = ServerSettings()
    runtime = IrodoriRuntimeSettings()
    paths = PathSettings()
    rvc_sidecar = RVCSidecarSettings()

    assert client.host == "127.0.0.1"
    assert client.port == DEFAULT_PORT
    assert server.host == "0.0.0.0"  # noqa: S104
    assert server.port == DEFAULT_PORT
    assert runtime.checkpoint == "Aratako/Irodori-TTS-500M-v2-VoiceDesign"
    assert runtime.num_steps == DEFAULT_NUM_STEPS
    assert runtime.cfg_scale_text == pytest.approx(DEFAULT_CFG_SCALE_TEXT)
    assert runtime.cfg_scale_caption == pytest.approx(DEFAULT_CFG_SCALE_CAPTION)
    assert runtime.model_device == "cuda"
    assert runtime.model_precision == "bf16"
    assert runtime.codec_device == "cuda"
    assert runtime.codec_precision == "fp32"
    assert runtime.decode_mode == "batch"
    assert runtime.context_kv_cache is True
    assert runtime.compile_model is False
    assert runtime.warmup_num_steps == DEFAULT_NUM_STEPS
    assert runtime.warmup_text == "テスト"
    assert paths.temp_wav_dir.name == "irodori-tts-wav"
    assert rvc_sidecar.url.host == "localhost"
    assert rvc_sidecar.url.port == DEFAULT_RVC_PORT
    assert rvc_sidecar.api_name == "/infer_convert"
    assert rvc_sidecar.connect_timeout_seconds == pytest.approx(DEFAULT_RVC_CONNECT_TIMEOUT_SECONDS)
    assert rvc_sidecar.convert_timeout_seconds == pytest.approx(DEFAULT_RVC_CONVERT_TIMEOUT_SECONDS)


def test_settings_load_env_overrides(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    temp_wav_dir = tmp_path / "wav"
    monkeypatch.setenv("IRODORI_TTS_CLIENT_HOST", "100.112.161.83")
    monkeypatch.setenv("IRODORI_TTS_CLIENT_PORT", str(OVERRIDE_PORT))
    monkeypatch.setenv("IRODORI_TTS_RUNTIME_NUM_STEPS", str(OVERRIDE_NUM_STEPS))
    monkeypatch.setenv("IRODORI_TTS_RUNTIME_CFG_SCALE_CAPTION", str(OVERRIDE_CFG_SCALE_CAPTION))
    monkeypatch.setenv("IRODORI_TTS_RUNTIME_COMPILE_MODEL", "true")
    monkeypatch.setenv("IRODORI_TTS_PATH_TEMP_WAV_DIR", str(temp_wav_dir))
    monkeypatch.setenv("IRODORI_RVC_SIDECAR_URL", f"http://127.0.0.1:{OVERRIDE_RVC_PORT}")
    monkeypatch.setenv("IRODORI_RVC_SIDECAR_API_NAME", "/custom_convert")
    monkeypatch.setenv(
        "IRODORI_RVC_SIDECAR_CONNECT_TIMEOUT_SECONDS",
        str(OVERRIDE_RVC_CONNECT_TIMEOUT_SECONDS),
    )
    monkeypatch.setenv(
        "IRODORI_RVC_SIDECAR_CONVERT_TIMEOUT_SECONDS",
        str(OVERRIDE_RVC_CONVERT_TIMEOUT_SECONDS),
    )

    assert ClientSettings().host == "100.112.161.83"
    assert ClientSettings().port == OVERRIDE_PORT
    assert IrodoriRuntimeSettings().num_steps == OVERRIDE_NUM_STEPS
    assert IrodoriRuntimeSettings().cfg_scale_caption == pytest.approx(OVERRIDE_CFG_SCALE_CAPTION)
    assert IrodoriRuntimeSettings().compile_model is True
    assert PathSettings().temp_wav_dir == temp_wav_dir
    assert RVCSidecarSettings().url.host == "127.0.0.1"
    assert RVCSidecarSettings().url.port == OVERRIDE_RVC_PORT
    assert RVCSidecarSettings().api_name == "/custom_convert"
    assert RVCSidecarSettings().connect_timeout_seconds == pytest.approx(
        OVERRIDE_RVC_CONNECT_TIMEOUT_SECONDS
    )
    assert RVCSidecarSettings().convert_timeout_seconds == pytest.approx(
        OVERRIDE_RVC_CONVERT_TIMEOUT_SECONDS
    )


def test_valid_port_boundary_values_are_accepted() -> None:
    min_port = 1
    max_port = 65_535
    assert ServerSettings(port=min_port).port == min_port
    assert ServerSettings(port=max_port).port == max_port
    assert ClientSettings(port=min_port).port == min_port
    assert ClientSettings(port=max_port).port == max_port


def test_invalid_port_values_are_rejected() -> None:
    with pytest.raises(ValidationError, match="less than or equal"):
        ServerSettings(port=65_536)

    with pytest.raises(ValidationError, match="greater than or equal"):
        ClientSettings(port=0)


def test_blank_path_values_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValidationError, match="temp_wav_dir"):
        PathSettings.model_validate({"temp_wav_dir": ""})

    monkeypatch.setenv("IRODORI_TTS_PATH_TEMP_WAV_DIR", "   ")
    with pytest.raises(ValidationError, match="temp_wav_dir"):
        PathSettings()


def test_rvc_sidecar_invalid_url_rejected() -> None:
    with pytest.raises(ValidationError, match="url"):
        RVCSidecarSettings.model_validate({"url": "not-a-valid-url"})


@pytest.mark.parametrize("api_name", ["", "infer_convert"])
def test_rvc_sidecar_api_name_must_start_with_slash(api_name: str) -> None:
    with pytest.raises(ValidationError, match="api_name"):
        RVCSidecarSettings.model_validate({"api_name": api_name})


@pytest.mark.parametrize("timeout_field", ["connect_timeout_seconds", "convert_timeout_seconds"])
def test_rvc_sidecar_zero_timeout_rejected(timeout_field: str) -> None:
    with pytest.raises(ValidationError, match="greater than"):
        RVCSidecarSettings.model_validate({timeout_field: 0})


def test_config_import_does_not_import_heavy_layers() -> None:
    code = """
import sys

import irodori_tts_infra.config

forbidden_prefixes = (
    "irodori_tts_infra.server",
    "irodori_tts_infra.engine",
    "irodori_tts_infra.client.sync",
    "irodori_tts_infra.client.async_",
    "irodori_tts_infra.cache",
)
raise SystemExit(
    any(name.startswith(forbidden_prefixes) for name in sys.modules)
)
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"

    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code],
        capture_output=True,
        env=env,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.integration
@pytest.mark.skipif(
    shutil.which("irodori-tts") is None,
    reason="console script not installed in this environment",
)
def test_console_script_help_exits_successfully() -> None:
    script = shutil.which("irodori-tts")
    assert script is not None

    result = subprocess.run(  # noqa: S603
        [script, "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Usage" in result.stdout
