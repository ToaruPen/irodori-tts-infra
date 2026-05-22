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
    ServerSettings,
)

if TYPE_CHECKING:
    from pathlib import Path

DEFAULT_PORT = 8923
DEFAULT_NUM_STEPS = 40
DEFAULT_CFG_SCALE_TEXT = 3.0
DEFAULT_CFG_SCALE_SPEAKER = 5.0
OVERRIDE_PORT = 9001
OVERRIDE_NUM_STEPS = 24
OVERRIDE_CFG_SCALE_SPEAKER = 4.0
OVERRIDE_DURATION_SCALE = 0.85
OVERRIDE_NUM_CANDIDATES = 2
OVERRIDE_SWAY_COEFF = -0.8


def test_settings_defaults_match_phase1_runtime_plan() -> None:
    client = ClientSettings()
    server = ServerSettings()
    runtime = IrodoriRuntimeSettings()
    paths = PathSettings()

    assert client.host == "127.0.0.1"
    assert client.port == DEFAULT_PORT
    assert server.host == "0.0.0.0"  # noqa: S104
    assert server.port == DEFAULT_PORT
    assert runtime.checkpoint == "Aratako/Irodori-TTS-500M-v3"
    assert runtime.num_steps == DEFAULT_NUM_STEPS
    assert runtime.cfg_scale_text == pytest.approx(DEFAULT_CFG_SCALE_TEXT)
    assert runtime.cfg_scale_speaker == pytest.approx(DEFAULT_CFG_SCALE_SPEAKER)
    assert runtime.seed is None
    assert runtime.duration_scale == pytest.approx(1.0)
    assert runtime.num_candidates == 1
    assert runtime.t_schedule_mode == "linear"
    assert runtime.sway_coeff == pytest.approx(-1.0)
    assert runtime.model_device == "cuda"
    assert runtime.model_precision == "bf16"
    assert runtime.codec_device == "cuda"
    assert runtime.codec_precision == "fp32"
    assert runtime.decode_mode == "sequential"
    assert runtime.context_kv_cache is True
    assert runtime.compile_model is False
    assert runtime.warmup_num_steps == DEFAULT_NUM_STEPS
    assert runtime.warmup_text == "テスト"
    assert paths.temp_wav_dir.name == "irodori-tts-wav"


def test_runtime_settings_rejects_blank_checkpoint_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("IRODORI_TTS_RUNTIME_CHECKPOINT", "   ")

    with pytest.raises(ValidationError, match="checkpoint"):
        IrodoriRuntimeSettings()


def test_settings_load_env_overrides(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    temp_wav_dir = tmp_path / "wav"
    monkeypatch.setenv("IRODORI_TTS_CLIENT_HOST", "100.112.161.83")
    monkeypatch.setenv("IRODORI_TTS_CLIENT_PORT", str(OVERRIDE_PORT))
    monkeypatch.setenv("IRODORI_TTS_RUNTIME_NUM_STEPS", str(OVERRIDE_NUM_STEPS))
    monkeypatch.setenv("IRODORI_TTS_RUNTIME_CFG_SCALE_SPEAKER", str(OVERRIDE_CFG_SCALE_SPEAKER))
    monkeypatch.setenv("IRODORI_TTS_RUNTIME_DURATION_SCALE", str(OVERRIDE_DURATION_SCALE))
    monkeypatch.setenv("IRODORI_TTS_RUNTIME_NUM_CANDIDATES", str(OVERRIDE_NUM_CANDIDATES))
    monkeypatch.setenv("IRODORI_TTS_RUNTIME_T_SCHEDULE_MODE", "sway")
    monkeypatch.setenv("IRODORI_TTS_RUNTIME_SWAY_COEFF", str(OVERRIDE_SWAY_COEFF))
    monkeypatch.setenv("IRODORI_TTS_RUNTIME_COMPILE_MODEL", "true")
    monkeypatch.setenv("IRODORI_TTS_PATH_TEMP_WAV_DIR", str(temp_wav_dir))

    assert ClientSettings().host == "100.112.161.83"
    assert ClientSettings().port == OVERRIDE_PORT
    assert IrodoriRuntimeSettings().num_steps == OVERRIDE_NUM_STEPS
    assert IrodoriRuntimeSettings().cfg_scale_speaker == pytest.approx(OVERRIDE_CFG_SCALE_SPEAKER)
    assert IrodoriRuntimeSettings().duration_scale == pytest.approx(OVERRIDE_DURATION_SCALE)
    assert IrodoriRuntimeSettings().num_candidates == OVERRIDE_NUM_CANDIDATES
    assert IrodoriRuntimeSettings().t_schedule_mode == "sway"
    assert IrodoriRuntimeSettings().sway_coeff == pytest.approx(OVERRIDE_SWAY_COEFF)
    assert IrodoriRuntimeSettings().compile_model is True
    assert PathSettings().temp_wav_dir == temp_wav_dir


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
