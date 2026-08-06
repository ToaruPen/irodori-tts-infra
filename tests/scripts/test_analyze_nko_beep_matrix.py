from __future__ import annotations

import importlib.util
import sys
import wave
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from types import ModuleType

pytestmark = pytest.mark.unit

SCRIPT_PATH = Path("scripts/analyze_nko_beep_matrix.py")
SAMPLE_RATE = 48_000
BEEP_FREQUENCY_HZ = 1_000.0
BEEP_START_SECONDS = 0.2
BEEP_END_SECONDS = 0.5
MIN_EXPECTED_BEEP_SECONDS = 0.15
PCM16_SCALE = 32_768.0


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("analyze_nko_beep_matrix", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_detect_narrowband_intervals_ignores_silence() -> None:
    module = _load_script()
    silence = np.zeros(SAMPLE_RATE, dtype=np.float64)

    intervals = module.detect_narrowband_intervals(silence, SAMPLE_RATE)

    assert intervals == ()


def test_detect_narrowband_intervals_finds_embedded_one_kilohertz_beep() -> None:
    module = _load_script()
    samples = np.zeros(SAMPLE_RATE, dtype=np.float64)
    start = round(BEEP_START_SECONDS * SAMPLE_RATE)
    end = round(BEEP_END_SECONDS * SAMPLE_RATE)
    t = np.arange(end - start, dtype=np.float64) / SAMPLE_RATE
    samples[start:end] = np.sin(2 * np.pi * BEEP_FREQUENCY_HZ * t) * 0.5

    intervals = module.detect_narrowband_intervals(samples, SAMPLE_RATE)

    assert len(intervals) == 1
    interval = intervals[0]
    assert interval.end_seconds - interval.start_seconds >= MIN_EXPECTED_BEEP_SECONDS
    assert interval.frequency_hz == pytest.approx(BEEP_FREQUENCY_HZ, abs=25.0)


def test_detect_narrowband_intervals_rejects_harmonic_voiced_signal() -> None:
    module = _load_script()
    t = np.arange(SAMPLE_RATE, dtype=np.float64) / SAMPLE_RATE
    voiced = sum(
        (0.15 / harmonic) * np.sin(2 * np.pi * 180 * harmonic * t) for harmonic in range(1, 7)
    )

    intervals = module.detect_narrowband_intervals(voiced, SAMPLE_RATE)

    assert intervals == ()


def test_detect_narrowband_intervals_rejects_voice_range_pure_tone() -> None:
    module = _load_script()
    t = np.arange(SAMPLE_RATE, dtype=np.float64) / SAMPLE_RATE
    voice_range_tone = 0.5 * np.sin(2 * np.pi * 350 * t)

    intervals = module.detect_narrowband_intervals(voice_range_tone, SAMPLE_RATE)

    assert intervals == ()


def test_detect_narrowband_intervals_rejects_voiced_signal_with_dominant_high_harmonic() -> None:
    module = _load_script()
    t = np.arange(SAMPLE_RATE, dtype=np.float64) / SAMPLE_RATE
    voiced = (
        0.25 * np.sin(2 * np.pi * 220 * t)
        + 0.12 * np.sin(2 * np.pi * 440 * t)
        + 0.10 * np.sin(2 * np.pi * 660 * t)
        + 0.45 * np.sin(2 * np.pi * 880 * t)
    )

    intervals = module.detect_narrowband_intervals(voiced, SAMPLE_RATE)

    assert intervals == ()


def test_detect_narrowband_intervals_finds_fading_stepped_beep() -> None:
    module = _load_script()
    samples = np.zeros(SAMPLE_RATE, dtype=np.float64)
    first_start = round(0.2 * SAMPLE_RATE)
    first_length = round(0.08 * SAMPLE_RATE)
    transition_gap_length = 0
    second_length = round(0.08 * SAMPLE_RATE)
    first_t = np.arange(first_length, dtype=np.float64) / SAMPLE_RATE
    second_t = np.arange(second_length, dtype=np.float64) / SAMPLE_RATE
    samples[first_start : first_start + first_length] = 0.5 * np.sin(2 * np.pi * 700 * first_t)
    second_start = first_start + first_length + transition_gap_length
    samples[second_start : second_start + second_length] = np.linspace(
        0.2, 0.03, second_length
    ) * np.sin(2 * np.pi * 650 * second_t)

    intervals = module.detect_narrowband_intervals(samples, SAMPLE_RATE)

    assert len(intervals) == 1
    assert intervals[0].frequency_hz == pytest.approx(675.0, abs=30.0)


def test_read_pcm16_wav_decodes_mono(tmp_path: Path) -> None:
    module = _load_script()
    wav_path = tmp_path / "mono.wav"
    pcm = np.array([-32768, 0, 32767], dtype="<i2")
    _write_pcm16_wav(wav_path, pcm=pcm, channels=1)

    samples, sample_rate = module.read_pcm16_wav(wav_path)

    assert sample_rate == SAMPLE_RATE
    np.testing.assert_allclose(samples, pcm.astype(np.float64) / PCM16_SCALE)


def test_read_pcm16_wav_averages_stereo_channels(tmp_path: Path) -> None:
    module = _load_script()
    wav_path = tmp_path / "stereo.wav"
    pcm = np.array(
        [
            [16384, -16384],
            [8192, 8192],
        ],
        dtype="<i2",
    )
    _write_pcm16_wav(wav_path, pcm=pcm.reshape(-1), channels=2)

    samples, sample_rate = module.read_pcm16_wav(wav_path)

    assert sample_rate == SAMPLE_RATE
    np.testing.assert_allclose(samples, np.array([0.0, 0.25]))


def _write_pcm16_wav(path: Path, *, pcm: np.ndarray, channels: int) -> None:
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(2)
        writer.setframerate(SAMPLE_RATE)
        writer.writeframes(pcm.tobytes())
