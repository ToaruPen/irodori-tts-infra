from __future__ import annotations

import io
import wave

import numpy as np
import pytest

from irodori_tts_infra.metrics.beep import find_censor_beeps, find_censor_beeps_in_wav

pytestmark = pytest.mark.unit

SAMPLE_RATE = 48_000
TONE_START_SECONDS = 1.0
TONE_SECONDS = 0.12
START_TOLERANCE_SECONDS = 0.03
FREQUENCY_TOLERANCE_HZ = 1.0


def _voice_like(seconds: float) -> np.ndarray:
    time = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    pitch = 220.0 + 25.0 * np.sin(2 * np.pi * 5.0 * time)
    phase = 2 * np.pi * np.cumsum(pitch) / SAMPLE_RATE
    harmonics = np.sum([np.sin(k * phase) / k for k in range(1, 9)], axis=0)
    return np.asarray(0.2 * harmonics)


def _with_tone(frequency_hz: float, *, seconds: float = TONE_SECONDS) -> np.ndarray:
    samples = _voice_like(2.0)
    start = int(TONE_START_SECONDS * SAMPLE_RATE)
    time = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    samples[start : start + time.size] = 0.3 * np.sin(2 * np.pi * frequency_hz * time)
    return samples


def _wav(frames: bytes, *, sample_width: int) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(sample_width)
        target.setframerate(SAMPLE_RATE)
        target.writeframes(frames)
    return buffer.getvalue()


def _pcm16_wav(samples: np.ndarray) -> bytes:
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
    return _wav(pcm.tobytes(), sample_width=2)


def test_voice_like_audio_has_no_beeps() -> None:
    assert find_censor_beeps(_voice_like(2.0), SAMPLE_RATE) == ()


@pytest.mark.parametrize("frequency_hz", [708.5, 712.5, 998.6, 1001.8])
def test_generated_beep_frequencies_are_detected(frequency_hz: float) -> None:
    beeps = find_censor_beeps(_with_tone(frequency_hz), SAMPLE_RATE)

    assert len(beeps) == 1
    assert beeps[0].frequency_hz == pytest.approx(frequency_hz, abs=FREQUENCY_TOLERANCE_HZ)
    assert beeps[0].start_seconds == pytest.approx(
        TONE_START_SECONDS,
        abs=START_TOLERANCE_SECONDS,
    )


def test_stable_tone_outside_beep_bands_is_ignored() -> None:
    assert find_censor_beeps(_with_tone(850.0), SAMPLE_RATE) == ()


def test_tone_shorter_than_minimum_run_is_ignored() -> None:
    assert find_censor_beeps(_with_tone(1000.0, seconds=0.02), SAMPLE_RATE) == ()


def test_in_band_tone_with_drifting_pitch_is_ignored() -> None:
    samples = _voice_like(2.0)
    start = int(TONE_START_SECONDS * SAMPLE_RATE)
    time = np.arange(int(0.3 * SAMPLE_RATE)) / SAMPLE_RATE
    pitch = 1000.0 + 8.0 * np.sin(2 * np.pi * 2.0 * time)
    samples[start : start + time.size] = 0.3 * np.sin(2 * np.pi * np.cumsum(pitch) / SAMPLE_RATE)

    assert find_censor_beeps(samples, SAMPLE_RATE) == ()


def test_in_band_tone_buried_in_noise_is_ignored() -> None:
    samples = _with_tone(1000.0)
    samples += np.random.default_rng(0).normal(0.0, 0.05, samples.size)

    assert find_censor_beeps(samples, SAMPLE_RATE) == ()


def test_audio_shorter_than_one_window_has_no_beeps() -> None:
    assert find_censor_beeps(np.zeros(512), SAMPLE_RATE) == ()


@pytest.mark.parametrize("sample_rate", [24_000, 96_000])
def test_sample_rates_outside_the_validated_range_are_rejected(sample_rate: int) -> None:
    with pytest.raises(ValueError, match="sample rate"):
        find_censor_beeps(_voice_like(0.5), sample_rate)


def test_wav_bytes_are_decoded_before_detection() -> None:
    assert len(find_censor_beeps_in_wav(_pcm16_wav(_with_tone(1000.6)))) == 1
    assert find_censor_beeps_in_wav(_pcm16_wav(_voice_like(2.0))) == ()


def test_non_pcm16_wav_is_rejected() -> None:
    with pytest.raises(ValueError, match="16-bit PCM"):
        find_censor_beeps_in_wav(_wav(bytes(3 * SAMPLE_RATE), sample_width=3))


def test_unreadable_wav_is_rejected() -> None:
    with pytest.raises(ValueError, match="readable WAV"):
        find_censor_beeps_in_wav(b"RIFF\x00\x00\x00\x00WAVEfake")
