"""Censor-beep detection for synthesized audio.

The v4 base model occasionally renders explicit words as a machine censor beep:
one stable spectral line near 710 Hz or 1 kHz that carries almost all frame
energy. Natural voice spreads energy over harmonics and drifts in pitch, so a
run of frames dominated by one steady line inside those bands is a beep.
"""

from __future__ import annotations

import io
import wave
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

# Window, hop and run length are sample counts validated at 44.1 and 48 kHz only.
VALIDATED_SAMPLE_RATES_HZ = (40_000, 50_000)
WINDOW_SAMPLES = 1024
HOP_SAMPLES = 96
BLOCK_FRAMES = 4000
SEARCH_BAND_HZ = (150.0, 8000.0)
MIN_FRAME_LINE_RATIO = 0.85
MIN_FRAME_RMS_DBFS = -50.0
MAX_FRAME_STEP_HZ = 2.0
MIN_PARABOLA_CURVATURE = 1e-12
# Generated beeps measured 708.1-714.4 Hz and 998.6-1002.1 Hz (83 runs); with
# these limits 13 of 30,258 real voice clips and 751 of 751 real beeps match.
BEEP_BANDS_HZ = ((705.0, 716.0), (990.0, 1010.0))
MIN_RUN_FRAMES = 15
MIN_RUN_LINE_RATIO = 0.95
MAX_RUN_FREQUENCY_STD_HZ = 1.5
PCM16_SAMPLE_WIDTH = 2
PCM16_FULL_SCALE = 32768.0
TINY = float(np.finfo(np.float64).tiny)


@dataclass(frozen=True, slots=True)
class BeepRun:
    start_seconds: float
    end_seconds: float
    frequency_hz: float


def find_censor_beeps_in_wav(wav_bytes: bytes) -> tuple[BeepRun, ...]:
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as source:
            sample_width = source.getsampwidth()
            channels = source.getnchannels()
            sample_rate = source.getframerate()
            frames = source.readframes(source.getnframes())
    except (wave.Error, EOFError) as exc:
        msg = "audio is not a readable WAV file"
        raise ValueError(msg) from exc
    if sample_width != PCM16_SAMPLE_WIDTH:
        msg = f"audio must be 16-bit PCM WAV, got sample width {sample_width}"
        raise ValueError(msg)
    pcm = np.frombuffer(frames, dtype="<i2").reshape(-1, channels)
    return find_censor_beeps(pcm.mean(axis=1) / PCM16_FULL_SCALE, sample_rate)


def find_censor_beeps(samples: ArrayLike, sample_rate: int) -> tuple[BeepRun, ...]:
    if not VALIDATED_SAMPLE_RATES_HZ[0] <= sample_rate <= VALIDATED_SAMPLE_RATES_HZ[1]:
        msg = f"beep detection is not validated for sample rate {sample_rate}"
        raise ValueError(msg)
    mono = np.asarray(samples, dtype=np.float64)
    if mono.size < WINDOW_SAMPLES:
        return ()
    frequency, ratio, loud = _frame_lines(mono, sample_rate)
    dominated = (ratio >= MIN_FRAME_LINE_RATIO) & loud
    steady = np.r_[True, np.abs(np.diff(frequency)) <= MAX_FRAME_STEP_HZ]
    beeps: list[BeepRun] = []
    start: int | None = None
    for index in range(dominated.size + 1):
        inside = index < dominated.size and bool(dominated[index])
        if inside and start is not None and steady[index]:
            continue
        if start is not None and _is_beep(frequency[start:index], ratio[start:index]):
            beeps.append(
                BeepRun(
                    start_seconds=start * HOP_SAMPLES / sample_rate,
                    end_seconds=((index - 1) * HOP_SAMPLES + WINDOW_SAMPLES) / sample_rate,
                    frequency_hz=float(np.median(frequency[start:index])),
                )
            )
        start = index if inside else None
    return tuple(beeps)


def _frame_lines(
    mono: NDArray[np.float64], sample_rate: int
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    frames_all: NDArray[np.float64] = np.lib.stride_tricks.sliding_window_view(
        mono, WINDOW_SAMPLES
    )[::HOP_SAMPLES]
    window = np.hanning(WINDOW_SAMPLES + 1)[:-1]
    bin_hz = sample_rate / WINDOW_SAMPLES
    frequencies, ratios, louds = [], [], []
    for begin in range(0, frames_all.shape[0], BLOCK_FRAMES):
        frames = frames_all[begin : begin + BLOCK_FRAMES]
        power = np.abs(np.fft.rfft(frames * window, axis=1)) ** 2
        peak_bin, line = _dominant_lines(power, bin_hz)
        frequencies.append(peak_bin * bin_hz)
        ratios.append(line / (power.sum(axis=1) + TINY))
        rms_dbfs = 10.0 * np.log10(np.maximum(np.mean(frames**2, axis=1), TINY))
        louds.append(rms_dbfs >= MIN_FRAME_RMS_DBFS)
    return np.concatenate(frequencies), np.concatenate(ratios), np.concatenate(louds)


def _dominant_lines(
    power: NDArray[np.float64], bin_hz: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    # Per frame: interpolated peak bin, and the power of that peak with its two neighbours.
    low = int(np.ceil(SEARCH_BAND_HZ[0] / bin_hz))
    high = min(int(SEARCH_BAND_HZ[1] / bin_hz), power.shape[1] - 1)
    rows = np.arange(power.shape[0])
    peak = power[:, low:high].argmax(axis=1) + low
    neighbours = [power[rows, peak + k] for k in (-1, 0, 1)]
    left, mid, right = (np.log(value + TINY) for value in neighbours)
    curvature = left - 2 * mid + right
    offset = np.divide(
        0.5 * (left - right),
        curvature,
        out=np.zeros_like(curvature),
        where=np.abs(curvature) > MIN_PARABOLA_CURVATURE,
    )
    return peak + offset, np.sum(neighbours, axis=0)


def _is_beep(frequency: NDArray[np.float64], ratio: NDArray[np.float64]) -> bool:
    if frequency.size < MIN_RUN_FRAMES:
        return False
    center = float(np.median(frequency))
    if not any(low <= center <= high for low, high in BEEP_BANDS_HZ):
        return False
    if float(np.std(frequency)) > MAX_RUN_FREQUENCY_STD_HZ:
        return False
    return float(np.mean(ratio)) >= MIN_RUN_LINE_RATIO
