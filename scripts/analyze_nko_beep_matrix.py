from __future__ import annotations

import argparse
import csv
import json
import subprocess  # noqa: S404
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

PCM16_SAMPLE_WIDTH = 2
PCM16_SCALE = 32_768.0
POWER_FLOOR = np.finfo(np.float64).tiny


@dataclass(frozen=True, slots=True)
class ToneConfig:
    window_size: int = 2048
    hop_size: int = 480
    analysis_min_frequency_hz: float = 80.0
    min_tone_frequency_hz: float = 500.0
    max_frequency_hz: float = 5000.0
    peak_half_width_hz: float = 40.0
    min_peak_energy_ratio: float = 0.95
    max_normalized_entropy: float = 0.20
    max_frequency_std_hz: float = 80.0
    min_rms_dbfs: float = -45.0
    min_qualifying_frames: int = 8
    max_gap_frames: int = 3
    min_interval_span_frames: int = 10


@dataclass(frozen=True, slots=True)
class ToneInterval:
    start_seconds: float
    end_seconds: float
    frequency_hz: float
    frequency_std_hz: float
    peak_energy_ratio: float
    normalized_entropy: float
    rms_dbfs: float


@dataclass(frozen=True, slots=True)
class SpectralFeatures:
    total_power: np.ndarray
    peak_frequencies: np.ndarray
    peak_energy_ratio: np.ndarray
    normalized_entropy: np.ndarray


DEFAULT_CONFIG = ToneConfig()


def read_pcm16_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as reader:
        if reader.getcomptype() != "NONE":
            message = f"compressed WAV is unsupported: {path}"
            raise ValueError(message)
        if reader.getsampwidth() != PCM16_SAMPLE_WIDTH:
            message = f"WAV must use PCM16 samples: {path}"
            raise ValueError(message)
        channels = reader.getnchannels()
        sample_rate = reader.getframerate()
        frames = reader.readframes(reader.getnframes())

    pcm = np.frombuffer(frames, dtype="<i2").astype(np.float64)
    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=1)
    return pcm / PCM16_SCALE, sample_rate


def detect_narrowband_intervals(
    samples: np.ndarray,
    sample_rate: int,
    config: ToneConfig = DEFAULT_CONFIG,
) -> tuple[ToneInterval, ...]:
    normalized = np.asarray(samples, dtype=np.float64)
    if normalized.ndim != 1:
        message = "samples must be one-dimensional"
        raise ValueError(message)
    if sample_rate <= 0:
        message = "sample_rate must be positive"
        raise ValueError(message)
    if normalized.size < config.window_size:
        return ()

    frames = np.lib.stride_tricks.sliding_window_view(
        normalized,
        config.window_size,
    )[:: config.hop_size]
    spectral = _spectral_features(frames, sample_rate=sample_rate, config=config)
    rms = np.sqrt(np.mean(frames**2, axis=1))
    rms_dbfs = 20.0 * np.log10(np.maximum(rms, POWER_FLOOR))

    qualifying = (
        (spectral.total_power > POWER_FLOOR)
        & (spectral.peak_frequencies >= config.min_tone_frequency_hz)
        & (spectral.peak_energy_ratio >= config.min_peak_energy_ratio)
        & (spectral.normalized_entropy <= config.max_normalized_entropy)
        & (rms_dbfs >= config.min_rms_dbfs)
    )
    return _tone_intervals(
        qualifying,
        peak_frequencies=spectral.peak_frequencies,
        peak_energy_ratio=spectral.peak_energy_ratio,
        normalized_entropy=spectral.normalized_entropy,
        rms_dbfs=rms_dbfs,
        sample_rate=sample_rate,
        config=config,
    )


def _spectral_features(
    frames: np.ndarray,
    *,
    sample_rate: int,
    config: ToneConfig,
) -> SpectralFeatures:
    power = np.abs(np.fft.rfft(frames * np.hanning(config.window_size), axis=1)) ** 2
    frequencies = np.fft.rfftfreq(config.window_size, d=1.0 / sample_rate)
    band_mask = (frequencies >= config.analysis_min_frequency_hz) & (
        frequencies <= config.max_frequency_hz
    )
    band_frequencies = frequencies[band_mask]
    band_power = power[:, band_mask]
    total_power = band_power.sum(axis=1)
    safe_total_power = np.maximum(total_power, POWER_FLOOR)

    peak_indices = np.argmax(band_power, axis=1)
    peak_frequencies = band_frequencies[peak_indices]
    peak_neighborhood = (
        np.abs(band_frequencies[np.newaxis, :] - peak_frequencies[:, np.newaxis])
        <= config.peak_half_width_hz
    )
    peak_energy_ratio = np.where(
        total_power > 0.0,
        np.sum(band_power * peak_neighborhood, axis=1) / safe_total_power,
        0.0,
    )

    probabilities = band_power / safe_total_power[:, np.newaxis]
    log_probabilities = np.zeros_like(probabilities)
    np.log(
        probabilities,
        out=log_probabilities,
        where=probabilities > 0.0,
    )
    normalized_entropy = -np.sum(
        probabilities * log_probabilities,
        axis=1,
    ) / np.log(band_power.shape[1])
    return SpectralFeatures(
        total_power=total_power,
        peak_frequencies=peak_frequencies,
        peak_energy_ratio=peak_energy_ratio,
        normalized_entropy=normalized_entropy,
    )


def _tone_intervals(
    qualifying: np.ndarray,
    *,
    peak_frequencies: np.ndarray,
    peak_energy_ratio: np.ndarray,
    normalized_entropy: np.ndarray,
    rms_dbfs: np.ndarray,
    sample_rate: int,
    config: ToneConfig,
) -> tuple[ToneInterval, ...]:
    frame_indices = np.flatnonzero(qualifying)
    if frame_indices.size == 0:
        return ()
    groups = np.split(
        frame_indices,
        np.flatnonzero(np.diff(frame_indices) > config.max_gap_frames + 1) + 1,
    )
    intervals: list[ToneInterval] = []
    for group in groups:
        if group.size < config.min_qualifying_frames:
            continue
        if group[-1] - group[0] < config.min_interval_span_frames:
            continue
        group_frequencies = peak_frequencies[group]
        frequency_std = float(np.std(group_frequencies))
        if frequency_std > config.max_frequency_std_hz:
            continue
        intervals.append(
            ToneInterval(
                start_seconds=round(float(group[0] * config.hop_size / sample_rate), 3),
                end_seconds=round(
                    float(
                        (group[-1] * config.hop_size + config.window_size) / sample_rate,
                    ),
                    3,
                ),
                frequency_hz=round(float(np.median(group_frequencies)), 3),
                frequency_std_hz=round(frequency_std, 3),
                peak_energy_ratio=round(float(np.min(peak_energy_ratio[group])), 6),
                normalized_entropy=round(float(np.max(normalized_entropy[group])), 6),
                rms_dbfs=round(float(np.min(rms_dbfs[group])), 3),
            ),
        )
    return tuple(intervals)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    generation_rows = _read_jsonl(args.generation_dir / "generation-results.jsonl")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    spectrogram_dir = args.output_dir / "spectrograms"
    spectrogram_dir.mkdir(parents=True, exist_ok=True)

    analysis_rows = [
        _analyze_row(
            row,
            generation_dir=args.generation_dir,
            spectrogram_dir=spectrogram_dir,
            ffmpeg=args.ffmpeg,
        )
        for row in generation_rows
    ]
    _write_jsonl(args.output_dir / "analysis-results.jsonl", analysis_rows)
    _write_summary_csv(args.output_dir / "summary.csv", analysis_rows)
    _write_summary_markdown(args.output_dir / "summary.md", analysis_rows)

    candidate_count = sum(row["analysis_status"] == "CANDIDATE" for row in analysis_rows)
    error_count = sum(row["analysis_status"] == "ERROR" for row in analysis_rows)
    print(
        f"analysis complete: {len(analysis_rows)} rows, "
        f"{candidate_count} CANDIDATE, {error_count} ERROR",
    )
    return 1 if error_count else 0


def _analyze_row(
    generation_row: dict[str, object],
    *,
    generation_dir: Path,
    spectrogram_dir: Path,
    ffmpeg: str,
) -> dict[str, object]:
    row = dict(generation_row)
    row["detector_config"] = asdict(DEFAULT_CONFIG)
    row["intervals"] = []
    row["spectrogram_path"] = None
    row["spectrogram_error"] = None

    if generation_row.get("status") != "SUCCESS":
        row["analysis_status"] = "ERROR"
        return row

    case_id = str(generation_row["case_id"])
    wav_path = _resolve_wav_path(generation_row, generation_dir=generation_dir)
    try:
        samples, sample_rate = read_pcm16_wav(wav_path)
        intervals = detect_narrowband_intervals(samples, sample_rate)
    except (OSError, ValueError, wave.Error) as exc:
        row["analysis_status"] = "ERROR"
        row["analysis_exception_type"] = type(exc).__name__
        row["analysis_exception_message"] = str(exc)
        return row

    row["sample_rate"] = sample_rate
    row["intervals"] = [asdict(interval) for interval in intervals]
    row["analysis_status"] = "CANDIDATE" if intervals else "CLEAR"
    if intervals:
        spectrogram_path = spectrogram_dir / f"{case_id}.png"
        row["spectrogram_path"] = str(spectrogram_path)
        try:
            _render_spectrogram(wav_path, spectrogram_path, ffmpeg=ffmpeg)
        except (OSError, subprocess.CalledProcessError) as exc:
            row["spectrogram_error"] = f"{type(exc).__name__}: {exc}"
    return row


def _resolve_wav_path(generation_row: dict[str, object], *, generation_dir: Path) -> Path:
    configured = generation_row.get("wav_path")
    if configured is not None:
        configured_path = Path(str(configured))
        if configured_path.is_file():
            return configured_path
    return generation_dir / "wav" / f"{generation_row['case_id']}.wav"


def _render_spectrogram(wav_path: Path, output_path: Path, *, ffmpeg: str) -> None:
    subprocess.run(  # noqa: S603
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(wav_path),
            "-lavfi",
            "showspectrumpic=s=1600x900:legend=1",
            str(output_path),
        ],
        check=True,
    )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _write_jsonl(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_summary_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    fieldnames = (
        "case_id",
        "speaker_filename",
        "text_id",
        "text",
        "control",
        "seed",
        "style",
        "analysis_status",
        "interval_count",
        "first_start_seconds",
        "first_end_seconds",
        "first_frequency_hz",
        "wav_path",
        "spectrogram_path",
        "exception_type",
        "exception_message",
    )
    with path.open("w", encoding="utf-8", newline="") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            intervals = row.get("intervals")
            interval_rows = intervals if isinstance(intervals, list) else []
            first = interval_rows[0] if interval_rows else {}
            writer.writerow(
                {
                    "case_id": row.get("case_id"),
                    "speaker_filename": row.get("speaker_filename"),
                    "text_id": row.get("text_id"),
                    "text": row.get("text"),
                    "control": row.get("control"),
                    "seed": row.get("seed"),
                    "style": row.get("style"),
                    "analysis_status": row.get("analysis_status"),
                    "interval_count": len(interval_rows),
                    "first_start_seconds": first.get("start_seconds"),
                    "first_end_seconds": first.get("end_seconds"),
                    "first_frequency_hz": first.get("frequency_hz"),
                    "wav_path": row.get("wav_path"),
                    "spectrogram_path": row.get("spectrogram_path"),
                    "exception_type": row.get("exception_type"),
                    "exception_message": row.get("exception_message"),
                },
            )


def _write_summary_markdown(path: Path, rows: Sequence[dict[str, object]]) -> None:
    counts = {
        status: sum(row.get("analysis_status") == status for row in rows)
        for status in ("CLEAR", "CANDIDATE", "ERROR")
    }
    candidates = [row for row in rows if row.get("analysis_status") == "CANDIDATE"]
    lines = [
        "# Narrowband beep screening",
        "",
        f"- Total: {len(rows)}",
        f"- CLEAR: {counts['CLEAR']}",
        f"- CANDIDATE: {counts['CANDIDATE']}",
        f"- ERROR: {counts['ERROR']}",
        "",
        "## Detector configuration",
        "",
        "```json",
        json.dumps(asdict(DEFAULT_CONFIG), indent=2, sort_keys=True),
        "```",
        "",
        "## Candidates",
        "",
    ]
    if not candidates:
        lines.append("No candidates.")
    else:
        lines.extend(
            f"- `{row['case_id']}`: `{row['wav_path']}` / `{row['spectrogram_path']}`"
            for row in candidates
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
