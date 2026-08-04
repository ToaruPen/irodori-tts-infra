from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import sys
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

_ANALYZER_PATH = Path(__file__).with_name("analyze_nko_beep_matrix.py")
_ANALYZER_SPEC = importlib.util.spec_from_file_location(
    "_audit_training_tones_analyzer",
    _ANALYZER_PATH,
)
if _ANALYZER_SPEC is None or _ANALYZER_SPEC.loader is None:
    message = f"cannot load tone analyzer: {_ANALYZER_PATH}"
    raise RuntimeError(message)
_ANALYZER = importlib.util.module_from_spec(_ANALYZER_SPEC)
sys.modules[_ANALYZER_SPEC.name] = _ANALYZER
_ANALYZER_SPEC.loader.exec_module(_ANALYZER)

CENSOR_MARKERS = ("◯", "○", "〇")  # noqa: RUF001 - source captions use this glyph
PCM16_SCALE = 32_767.0
BROAD_TONE_CONFIG = _ANALYZER.ToneConfig(
    analysis_min_frequency_hz=40.0,
    min_tone_frequency_hz=80.0,
    max_frequency_hz=20_000.0,
)


@dataclass(frozen=True, slots=True)
class SourceRecord:
    audio_path: Path
    text: str
    speaker: str
    caption_has_censor: bool


def load_source_records(index_path: Path, *, speaker: str) -> tuple[SourceRecord, ...]:
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        message = f"source index must contain a list: {index_path}"
        raise TypeError(message)

    records: list[SourceRecord] = []
    for item in payload:
        if not isinstance(item, dict) or str(item.get("Speaker") or "") != speaker:
            continue
        text = str(item.get("Text") or "").strip()
        source_file = str(item.get("FilePath") or "")
        if not text or not source_file:
            continue
        records.append(
            SourceRecord(
                audio_path=index_path.parent / Path(source_file.replace("\\", "/")),
                text=text,
                speaker=speaker,
                caption_has_censor=any(marker in text for marker in CENSOR_MARKERS),
            ),
        )
    return tuple(records)


def analyze_training_record(
    record: SourceRecord,
    *,
    samples: np.ndarray,
    sample_rate: int,
) -> dict[str, object]:
    intervals = _ANALYZER.detect_narrowband_intervals(
        samples,
        sample_rate,
        BROAD_TONE_CONFIG,
    )
    return {
        "audio_path": str(record.audio_path),
        "text": record.text,
        "speaker": record.speaker,
        "caption_has_censor": record.caption_has_censor,
        "sample_rate": sample_rate,
        "analysis_status": "CANDIDATE" if intervals else "CLEAR",
        "intervals": [asdict(interval) for interval in intervals],
        "detector_config": asdict(BROAD_TONE_CONFIG),
    }


def decode_audio(path: Path) -> tuple[np.ndarray, int]:
    soundfile = importlib.import_module("soundfile")
    samples, sample_rate = soundfile.read(
        str(path),
        dtype="float64",
        always_2d=True,
    )
    mono = np.asarray(samples, dtype=np.float64).mean(axis=1)
    if mono.size == 0:
        message = f"decoded audio is empty: {path}"
        raise ValueError(message)
    return mono, int(sample_rate)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    records = load_source_records(args.index_json, speaker=args.speaker)
    if not records:
        message = f"no records found for speaker {args.speaker!r}"
        raise ValueError(message)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidate_dir = args.output_dir / "candidate-audio"
    candidate_dir.mkdir(exist_ok=True)
    results_path = args.output_dir / "training-tone-results.jsonl"

    rows: list[dict[str, object]] = []
    with results_path.open("w", encoding="utf-8", newline="\n") as results_file:
        for index, record in enumerate(records, start=1):
            row = _audit_record(record)
            if row["analysis_status"] == "CANDIDATE":
                samples, sample_rate = decode_audio(record.audio_path)
                candidate_path = candidate_dir / f"{index:05d}_{record.audio_path.stem}.wav"
                _write_pcm16_wav(candidate_path, samples=samples, sample_rate=sample_rate)
                row["candidate_wav_path"] = str(candidate_path)
            rows.append(row)
            results_file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            results_file.flush()
            if index % args.progress_every == 0 or index == len(records):
                print(f"[{index}/{len(records)}] {record.speaker}")

    _write_summary(args.output_dir / "summary.json", rows)
    error_count = sum(row["analysis_status"] == "ERROR" for row in rows)
    return 1 if error_count else 0


def _audit_record(record: SourceRecord) -> dict[str, object]:
    try:
        samples, sample_rate = decode_audio(record.audio_path)
        return analyze_training_record(
            record,
            samples=samples,
            sample_rate=sample_rate,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        return {
            "audio_path": str(record.audio_path),
            "text": record.text,
            "speaker": record.speaker,
            "caption_has_censor": record.caption_has_censor,
            "analysis_status": "ERROR",
            "intervals": [],
            "detector_config": asdict(BROAD_TONE_CONFIG),
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
        }


def _write_pcm16_wav(path: Path, *, samples: np.ndarray, sample_rate: int) -> None:
    pcm = np.round(np.clip(samples, -1.0, 1.0) * PCM16_SCALE).astype("<i2")
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(pcm.tobytes())


def _write_summary(path: Path, rows: Sequence[dict[str, object]]) -> None:
    candidates = [row for row in rows if row["analysis_status"] == "CANDIDATE"]
    payload = {
        "total": len(rows),
        "clear": sum(row["analysis_status"] == "CLEAR" for row in rows),
        "candidate": len(candidates),
        "error": sum(row["analysis_status"] == "ERROR" for row in rows),
        "caption_with_censor": sum(bool(row["caption_has_censor"]) for row in rows),
        "caption_without_censor": sum(not bool(row["caption_has_censor"]) for row in rows),
        "candidates_with_censor_caption": sum(
            bool(row["caption_has_censor"]) for row in candidates
        ),
        "candidates_without_censor_caption": sum(
            not bool(row["caption_has_censor"]) for row in candidates
        ),
        "detector_config": asdict(BROAD_TONE_CONFIG),
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-json", type=Path, required=True)
    parser.add_argument("--speaker", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
