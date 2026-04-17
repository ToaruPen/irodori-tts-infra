from __future__ import annotations

import re
import struct
import sys
import wave
from array import array
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

import structlog

from irodori_tts_infra.datasets.models import (
    MAX_SAMPLE_RATE,
    MIN_SAMPLE_RATE,
    ExtractedClip,
    ExtractionIndex,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

_LOGGER = structlog.get_logger(__name__)
_REPO_SEGMENT_PATTERN = r"[0-9A-Za-z_-]+"
_REPO_SEGMENT_RE = re.compile(_REPO_SEGMENT_PATTERN)
_REPO_PATH_RE = re.compile(
    rf"(?:^|/)data/(?P<character>{_REPO_SEGMENT_PATTERN})/wav/(?P<filename>[^/]+\.wav)$"
)

DEFAULT_DATASET_REPO = "litagin/moe-speech"
DEFAULT_MAX_BYTES = 1_073_741_824
DEFAULT_OUTPUT_SAMPLE_RATE = 24_000
PCM16_SAMPLE_WIDTH_BYTES = 2


class DatasetExtractionError(RuntimeError):
    """Raised when extraction cannot proceed."""


class NsfwSubsetUnavailableError(DatasetExtractionError):
    """Raised when the caller opts out of NSFW content for moe-speech."""


class UnsupportedAudioFormatError(DatasetExtractionError):
    """Raised when a source clip is not a mono 16-bit WAV."""


@dataclass(frozen=True, slots=True)
class MoeSpeechRecord:
    repo_path: str
    wav_bytes: bytes


def extract_character_dataset(
    *,
    character: str,
    out_dir: Path,
    max_bytes: int = DEFAULT_MAX_BYTES,
    sample_rate: int = DEFAULT_OUTPUT_SAMPLE_RATE,
    include_nsfw: bool = False,
    dataset_repo: str = DEFAULT_DATASET_REPO,
    hf_token: str | None = None,
    records: Iterable[MoeSpeechRecord] | None = None,
) -> ExtractionIndex:
    normalized_character = _normalize_character(character)
    _validate_sample_rate(sample_rate)
    _validate_max_bytes(max_bytes)
    _ensure_nsfw_allowed(include_nsfw=include_nsfw)
    _ensure_output_dir_available(out_dir)

    if records is None:
        ordered_records: Iterable[MoeSpeechRecord] = stream_character_records(
            normalized_character,
            dataset_repo=dataset_repo,
            hf_token=hf_token,
        )
    else:
        # Tests inject preloaded records, so this path sorts them eagerly
        # to keep fixtures deterministic.
        ordered_records = tuple(
            sorted(
                (
                    record
                    for record in records
                    if _character_from_repo_path(record.repo_path) == normalized_character
                ),
                key=lambda record: record.repo_path,
            )
        )

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{out_dir.name}.", dir=out_dir.parent) as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        extracted: list[ExtractedClip] = []
        total_bytes = 0
        total_duration_s = 0.0

        for record in ordered_records:
            file_name = _file_name_from_repo_path(record.repo_path)
            output_bytes, duration_s = _build_output_wav(record.wav_bytes, sample_rate=sample_rate)
            if total_bytes + len(output_bytes) > max_bytes:
                _LOGGER.info(
                    "datasets_moe_speech_disk_cap_reached",
                    character=normalized_character,
                    current_total_bytes=total_bytes,
                    max_bytes=max_bytes,
                    next_file=file_name,
                )
                break

            (temp_dir / file_name).write_bytes(output_bytes)
            extracted.append(ExtractedClip(path=file_name, duration_s=duration_s))
            total_bytes += len(output_bytes)
            total_duration_s += duration_s

        index = ExtractionIndex(
            dataset=dataset_repo,
            sample_rate=sample_rate,
            include_nsfw=include_nsfw,
            total_bytes=total_bytes,
            total_duration_s=total_duration_s,
            characters={normalized_character: tuple(extracted)},
        )
        serialized_index = index.to_json()
        (temp_dir / "index.json").write_text(serialized_index, encoding="utf-8")
        temp_dir.replace(out_dir)

    _LOGGER.info(
        "datasets_moe_speech_extracted",
        character=normalized_character,
        sample_count=len(extracted),
        total_bytes=total_bytes,
        total_duration_s=total_duration_s,
    )
    return ExtractionIndex.from_json(serialized_index)


def stream_character_records(
    character: str,
    *,
    dataset_repo: str = DEFAULT_DATASET_REPO,
    hf_token: str | None = None,
) -> Iterator[MoeSpeechRecord]:
    normalized_character = _normalize_character(character)
    list_repo_tree, hf_hub_download = _load_huggingface_helpers()

    prefix = f"data/{normalized_character}/wav"
    entries = list(
        list_repo_tree(
            repo_id=dataset_repo,
            path_in_repo=prefix,
            recursive=True,
            repo_type="dataset",
            token=hf_token,
        )
    )
    sorted_paths = sorted(
        path for entry in entries if (path := _entry_path(entry)).endswith(".wav")
    )

    _LOGGER.info(
        "datasets_moe_speech_listed",
        character=normalized_character,
        dataset_repo=dataset_repo,
        file_count=len(sorted_paths),
    )

    for repo_path in sorted_paths:
        local_path = Path(
            hf_hub_download(
                repo_id=dataset_repo,
                filename=repo_path,
                repo_type="dataset",
                token=hf_token,
            )
        )
        yield MoeSpeechRecord(repo_path=repo_path, wav_bytes=local_path.read_bytes())


def _build_output_wav(wav_bytes: bytes, *, sample_rate: int) -> tuple[bytes, float]:
    source_rate, samples = _read_mono_pcm16_samples(wav_bytes)
    resampled = _resample_samples_linear(samples, source_rate=source_rate, target_rate=sample_rate)
    duration_s = len(resampled) / sample_rate
    if duration_s <= 0.0:
        msg = "moe-speech clips must have positive duration"
        raise UnsupportedAudioFormatError(msg)
    return _encode_wav(resampled, sample_rate=sample_rate), duration_s


def _read_mono_pcm16_samples(wav_bytes: bytes) -> tuple[int, array[int]]:
    try:
        with wave.open(BytesIO(wav_bytes), "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sample_rate = wav_file.getframerate()
            frames = wav_file.readframes(wav_file.getnframes())

        samples = array("h")
        samples.frombytes(frames)
    except (EOFError, ValueError, wave.Error, struct.error) as exc:
        msg = f"moe-speech clips must be valid mono 16-bit PCM WAV files: {exc}"
        raise UnsupportedAudioFormatError(msg) from exc

    if channels != 1:
        msg = "moe-speech clips must be mono WAV files"
        raise UnsupportedAudioFormatError(msg)
    if sample_width != PCM16_SAMPLE_WIDTH_BYTES:
        msg = "moe-speech clips must be 16-bit PCM WAV files"
        raise UnsupportedAudioFormatError(msg)
    if sample_rate <= 0:
        msg = "moe-speech clips must declare a positive sample rate"
        raise UnsupportedAudioFormatError(msg)
    if sys.byteorder != "little":
        samples.byteswap()
    return sample_rate, samples


def _resample_samples_linear(
    samples: array[int],
    *,
    source_rate: int,
    target_rate: int,
) -> array[int]:
    if not samples:
        return array("h")
    if source_rate == target_rate:
        return array("h", samples)

    output_length = max(1, round(len(samples) * target_rate / source_rate))
    if output_length == 1:
        return array("h", [samples[0]])

    resampled = array("h", [0]) * output_length
    for index in range(output_length):
        source_position = index * source_rate / target_rate
        lower_index = min(int(source_position), len(samples) - 1)
        upper_index = min(lower_index + 1, len(samples) - 1)
        fraction = source_position - lower_index
        lower_value = samples[lower_index]
        upper_value = samples[upper_index]
        interpolated = round(lower_value + (upper_value - lower_value) * fraction)
        resampled[index] = _clamp_pcm16(interpolated)
    return resampled


def _encode_wav(samples: array[int], *, sample_rate: int) -> bytes:
    encoded = array("h", samples)
    if sys.byteorder != "little":
        encoded.byteswap()
    with BytesIO() as buffer:
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(encoded.tobytes())
        return buffer.getvalue()


def _clamp_pcm16(value: int) -> int:
    return max(-32_768, min(32_767, value))


def _ensure_nsfw_allowed(*, include_nsfw: bool) -> None:
    if include_nsfw:
        return
    msg = (
        "litagin/moe-speech does not publish a separate non-NSFW subset; "
        "rerun with --include-nsfw or use a different dataset."
    )
    _LOGGER.warning("datasets_moe_speech_nsfw_subset_unavailable")
    raise NsfwSubsetUnavailableError(msg)


def _validate_sample_rate(sample_rate: int) -> None:
    raw_sample_rate: object = sample_rate
    if not isinstance(raw_sample_rate, int) or isinstance(raw_sample_rate, bool):
        msg = "sample_rate must be an integer"
        raise TypeError(msg)
    if not MIN_SAMPLE_RATE <= raw_sample_rate <= MAX_SAMPLE_RATE:
        msg = "sample_rate must be between 16000 and 48000"
        raise ValueError(msg)


def _validate_max_bytes(max_bytes: int) -> None:
    raw_max_bytes: object = max_bytes
    if not isinstance(raw_max_bytes, int) or isinstance(raw_max_bytes, bool):
        msg = "max_bytes must be an integer"
        raise TypeError(msg)
    if raw_max_bytes <= 0:
        msg = "max_bytes must be positive"
        raise ValueError(msg)


def _ensure_output_dir_available(out_dir: Path) -> None:
    if not out_dir.exists():
        return
    if not out_dir.is_dir():
        msg = "out_dir must be a directory"
        raise ValueError(msg)
    if any(out_dir.iterdir()):
        msg = "out_dir must be empty before extraction"
        raise ValueError(msg)


def _normalize_character(character: str) -> str:
    raw_character: object = character
    if not isinstance(raw_character, str):
        msg = "character must be a string"
        raise TypeError(msg)
    normalized = raw_character.strip()
    if not normalized:
        msg = "character must not be blank"
        raise ValueError(msg)
    if _REPO_SEGMENT_RE.fullmatch(normalized) is None:
        msg = "character must be a repository identifier matching [0-9A-Za-z_-]+"
        raise ValueError(msg)
    return normalized


def _character_from_repo_path(repo_path: str) -> str | None:
    match = _REPO_PATH_RE.search(repo_path.replace("\\", "/"))
    if match is None:
        return None
    return match.group("character")


def _file_name_from_repo_path(repo_path: str) -> str:
    match = _REPO_PATH_RE.search(repo_path.replace("\\", "/"))
    if match is None:
        msg = f"unsupported moe-speech path: {repo_path}"
        raise ValueError(msg)
    return match.group("filename")


def _entry_path(entry: object) -> str:
    if isinstance(entry, str):
        return entry
    path = getattr(entry, "path", None)
    if not isinstance(path, str):
        msg = "huggingface_hub list_repo_tree entries must expose a string path"
        raise TypeError(msg)
    return path


def _load_huggingface_helpers() -> tuple[Callable[..., Any], Callable[..., str]]:
    from huggingface_hub import hf_hub_download, list_repo_tree  # noqa: PLC0415

    return list_repo_tree, hf_hub_download
