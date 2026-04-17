from __future__ import annotations

import wave
from io import BytesIO
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from irodori_tts_infra.datasets.moe_speech import (
    MoeSpeechRecord,
    NsfwSubsetUnavailableError,
    UnsupportedAudioFormatError,
    extract_character_dataset,
    stream_character_records,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

OUTPUT_SAMPLE_RATE = 24_000
EXPECTED_EXTRACTED_CLIP_COUNT = 2

pytestmark = pytest.mark.unit


def _make_wav_bytes(
    *,
    sample_rate: int = 44_100,
    channels: int = 1,
    seconds: float = 1.0,
    sample_width: int = 2,
) -> bytes:
    frame_count = max(1, round(sample_rate * seconds))
    signed = sample_width > 1
    silent_frame = (0).to_bytes(sample_width, byteorder="little", signed=signed)
    frame_bytes = silent_frame * frame_count * channels
    with BytesIO() as buffer:
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(sample_width)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(frame_bytes)
        return buffer.getvalue()


def _make_empty_wav_bytes(*, sample_rate: int = 44_100, channels: int = 1) -> bytes:
    with BytesIO() as buffer:
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(b"")
        return buffer.getvalue()


def test_extract_character_filters_and_orders_by_character(tmp_path: Path) -> None:
    records = (
        MoeSpeechRecord("data/bob/wav/bob_000.wav", _make_wav_bytes()),
        MoeSpeechRecord("data/alice/wav/alice_002.wav", _make_wav_bytes()),
        MoeSpeechRecord("data/alice/wav/alice_001.wav", _make_wav_bytes()),
    )

    index = extract_character_dataset(
        character="alice",
        out_dir=tmp_path,
        sample_rate=44_100,
        records=records,
    )

    assert index.path_durations_by_character == {
        "alice": (("alice_001.wav", 1.0), ("alice_002.wav", 1.0))
    }
    assert (tmp_path / "alice_001.wav").is_file()
    assert (tmp_path / "alice_002.wav").is_file()
    assert not (tmp_path / "bob_000.wav").exists()


def test_extract_character_returns_empty_index_when_no_records_match(tmp_path: Path) -> None:
    records = (
        MoeSpeechRecord("data/bob/wav/bob_000.wav", _make_wav_bytes()),
        MoeSpeechRecord("data/carol/wav/carol_000.wav", _make_wav_bytes()),
    )

    index = extract_character_dataset(
        character="alice",
        out_dir=tmp_path,
        records=records,
    )

    assert index.characters["alice"] == ()
    assert index.total_bytes == 0
    assert index.total_duration_s == 0.0  # noqa: RUF069 - exact zero is the behavior under test.
    assert not any(tmp_path.glob("*.wav"))


def test_extract_character_tracks_duration_bytes_and_resamples(tmp_path: Path) -> None:
    index = extract_character_dataset(
        character="alice",
        out_dir=tmp_path,
        sample_rate=24_000,
        records=(MoeSpeechRecord("data/alice/wav/alice_000.wav", _make_wav_bytes()),),
    )

    written_path = tmp_path / "alice_000.wav"
    with wave.open(str(written_path), "rb") as wav_file:
        assert wav_file.getframerate() == OUTPUT_SAMPLE_RATE
        assert wav_file.getnchannels() == 1
        assert wav_file.getnframes() == OUTPUT_SAMPLE_RATE

    assert index.total_duration_s == pytest.approx(1.0, abs=1e-3)
    assert index.total_bytes == len(written_path.read_bytes())


def test_extract_character_stops_before_exceeding_disk_cap(tmp_path: Path) -> None:
    source_bytes = _make_wav_bytes()
    expected_output_bytes = len(_make_wav_bytes(sample_rate=OUTPUT_SAMPLE_RATE))
    max_bytes = expected_output_bytes * EXPECTED_EXTRACTED_CLIP_COUNT + 100
    records = tuple(
        MoeSpeechRecord(f"data/alice/wav/alice_{index:03d}.wav", source_bytes) for index in range(3)
    )

    index = extract_character_dataset(
        character="alice",
        out_dir=tmp_path,
        sample_rate=OUTPUT_SAMPLE_RATE,
        max_bytes=max_bytes,
        records=records,
    )

    assert len(index.characters["alice"]) == EXPECTED_EXTRACTED_CLIP_COUNT
    assert index.total_bytes <= max_bytes
    assert not (tmp_path / "alice_002.wav").exists()


def test_extract_character_rejects_blank_character(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="blank"):
        extract_character_dataset(
            character="   ",
            out_dir=tmp_path,
            records=(),
        )


def test_extract_character_rejects_out_of_range_sample_rate(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="between"):
        extract_character_dataset(
            character="alice",
            out_dir=tmp_path,
            sample_rate=8_000,
            records=(),
        )


def test_extract_character_rejects_sample_rate_above_maximum(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="between"):
        extract_character_dataset(
            character="alice",
            out_dir=tmp_path,
            sample_rate=50_000,
            records=(),
        )


def test_extract_character_rejects_non_positive_max_bytes(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="positive"):
        extract_character_dataset(
            character="alice",
            out_dir=tmp_path,
            max_bytes=0,
            records=(),
        )


def test_extract_character_rejects_nsfw_opt_out(tmp_path: Path) -> None:
    with pytest.raises(NsfwSubsetUnavailableError, match="non-NSFW subset"):
        extract_character_dataset(
            character="alice",
            out_dir=tmp_path,
            include_nsfw=False,
            records=(),
        )


def test_extract_character_rejects_non_mono_wav(tmp_path: Path) -> None:
    with pytest.raises(UnsupportedAudioFormatError, match="mono"):
        extract_character_dataset(
            character="alice",
            out_dir=tmp_path,
            records=(
                MoeSpeechRecord(
                    "data/alice/wav/alice_000.wav",
                    _make_wav_bytes(channels=2),
                ),
            ),
        )


def test_extract_character_rejects_non_pcm16_wav(tmp_path: Path) -> None:
    with pytest.raises(UnsupportedAudioFormatError, match="16-bit"):
        extract_character_dataset(
            character="alice",
            out_dir=tmp_path,
            records=(
                MoeSpeechRecord(
                    "data/alice/wav/alice_000.wav",
                    _make_wav_bytes(sample_width=1),
                ),
            ),
        )


def test_extract_character_wraps_malformed_wav_errors(tmp_path: Path) -> None:
    with pytest.raises(UnsupportedAudioFormatError, match="valid"):
        extract_character_dataset(
            character="alice",
            out_dir=tmp_path,
            records=(
                MoeSpeechRecord(
                    "data/alice/wav/alice_000.wav",
                    b"not a wav file",
                ),
            ),
        )


def test_extract_character_rejects_empty_wav_after_resample(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="positive"):
        extract_character_dataset(
            character="alice",
            out_dir=tmp_path,
            sample_rate=24_000,
            records=(
                MoeSpeechRecord(
                    "data/alice/wav/alice_000.wav",
                    _make_empty_wav_bytes(),
                ),
            ),
        )


def test_stream_character_records_downloads_sorted_repo_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    first = tmp_path / "alice_001.wav"
    second = tmp_path / "alice_002.wav"
    first_bytes = _make_wav_bytes()
    second_bytes = _make_wav_bytes(seconds=0.5)
    first.write_bytes(first_bytes)
    second.write_bytes(second_bytes)

    def fake_list_repo_tree(**_kwargs: object) -> list[object]:
        return [
            SimpleNamespace(path="data/alice/wav/alice_002.wav"),
            SimpleNamespace(path="data/alice/wav/alice_001.wav"),
        ]

    def fake_hf_hub_download(**kwargs: object) -> str:
        filename = kwargs["filename"]
        if filename == "data/alice/wav/alice_001.wav":
            return str(first)
        if filename == "data/alice/wav/alice_002.wav":
            return str(second)
        message = f"Unexpected fixture filename: {filename}"
        raise AssertionError(message)

    monkeypatch.setattr(
        "irodori_tts_infra.datasets.moe_speech._load_huggingface_helpers",
        lambda: (fake_list_repo_tree, fake_hf_hub_download),
    )

    records = list(stream_character_records("alice"))

    assert [record.repo_path for record in records] == [
        "data/alice/wav/alice_001.wav",
        "data/alice/wav/alice_002.wav",
    ]
    assert [record.wav_bytes for record in records] == [first_bytes, second_bytes]


def test_extract_character_stops_streaming_once_disk_cap_is_reached(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    yielded_paths: list[str] = []
    source_bytes = _make_wav_bytes()

    def fake_stream_character_records(
        _character: str,
        *,
        dataset_repo: str,
        hf_token: str | None,
    ) -> Iterator[MoeSpeechRecord]:
        del dataset_repo, hf_token
        for index in range(3):
            repo_path = f"data/alice/wav/alice_{index:03d}.wav"
            yielded_paths.append(repo_path)
            yield MoeSpeechRecord(repo_path=repo_path, wav_bytes=source_bytes)

    monkeypatch.setattr(
        "irodori_tts_infra.datasets.moe_speech.stream_character_records",
        fake_stream_character_records,
    )

    max_bytes = len(source_bytes) + 100
    index = extract_character_dataset(
        character="alice",
        out_dir=tmp_path,
        sample_rate=44_100,
        max_bytes=max_bytes,
    )

    assert len(index.characters["alice"]) == 1
    assert yielded_paths == [
        "data/alice/wav/alice_000.wav",
        "data/alice/wav/alice_001.wav",
    ]
