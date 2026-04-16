from __future__ import annotations

import pytest

from irodori_tts_infra.datasets.models import ExtractedClip, ExtractionIndex

pytestmark = pytest.mark.unit


def test_extracted_clip_requires_positive_duration() -> None:
    with pytest.raises(ValueError, match="positive"):
        ExtractedClip(path="alice_000.wav", duration_s=0)


def test_extraction_index_rejects_blank_dataset() -> None:
    with pytest.raises(ValueError, match="dataset"):
        ExtractionIndex(
            dataset=" ",
            sample_rate=24_000,
            include_nsfw=True,
            total_bytes=0,
            total_duration_s=0.0,
            characters={},
        )


def test_extraction_index_rejects_out_of_range_sample_rate() -> None:
    with pytest.raises(ValueError, match="sample_rate"):
        ExtractionIndex(
            dataset="litagin/moe-speech",
            sample_rate=8_000,
            include_nsfw=True,
            total_bytes=0,
            total_duration_s=0.0,
            characters={},
        )


def test_extraction_index_round_trips_json() -> None:
    index = ExtractionIndex(
        dataset="litagin/moe-speech",
        sample_rate=24_000,
        include_nsfw=True,
        total_bytes=123,
        total_duration_s=2.5,
        characters={
            "alice": (
                ExtractedClip(path="alice_000.wav", duration_s=1.0),
                ExtractedClip(path="alice_001.wav", duration_s=1.5),
            )
        },
    )

    restored = ExtractionIndex.from_json(index.to_json())

    assert restored == index


def test_extraction_index_exposes_path_duration_mapping() -> None:
    index = ExtractionIndex(
        dataset="litagin/moe-speech",
        sample_rate=24_000,
        include_nsfw=True,
        total_bytes=456,
        total_duration_s=1.0,
        characters={"alice": (ExtractedClip(path="alice_000.wav", duration_s=1.0),)},
    )

    assert index.path_durations_by_character == {"alice": (("alice_000.wav", 1.0),)}
