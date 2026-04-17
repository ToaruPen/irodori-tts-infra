from __future__ import annotations

import json

import pytest

from irodori_tts_infra.datasets.models import ExtractedClip, ExtractionIndex

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ALICE_CLIP_A = ExtractedClip(path="alice_000.wav", duration_s=1.5)
ALICE_CLIP_B = ExtractedClip(path="alice_001.wav", duration_s=2.0)


def _make_index(
    characters: dict[str, tuple[ExtractedClip, ...]] | None = None,
) -> ExtractionIndex:
    if characters is None:
        characters = {"alice": (ALICE_CLIP_A, ALICE_CLIP_B)}
    return ExtractionIndex(
        dataset="litagin/moe-speech",
        sample_rate=24_000,
        include_nsfw=True,
        total_bytes=168_088,
        total_duration_s=3.5,
        characters=characters,
    )


# ---------------------------------------------------------------------------
# ExtractedClip validation
# ---------------------------------------------------------------------------


def test_extracted_clip_requires_positive_duration() -> None:
    with pytest.raises(ValueError, match="positive"):
        ExtractedClip(path="alice_000.wav", duration_s=0)


def test_extracted_clip_rejects_negative_duration() -> None:
    with pytest.raises(ValueError, match="positive"):
        ExtractedClip(path="alice_000.wav", duration_s=-0.1)


def test_extracted_clip_rejects_blank_path() -> None:
    with pytest.raises(ValueError, match="blank"):
        ExtractedClip(path="   ", duration_s=1.0)


def test_extracted_clip_is_frozen() -> None:
    with pytest.raises(AttributeError):
        ALICE_CLIP_A.path = "mutated"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ExtractedClip JSON round-trip
# ---------------------------------------------------------------------------


def test_extracted_clip_to_json_dict_schema() -> None:
    d = ALICE_CLIP_A.to_json_dict()
    assert set(d.keys()) == {"path", "duration_s"}
    assert d["path"] == "alice_000.wav"
    assert d["duration_s"] == pytest.approx(ALICE_CLIP_A.duration_s)


def test_extracted_clip_from_json_dict_round_trip() -> None:
    restored = ExtractedClip.from_json_dict(ALICE_CLIP_A.to_json_dict())
    assert restored == ALICE_CLIP_A


# ---------------------------------------------------------------------------
# ExtractionIndex validation
# ---------------------------------------------------------------------------


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


def test_extraction_index_rejects_negative_total_bytes() -> None:
    with pytest.raises(ValueError, match="total_bytes"):
        ExtractionIndex(
            dataset="litagin/moe-speech",
            sample_rate=24_000,
            include_nsfw=True,
            total_bytes=-1,
            total_duration_s=0.0,
            characters={},
        )


def test_extraction_index_rejects_negative_total_duration() -> None:
    with pytest.raises(ValueError, match="total_duration_s"):
        ExtractionIndex(
            dataset="litagin/moe-speech",
            sample_rate=24_000,
            include_nsfw=True,
            total_bytes=0,
            total_duration_s=-0.1,
            characters={},
        )


def test_extraction_index_from_json_rejects_non_dict_characters() -> None:
    bad_json = (
        '{"dataset": "litagin/moe-speech", "sample_rate": 24000, "include_nsfw": true,'
        ' "total_bytes": 0, "total_duration_s": 0.0, "characters": ["oops"]}'
    )
    with pytest.raises(TypeError, match="mapping"):
        ExtractionIndex.from_json(bad_json)


def test_extraction_index_rejects_blank_character_name() -> None:
    with pytest.raises(ValueError, match="character"):
        ExtractionIndex(
            dataset="litagin/moe-speech",
            sample_rate=24_000,
            include_nsfw=True,
            total_bytes=0,
            total_duration_s=0.0,
            characters={"  ": ()},
        )


def test_extraction_index_is_frozen() -> None:
    idx = _make_index()
    with pytest.raises(AttributeError):
        idx.dataset = "mutated"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ExtractionIndex path_durations_by_character
# ---------------------------------------------------------------------------


def test_extraction_index_path_durations_by_character() -> None:
    idx = _make_index()
    result = idx.path_durations_by_character
    assert result == {
        "alice": (("alice_000.wav", 1.5), ("alice_001.wav", 2.0)),
    }


def test_extraction_index_path_durations_empty_characters() -> None:
    idx = _make_index(characters={})
    assert idx.path_durations_by_character == {}


# ---------------------------------------------------------------------------
# ExtractionIndex JSON round-trip
# ---------------------------------------------------------------------------


def test_extraction_index_to_json_valid() -> None:
    idx = _make_index()
    raw = idx.to_json()
    parsed = json.loads(raw)
    assert parsed["dataset"] == "litagin/moe-speech"
    assert "alice" in parsed["characters"]


def test_extraction_index_to_json_sorted_keys() -> None:
    idx = _make_index()
    raw = idx.to_json()
    top_keys = list(json.loads(raw).keys())
    assert top_keys == sorted(top_keys)


def test_extraction_index_from_json_round_trip() -> None:
    idx = _make_index()
    restored = ExtractionIndex.from_json(idx.to_json())
    assert restored == idx


def test_extraction_index_from_json_empty_characters() -> None:
    idx = _make_index(characters={})
    restored = ExtractionIndex.from_json(idx.to_json())
    assert restored.characters == {}
