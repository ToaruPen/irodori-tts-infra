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


@pytest.mark.parametrize("duration_s", [float("nan"), float("inf")])
def test_extracted_clip_rejects_non_finite_duration(duration_s: float) -> None:
    with pytest.raises(ValueError, match="positive"):
        ExtractedClip(path="alice_000.wav", duration_s=duration_s)


def test_extracted_clip_rejects_blank_path() -> None:
    with pytest.raises(ValueError, match="blank"):
        ExtractedClip(path="   ", duration_s=1.0)


@pytest.mark.parametrize("path", ["", None])
def test_extracted_clip_rejects_empty_or_none_path(path: object) -> None:
    with pytest.raises(ValueError, match="non-empty string"):
        ExtractedClip(path=path, duration_s=1.0)  # type: ignore[arg-type]


@pytest.mark.parametrize("duration_s", ["1.0", True, None])
def test_extracted_clip_constructor_rejects_non_numeric_duration(
    duration_s: object,
) -> None:
    with pytest.raises(ValueError, match="positive number"):
        ExtractedClip(path="alice_000.wav", duration_s=duration_s)  # type: ignore[arg-type]


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


def test_extracted_clip_from_json_dict_rejects_non_string_path() -> None:
    with pytest.raises(TypeError, match="path"):
        ExtractedClip.from_json_dict({"path": 123, "duration_s": 1.0})


@pytest.mark.parametrize("duration_s", ["1.0", True, None])
def test_extracted_clip_from_json_dict_rejects_non_numeric_duration(
    duration_s: object,
) -> None:
    with pytest.raises(TypeError, match="duration_s"):
        ExtractedClip.from_json_dict({"path": "alice_000.wav", "duration_s": duration_s})


def test_extracted_clip_from_json_dict_rejects_non_finite_duration() -> None:
    with pytest.raises(ValueError, match="positive"):
        ExtractedClip.from_json_dict({"path": "alice_000.wav", "duration_s": float("nan")})


@pytest.mark.parametrize("field", ["path", "duration_s"])
def test_extracted_clip_from_json_dict_reports_single_missing_field(
    field: str,
) -> None:
    payload = ALICE_CLIP_A.to_json_dict()
    del payload[field]

    with pytest.raises(TypeError, match=field) as exc_info:
        ExtractedClip.from_json_dict(payload)

    assert "KeyError" not in str(exc_info.value)


def test_extracted_clip_from_json_dict_reports_multiple_missing_fields() -> None:
    payload = ALICE_CLIP_A.to_json_dict()
    del payload["duration_s"]
    del payload["path"]

    with pytest.raises(TypeError) as exc_info:
        ExtractedClip.from_json_dict(payload)

    message = str(exc_info.value)
    assert "path" in message
    assert "duration_s" in message
    assert message.index("duration_s") < message.index("path")
    assert "KeyError" not in message


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


def test_extraction_index_rejects_non_finite_total_duration() -> None:
    with pytest.raises(ValueError, match="total_duration_s"):
        ExtractionIndex(
            dataset="litagin/moe-speech",
            sample_rate=24_000,
            include_nsfw=True,
            total_bytes=0,
            total_duration_s=float("inf"),
            characters={},
        )


def test_extraction_index_rejects_non_bool_include_nsfw() -> None:
    with pytest.raises(TypeError, match="include_nsfw"):
        ExtractionIndex(
            dataset="litagin/moe-speech",
            sample_rate=24_000,
            include_nsfw="false",  # type: ignore[arg-type]
            total_bytes=0,
            total_duration_s=0.0,
            characters={},
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dataset", None),
        ("sample_rate", "24000"),
        ("sample_rate", True),
        ("total_bytes", "0"),
        ("total_bytes", False),
        ("total_duration_s", "0.0"),
        ("total_duration_s", False),
        ("characters", []),
    ],
)
def test_extraction_index_constructor_rejects_invalid_field_types(
    field: str,
    value: object,
) -> None:
    kwargs: dict[str, object] = {
        "characters": {},
        "dataset": "litagin/moe-speech",
        "include_nsfw": True,
        "sample_rate": 24_000,
        "total_bytes": 0,
        "total_duration_s": 0.0,
    }
    kwargs[field] = value

    with pytest.raises(TypeError, match=field):
        ExtractionIndex(**kwargs)  # type: ignore[arg-type]


def test_extraction_index_constructor_rejects_non_string_character_key() -> None:
    with pytest.raises(TypeError, match="character"):
        ExtractionIndex(
            dataset="litagin/moe-speech",
            sample_rate=24_000,
            include_nsfw=True,
            total_bytes=0,
            total_duration_s=0.0,
            characters={1: ()},  # type: ignore[dict-item]
        )


@pytest.mark.parametrize("clips", ["", b"", bytearray(), {ALICE_CLIP_A}])
def test_extraction_index_rejects_unordered_or_text_character_clips(
    clips: object,
) -> None:
    with pytest.raises(TypeError, match="sequence"):
        ExtractionIndex(
            dataset="litagin/moe-speech",
            sample_rate=24_000,
            include_nsfw=True,
            total_bytes=0,
            total_duration_s=0.0,
            characters={"alice": clips},  # type: ignore[dict-item]
        )


def test_extraction_index_rejects_non_iterable_character_clips() -> None:
    with pytest.raises(TypeError, match="sequence"):
        ExtractionIndex(
            dataset="litagin/moe-speech",
            sample_rate=24_000,
            include_nsfw=True,
            total_bytes=0,
            total_duration_s=0.0,
            characters={"alice": object()},  # type: ignore[dict-item]
        )


def test_extraction_index_rejects_non_clip_character_values() -> None:
    with pytest.raises(TypeError, match="ExtractedClip"):
        ExtractionIndex(
            dataset="litagin/moe-speech",
            sample_rate=24_000,
            include_nsfw=True,
            total_bytes=0,
            total_duration_s=0.0,
            characters={"alice": ("oops",)},  # type: ignore[dict-item]
        )


def test_extraction_index_from_json_rejects_non_dict_characters() -> None:
    bad_json = (
        '{"dataset": "litagin/moe-speech", "sample_rate": 24000, "include_nsfw": true,'
        ' "total_bytes": 0, "total_duration_s": 0.0, "characters": ["oops"]}'
    )
    with pytest.raises(TypeError, match="mapping"):
        ExtractionIndex.from_json(bad_json)


@pytest.mark.parametrize("payload", ["[]", "null"])
def test_extraction_index_from_json_rejects_non_object_payload(payload: str) -> None:
    with pytest.raises(TypeError, match="object"):
        ExtractionIndex.from_json(payload)


@pytest.mark.parametrize("include_nsfw", ["false", "true", 0, 1, None])
def test_extraction_index_from_json_rejects_non_bool_include_nsfw(
    include_nsfw: object,
) -> None:
    bad_json = json.dumps(
        {
            "characters": {},
            "dataset": "litagin/moe-speech",
            "include_nsfw": include_nsfw,
            "sample_rate": 24_000,
            "total_bytes": 0,
            "total_duration_s": 0.0,
        }
    )

    with pytest.raises(TypeError, match="include_nsfw"):
        ExtractionIndex.from_json(bad_json)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dataset", None),
        ("sample_rate", "24000"),
        ("sample_rate", True),
        ("total_bytes", "0"),
        ("total_bytes", False),
        ("total_duration_s", "0.0"),
        ("total_duration_s", True),
    ],
)
def test_extraction_index_from_json_rejects_coerced_scalar_fields(
    field: str,
    value: object,
) -> None:
    payload: dict[str, object] = {
        "characters": {},
        "dataset": "litagin/moe-speech",
        "include_nsfw": True,
        "sample_rate": 24_000,
        "total_bytes": 0,
        "total_duration_s": 0.0,
    }
    payload[field] = value

    with pytest.raises(TypeError, match=field):
        ExtractionIndex.from_json(json.dumps(payload))


def test_extraction_index_from_json_rejects_invalid_clip_payload() -> None:
    payload = {
        "characters": {"alice": [[123, "1.0"]]},
        "dataset": "litagin/moe-speech",
        "include_nsfw": True,
        "sample_rate": 24_000,
        "total_bytes": 0,
        "total_duration_s": 0.0,
    }

    with pytest.raises(TypeError, match="path"):
        ExtractionIndex.from_json(json.dumps(payload))


@pytest.mark.parametrize("entries", [[123], [["a", 1, 2]], "oops"])
def test_extraction_index_from_json_rejects_malformed_character_entries(
    entries: object,
) -> None:
    payload = {
        "characters": {"alice": entries},
        "dataset": "litagin/moe-speech",
        "include_nsfw": True,
        "sample_rate": 24_000,
        "total_bytes": 0,
        "total_duration_s": 0.0,
    }

    with pytest.raises(TypeError, match="alice"):
        ExtractionIndex.from_json(json.dumps(payload))


@pytest.mark.parametrize(
    "field",
    [
        "characters",
        "dataset",
        "include_nsfw",
        "sample_rate",
        "total_bytes",
        "total_duration_s",
    ],
)
def test_extraction_index_from_json_reports_single_missing_field(
    field: str,
) -> None:
    payload = _make_index().to_json_dict()
    del payload[field]

    with pytest.raises(TypeError, match=field) as exc_info:
        ExtractionIndex.from_json(json.dumps(payload))

    assert "KeyError" not in str(exc_info.value)


def test_extraction_index_from_json_reports_multiple_missing_fields() -> None:
    payload = _make_index().to_json_dict()
    del payload["total_duration_s"]
    del payload["characters"]
    del payload["sample_rate"]

    with pytest.raises(TypeError) as exc_info:
        ExtractionIndex.from_json(json.dumps(payload))

    message = str(exc_info.value)
    assert "characters" in message
    assert "sample_rate" in message
    assert "total_duration_s" in message
    assert message.index("characters") < message.index("sample_rate")
    assert message.index("sample_rate") < message.index("total_duration_s")
    assert "KeyError" not in message


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


def test_extraction_index_characters_mapping_is_read_only() -> None:
    idx = _make_index()

    with pytest.raises(TypeError):
        idx.characters["bob"] = ()  # type: ignore[index]


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
