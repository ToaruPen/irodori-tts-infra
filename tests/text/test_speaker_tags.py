from __future__ import annotations

import pytest

from irodori_tts_infra.text import SpeakerTag, parse_speaker_tag

pytestmark = pytest.mark.unit


def test_parse_speaker_tag_accepts_name_only_tag() -> None:
    assert parse_speaker_tag("【チヅル】") == SpeakerTag(name="チヅル")


def test_parse_speaker_tag_accepts_tag_with_outer_whitespace() -> None:
    assert parse_speaker_tag(" 【チヅル】 ") == SpeakerTag(name="チヅル")


def test_parse_speaker_tag_accepts_voice_direction() -> None:
    assert parse_speaker_tag("【チヅル:穏やかに微笑みながら】") == SpeakerTag(
        name="チヅル",
        direction="穏やかに微笑みながら",
    )


@pytest.mark.parametrize(
    "source",
    [
        "【チヅル】「おやすみなさい」",
        "【】",
        "【   】",
        "【チヅル:】",
        "【チヅル:   】",
        "【:小声で】",
        "チヅル",
        "【チヅル",
        "チヅル】",
    ],
)
def test_parse_speaker_tag_rejects_non_tag_or_empty_parts(source: str) -> None:
    assert parse_speaker_tag(source) is None
