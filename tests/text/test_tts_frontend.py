from __future__ import annotations

import pytest

from irodori_tts_infra.text import Segment, SegmentKind
from irodori_tts_infra.text.tts_frontend import (
    normalize_tts_text,
    prepare_tts_segments,
    split_tts_text,
)

pytestmark = pytest.mark.unit


def test_normalize_tts_text_removes_decorative_symbols_and_collapses_whitespace() -> None:
    source = "  \uff21\uff22\uff23\uff11\uff12\uff13。今日は……\n静かに\u3000話す♡  "

    assert normalize_tts_text(source) == "ABC123。今日は……静かに 話す"


def test_normalize_tts_text_preserves_space_after_ascii_punctuation() -> None:
    assert normalize_tts_text("Hello, world: test; done") == "Hello, world: test; done"


def test_split_tts_text_prefers_sentence_boundaries() -> None:
    source = "一文目です。二文目です\uff1f三文目です\uff01"

    assert split_tts_text(source, max_chars=20) == [
        "一文目です。",
        "二文目です\uff1f",
        "三文目です\uff01",
    ]


def test_split_tts_text_keeps_closing_quote_with_sentence() -> None:
    source = "「一文です\uff01」続きです。"

    assert split_tts_text(source, max_chars=20) == [
        "「一文です\uff01」",
        "続きです。",
    ]


def test_split_tts_text_keeps_repeated_sentence_ends_together() -> None:
    assert split_tts_text("本当\uff01\uff1f続きです。", max_chars=20) == [
        "本当\uff01\uff1f",
        "続きです。",
    ]


def test_split_tts_text_keeps_closing_quote_when_hard_limit_hits_sentence_end() -> None:
    assert split_tts_text("ああああ。」", max_chars=5) == ["ああああ。」"]


def test_split_tts_text_returns_empty_for_blank_input() -> None:
    assert split_tts_text("  ♡ \n  ", max_chars=20) == []


def test_split_tts_text_does_not_use_soft_boundaries_for_short_sentences() -> None:
    assert split_tts_text("こんにちは、世界です。", max_chars=20) == [
        "こんにちは、世界です。",
    ]


def test_split_tts_text_uses_soft_boundaries_before_hard_limits() -> None:
    source = "朝の教室には柔らかい光が差し込み、窓際の机だけが少し暖かかった。"

    assert split_tts_text(source, max_chars=24) == [
        "朝の教室には柔らかい光が差し込み、",
        "窓際の机だけが少し暖かかった。",
    ]


def test_split_tts_text_hard_splits_text_without_boundaries() -> None:
    source = "あ" * 25

    assert split_tts_text(source, max_chars=10) == ["あ" * 10, "あ" * 10, "あ" * 5]


def test_split_tts_text_rejects_non_positive_max_chars() -> None:
    with pytest.raises(ValueError, match="max_chars"):
        split_tts_text("本文です。", max_chars=0)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_segments": 0}, "max_segments"),
        ({"max_total_chars": 0}, "max_total_chars"),
    ],
)
def test_prepare_tts_segments_rejects_non_positive_limits(
    kwargs: dict[str, int],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        prepare_tts_segments(
            [Segment(kind=SegmentKind.NARRATION, text="本文です。")],
            **kwargs,
        )


def test_prepare_tts_segments_rejects_excessive_total_text() -> None:
    with pytest.raises(ValueError, match="too large"):
        prepare_tts_segments(
            [Segment(kind=SegmentKind.NARRATION, text="あ" * 11)],
            max_total_chars=10,
        )


def test_prepare_tts_segments_rejects_excessive_segment_count() -> None:
    with pytest.raises(ValueError, match="too many TTS segments"):
        prepare_tts_segments(
            [Segment(kind=SegmentKind.NARRATION, text="一。二。三。")],
            max_segments=2,
        )


def test_prepare_tts_segments_preserves_segment_metadata() -> None:
    segments = [
        Segment(
            kind=SegmentKind.DIALOGUE,
            text="近くに来て。もう少しだけ、静かに話して。",
            speaker="カスミ",
            direction="囁くように",
        )
    ]

    assert prepare_tts_segments(segments, max_chars=12) == [
        Segment(
            kind=SegmentKind.DIALOGUE,
            text="近くに来て。",
            speaker="カスミ",
            direction="囁くように",
        ),
        Segment(
            kind=SegmentKind.DIALOGUE,
            text="もう少しだけ、",
            speaker="カスミ",
            direction="囁くように",
        ),
        Segment(
            kind=SegmentKind.DIALOGUE,
            text="静かに話して。",
            speaker="カスミ",
            direction="囁くように",
        ),
    ]
