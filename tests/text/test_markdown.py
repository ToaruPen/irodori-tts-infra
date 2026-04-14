from __future__ import annotations

import pytest

from irodori_tts_infra.text import (
    Segment,
    SegmentKind,
    is_skippable_markdown_line,
    parse_turn_markdown,
    strip_turn_metadata,
)

pytestmark = pytest.mark.unit
METADATA_MARKER = "「「\U0001f3f7\ufe0f情報」:"


def test_parse_turn_markdown_extracts_ordered_tagged_dialogue() -> None:
    content = """
夕暮れの教室は静かだった。
【チヅル】「おやすみなさい、ショウタくん」
廊下の向こうで足音が消えた。
"""

    assert parse_turn_markdown(content) == [
        Segment(kind=SegmentKind.NARRATION, text="夕暮れの教室は静かだった。"),
        Segment(
            kind=SegmentKind.DIALOGUE,
            text="おやすみなさい、ショウタくん",
            speaker="チヅル",
        ),
        Segment(kind=SegmentKind.NARRATION, text="廊下の向こうで足音が消えた。"),
    ]


def test_parse_turn_markdown_extracts_tagged_dialogue_with_direction() -> None:
    content = "【チヅル:小声で囁くように】「しっ……誰かいる」"

    assert parse_turn_markdown(content) == [
        Segment(
            kind=SegmentKind.DIALOGUE,
            text="しっ……誰かいる",
            speaker="チヅル",
            direction="小声で囁くように",
        ),
    ]


def test_parse_turn_markdown_treats_bare_dialogue_as_unknown_speaker() -> None:
    assert parse_turn_markdown("「誰かいるの?」") == [
        Segment(kind=SegmentKind.DIALOGUE, text="誰かいるの?"),
    ]


def test_parse_turn_markdown_skips_headings_and_horizontal_rules() -> None:
    content = """
# Scene title
---
地の文です。
"""

    assert is_skippable_markdown_line("# Scene title") is True
    assert is_skippable_markdown_line("-----") is True
    assert is_skippable_markdown_line("地の文です。") is False
    assert parse_turn_markdown(content) == [
        Segment(kind=SegmentKind.NARRATION, text="地の文です。"),
    ]


def test_strip_turn_metadata_removes_block_after_separator_and_marker() -> None:
    content = f"""
本文です。
【チヅル】「読んでほしい本文」
---
{METADATA_MARKER}
◆ReplyRules: metadata
【チヅル】「これは読まない」
"""

    assert strip_turn_metadata(content).strip() == "本文です。\n【チヅル】「読んでほしい本文」"
    assert parse_turn_markdown(content) == [
        Segment(kind=SegmentKind.NARRATION, text="本文です。"),
        Segment(kind=SegmentKind.DIALOGUE, text="読んでほしい本文", speaker="チヅル"),
    ]


def test_parse_turn_markdown_preserves_inner_japanese_corner_quotes() -> None:
    content = "【チヅル】「『相性表』を見ましたか?」\n「『はい』と答えた。」"

    assert parse_turn_markdown(content) == [
        Segment(
            kind=SegmentKind.DIALOGUE,
            text="『相性表』を見ましたか?",
            speaker="チヅル",
        ),
        Segment(kind=SegmentKind.DIALOGUE, text="『はい』と答えた。"),
    ]


def test_parse_turn_markdown_returns_empty_list_for_empty_file() -> None:
    assert parse_turn_markdown("") == []
    assert parse_turn_markdown("\n# Heading\n---\n") == []


def test_parse_turn_markdown_joins_consecutive_narration_lines() -> None:
    content = """
雨が窓を叩いていた。
部屋の空気は冷たい。

チヅルは息をひそめた。
"""

    assert parse_turn_markdown(content) == [
        Segment(
            kind=SegmentKind.NARRATION,
            text="雨が窓を叩いていた。部屋の空気は冷たい。",
        ),
        Segment(kind=SegmentKind.NARRATION, text="チヅルは息をひそめた。"),
    ]


def test_parse_turn_markdown_keeps_malformed_tags_as_narration() -> None:
    content = """
【チヅル:】「これはタグとして扱わない」
【チヅル「閉じ括弧がない」
"""

    assert parse_turn_markdown(content) == [
        Segment(
            kind=SegmentKind.NARRATION,
            text="【チヅル:】「これはタグとして扱わない」【チヅル「閉じ括弧がない」",
        ),
    ]
