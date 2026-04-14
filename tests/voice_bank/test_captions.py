from __future__ import annotations

from textwrap import dedent

import pytest

from irodori_tts_infra.text import Segment, SegmentKind
from irodori_tts_infra.voice_bank import (
    CharacterVoice,
    VoiceProfile,
    build_voicedesign_caption,
    load_characters_markdown,
    resolve_segment_caption,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("content", "expected_names"),
    [
        ("## 名前\uff08なまえ\uff09\n- **性格**: 明るい\n", ["名前"]),
        ("### 名前\n- **性格**: 明るい\n", ["名前"]),
        ("# 名前\n- **性格**: 明るい\n", []),
    ],
)
def test_load_characters_markdown_parses_supported_heading_levels(
    content: str,
    expected_names: list[str],
) -> None:
    assert list(load_characters_markdown(content)) == expected_names


def test_load_characters_markdown_skips_heading_with_zero_attrs() -> None:
    content = """
## 空の見出し

## ミカ
- **性格**: 未設定
"""

    characters = load_characters_markdown(content)

    assert set(characters) == {"ミカ"}
    assert characters["ミカ"].caption == "女性が、自然な口調で話している。"


def test_load_characters_markdown_captures_distinct_attribute_keys() -> None:
    content = """
## チヅル
- **年齢**: 17歳
- **性格**: 明るい
- **年齢/外見**: 中学生の女子
"""

    characters = load_characters_markdown(content)

    assert characters["チヅル"].caption == "若い女性が、明るく楽しそうに話している。若々しい声。"


def test_load_characters_markdown_emits_multiple_blocks() -> None:
    content = dedent("""\
        ## 名前A
        - **性格**: 明るい

        ## 名前B
        - **性格**: 静か
    """)

    result = load_characters_markdown(content)

    assert set(result.keys()) == {"名前A", "名前B"}
    assert result["名前A"].caption == "女性が、明るく楽しそうに話している。"
    assert result["名前B"].caption == "女性が、自然な口調で話している。"


@pytest.mark.parametrize(
    ("attrs", "expected"),
    [
        ({"説明": "落ち着いた男性"}, "男性が、自然な口調で話している。"),
        ({"説明": "落ち着いた女性"}, "女性が、自然な口調で話している。"),
        ({"説明": "男性でもあり女性でもある"}, "男性が、自然な口調で話している。"),
        ({"説明": "人"}, "女性が、自然な口調で話している。"),
    ],
)
def test_build_voicedesign_caption_detects_gender_with_prototype_precedence(
    attrs: dict[str, str],
    expected: str,
) -> None:
    assert build_voicedesign_caption(attrs) == expected


@pytest.mark.parametrize(
    ("attrs", "expected"),
    [
        ({"説明": "小学生の女子"}, "女の子が、自然な口調で話している。やや高めの声。"),
        ({"説明": "中学生の男性"}, "若い男性が、自然な口調で話している。若々しい声。"),
        ({"説明": "大学生の男性"}, "男性が、自然な口調で話している。"),
        ({"説明": "小学生で中学にも通う男子"}, "男の子が、自然な口調で話している。やや高めの声。"),
        ({"説明": "中学から大学まで知る男性"}, "若い男性が、自然な口調で話している。若々しい声。"),
        ({"説明": "年齢不詳"}, "女性が、自然な口調で話している。"),
    ],
)
def test_build_voicedesign_caption_detects_age_with_prototype_precedence(
    attrs: dict[str, str],
    expected: str,
) -> None:
    assert build_voicedesign_caption(attrs) == expected


def test_build_voicedesign_caption_uses_only_personality_attr_for_style() -> None:
    attrs = {"年齢/外見": "明るい容姿の女性"}

    assert build_voicedesign_caption(attrs) == "女性が、自然な口調で話している。"


@pytest.mark.parametrize(
    ("attrs", "expected"),
    [
        ({"性格": "明るい"}, "女性が、明るく楽しそうに話している。"),
        ({"性格": "素直で元気"}, "女性が、素直に、元気よく話している。"),
        ({"性格": "静か"}, "女性が、自然な口調で話している。"),
    ],
)
def test_build_voicedesign_caption_matches_personality_keywords(
    attrs: dict[str, str],
    expected: str,
) -> None:
    assert build_voicedesign_caption(attrs) == expected


@pytest.mark.parametrize(
    ("attrs", "expected"),
    [
        ({"年齢": "小学生"}, "女の子が、自然な口調で話している。やや高めの声。"),
        ({"年齢": "高校生"}, "若い女性が、自然な口調で話している。若々しい声。"),
        ({"年齢": "20代"}, "女性が、自然な口調で話している。"),
    ],
)
def test_build_voicedesign_caption_adds_pitch_suffix_by_age(
    attrs: dict[str, str],
    expected: str,
) -> None:
    assert build_voicedesign_caption(attrs) == expected


@pytest.mark.parametrize(
    ("character", "direction", "expected"),
    [
        (
            CharacterVoice(
                name="ショウタ",
                caption="男の子が、元気よく話している。やや高めの声。",
            ),
            "小声で",
            "男の子が、元気よく小声で話している。クリアな音質。やや高めの声。",
        ),
        (
            CharacterVoice(
                name="チヅル",
                caption="若い女性が、落ち着いたクールな調子で話している。若々しい声。",
            ),
            "震えながら",
            "若い女性が、落ち着いたクールな調子で震えながら話している。クリアな音質。若々しい声。",
        ),
    ],
)
def test_resolve_segment_caption_keeps_pitch_after_direction_injection(
    character: CharacterVoice,
    direction: str,
    expected: str,
) -> None:
    profile = VoiceProfile(
        characters={character.name: character},
        narrator_caption="ナレーター",
        generic_dialogue_caption="汎用",
    )
    segment = Segment(
        kind=SegmentKind.DIALOGUE,
        text="本文",
        speaker=character.name,
        direction=direction,
    )

    assert resolve_segment_caption(segment, profile) == expected


def test_resolve_segment_caption_returns_narrator_caption_for_narration() -> None:
    profile = VoiceProfile(
        characters={"チヅル": CharacterVoice(name="チヅル", caption="キャラ")},
        narrator_caption="語り手の声。",
        generic_dialogue_caption="汎用",
    )
    segment = Segment(
        kind=SegmentKind.NARRATION,
        text="本文",
        speaker="チヅル",
        direction="慌てて",
    )

    assert resolve_segment_caption(segment, profile) == "語り手の声。"


def test_resolve_segment_caption_returns_known_speaker_base_without_direction() -> None:
    profile = VoiceProfile(
        characters={"チヅル": CharacterVoice(name="チヅル", caption="チヅルの声。")},
        narrator_caption="ナレーター",
        generic_dialogue_caption="汎用",
    )
    segment = Segment(kind=SegmentKind.DIALOGUE, text="本文", speaker="チヅル")

    assert resolve_segment_caption(segment, profile) == "チヅルの声。"


def test_resolve_segment_caption_uses_generic_caption_for_unknown_speaker() -> None:
    profile = VoiceProfile(
        characters={},
        narrator_caption="ナレーター",
        generic_dialogue_caption="若い人が自然な口調で話している。",
    )
    segment = Segment(kind=SegmentKind.DIALOGUE, text="本文", speaker="不明")

    assert resolve_segment_caption(segment, profile) == "若い人が自然な口調で話している。"


def test_resolve_segment_caption_splices_direction_before_talking_phrase() -> None:
    profile = VoiceProfile(
        characters={},
        narrator_caption="ナレーター",
        generic_dialogue_caption="若い人が自然な口調で話している。",
    )
    segment = Segment(
        kind=SegmentKind.DIALOGUE,
        text="本文",
        speaker=None,
        direction="慌てて",
    )

    assert (
        resolve_segment_caption(segment, profile)
        == "若い人が自然な口調で慌てて話している。クリアな音質。"
    )


def test_resolve_segment_caption_appends_direction_to_degenerate_base() -> None:
    profile = VoiceProfile(
        characters={},
        narrator_caption="ナレーター",
        generic_dialogue_caption="カスタム音声。",
    )
    segment = Segment(
        kind=SegmentKind.DIALOGUE,
        text="本文",
        speaker=None,
        direction="急いで",
    )

    assert resolve_segment_caption(segment, profile) == "カスタム音声急いで。クリアな音質。"


def test_resolve_segment_caption_leaves_base_unchanged_for_empty_direction() -> None:
    profile = VoiceProfile(
        characters={},
        narrator_caption="ナレーター",
        generic_dialogue_caption="若い人が自然な口調で話している。",
    )
    segment = Segment(kind=SegmentKind.DIALOGUE, text="本文", direction="")

    assert resolve_segment_caption(segment, profile) == "若い人が自然な口調で話している。"


@pytest.mark.parametrize("content", ["", "  \n\t  "])
def test_load_characters_markdown_returns_empty_map_for_blank_content(content: str) -> None:
    assert load_characters_markdown(content) == {}


def test_load_characters_markdown_ignores_attrs_without_heading() -> None:
    content = "- **性格**: 明るい\n"

    assert load_characters_markdown(content) == {}


@pytest.mark.parametrize("name", ["設定", "行動者", "主人公", "ヒロイン"])
def test_load_characters_markdown_keeps_previously_skipped_heading_names(name: str) -> None:
    content = f"## {name}\n- **性格**: 明るい\n"

    characters = load_characters_markdown(content)

    assert characters[name].caption == "女性が、明るく楽しそうに話している。"


def test_build_voicedesign_caption_is_deterministic() -> None:
    attrs = {"性格": "明るい元気", "年齢": "高校生の女子"}

    first = build_voicedesign_caption(attrs)
    second = build_voicedesign_caption(attrs)

    assert first == second
