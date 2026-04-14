from __future__ import annotations

import re
from typing import TYPE_CHECKING, Literal

from irodori_tts_infra.text.models import Segment, SegmentKind
from irodori_tts_infra.voice_bank.models import CharacterVoice, VoiceProfile

if TYPE_CHECKING:
    from collections.abc import Mapping

DEFAULT_NARRATOR_CAPTION = (
    "落ち着いた大人の女性が、淡々としたトーンで読み上げている。クリアな音質。"
)
DEFAULT_GENERIC_DIALOGUE_CAPTION = "若い人が自然な口調で話している。"

_HEADING_RE = re.compile(r"^#{2,3}\s+(.+?)(?:\uff08.+?\uff09)?$", re.MULTILINE)
_ATTR_RE = re.compile(r"^-\s+\*\*(.+?)\*\*:\s*(.+)$")

_GENDER_KEYWORDS = {
    "male": ["おとこのこ", "男の子", "男子", "男性", "少年", "男児"],
    "female": ["おんなのこ", "女の子", "女子", "女性", "少女", "女児", "巫女", "ヒロイン"],
}

_AGE_KEYWORDS = {
    "child": [
        "小学生",
        "男児",
        "女児",
        "幼い",
        "子供",
        "男の子",
        "女の子",
        "おとこのこ",
        "おんなのこ",
    ],
    "teen": ["中学", "高校", "10代"],
    "young_adult": ["20代", "大学", "若い"],
}

_PERSONALITY_MAP = {
    "明るい": "明るく楽しそうに",
    "素直": "素直に",
    "元気": "元気よく",
    "真面目": "真面目で丁寧に",
    "クール": "落ち着いたクールな調子で",
    "お人好し": "優しく穏やかに",
    "恥ずかしがり": "はにかみながら控えめに",
    "強気": "自信を持ってはっきりと",
    "無邪気": "無邪気に楽しそうに",
    "好奇心旺盛": "好奇心いっぱいに",
    "天然": "ふんわりとマイペースに",
}

_VOICE_PITCH = {
    "child": "やや高めの声。",
    "teen": "若々しい声。",
    "young_adult": "",
}

_Gender = Literal["male", "female"]
_Age = Literal["child", "teen", "young_adult"]


def load_characters_markdown(content: str) -> dict[str, CharacterVoice]:
    if not content.strip():
        return {}

    blocks = _split_character_blocks(content)
    return {
        name: CharacterVoice(name=name, caption=build_voicedesign_caption(attrs))
        for name, attrs in blocks.items()
    }


def build_voicedesign_caption(attrs: Mapping[str, str]) -> str:
    all_text = " ".join(attrs.values())
    gender = _detect_gender(all_text)
    age = _detect_age(all_text)
    speaking_style = _detect_personality(attrs.get("性格", ""))
    voice_desc = _voice_description(gender, age)
    pitch = _VOICE_PITCH[age]

    return f"{voice_desc}が、{speaking_style}話している。{pitch}"


def resolve_segment_caption(segment: Segment, profile: VoiceProfile) -> str:
    if segment.kind == SegmentKind.NARRATION:
        return profile.narrator_caption

    base = _dialogue_base_caption(segment, profile)
    if not segment.direction:
        return base

    if "話している。" in base:
        return base.replace("話している。", f"{segment.direction}話している。クリアな音質。")
    return f"{base.rstrip('。')}{segment.direction}。クリアな音質。"


def _split_character_blocks(content: str) -> dict[str, dict[str, str]]:
    blocks: dict[str, dict[str, str]] = {}
    current_name: str | None = None
    current_attrs: dict[str, str] = {}

    for line in content.splitlines():
        stripped = line.strip()
        heading_match = _HEADING_RE.match(stripped)
        if heading_match:
            if current_name and current_attrs:
                blocks[current_name] = current_attrs
            current_name = heading_match.group(1).strip()
            current_attrs = {}
            continue

        attr_match = _ATTR_RE.match(stripped)
        if attr_match and current_name:
            current_attrs[attr_match.group(1)] = attr_match.group(2)

    if current_name and current_attrs:
        blocks[current_name] = current_attrs

    return blocks


def _detect_gender(text: str) -> _Gender:
    for keyword in _GENDER_KEYWORDS["male"]:
        if keyword in text:
            return "male"
    for keyword in _GENDER_KEYWORDS["female"]:
        if keyword in text:
            return "female"
    return "female"


def _detect_age(text: str) -> _Age:
    for keyword in _AGE_KEYWORDS["child"]:
        if keyword in text:
            return "child"
    for keyword in _AGE_KEYWORDS["teen"]:
        if keyword in text:
            return "teen"
    for keyword in _AGE_KEYWORDS["young_adult"]:
        if keyword in text:
            return "young_adult"
    return "young_adult"


def _detect_personality(personality_text: str) -> str:
    parts = [phrase for keyword, phrase in _PERSONALITY_MAP.items() if keyword in personality_text]
    if parts:
        return "、".join(parts[:2])
    return "自然な口調で"


def _voice_description(gender: _Gender, age: _Age) -> str:
    if age == "child":
        return "男の子" if gender == "male" else "女の子"
    if age == "teen":
        return f"若い{'男性' if gender == 'male' else '女性'}"
    return "男性" if gender == "male" else "女性"


def _dialogue_base_caption(segment: Segment, profile: VoiceProfile) -> str:
    if segment.speaker and segment.speaker in profile.characters:
        return profile.characters[segment.speaker].caption
    return profile.generic_dialogue_caption
