from __future__ import annotations

from irodori_tts_infra.text.markdown import (
    is_skippable_markdown_line,
    parse_turn_markdown,
    strip_turn_metadata,
)
from irodori_tts_infra.text.models import Segment, SegmentKind, SpeakerTag
from irodori_tts_infra.text.speaker_tags import parse_speaker_tag
from irodori_tts_infra.text.tts_frontend import (
    DEFAULT_TTS_MAX_CHARS,
    DEFAULT_TTS_MAX_SEGMENTS,
    DEFAULT_TURN_TEXT_MAX_CHARS,
    normalize_tts_text,
    prepare_tts_segments,
    split_tts_text,
)

__all__ = [
    "DEFAULT_TTS_MAX_CHARS",
    "DEFAULT_TTS_MAX_SEGMENTS",
    "DEFAULT_TURN_TEXT_MAX_CHARS",
    "Segment",
    "SegmentKind",
    "SpeakerTag",
    "is_skippable_markdown_line",
    "normalize_tts_text",
    "parse_speaker_tag",
    "parse_turn_markdown",
    "prepare_tts_segments",
    "split_tts_text",
    "strip_turn_metadata",
]
