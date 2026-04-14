from __future__ import annotations

import re

from irodori_tts_infra.text.models import Segment, SegmentKind
from irodori_tts_infra.text.speaker_tags import parse_speaker_tag

METADATA_MARKER = "「「\U0001f3f7\ufe0f情報」:"
HEADING_RE = re.compile(r"^#{1,6}\s")
SEPARATOR_RE = re.compile(r"^-{3,}$")
SEPARATOR_LINE_RE = re.compile(r"(?m)^-{3,}\s*$")


def parse_turn_markdown(content: str) -> list[Segment]:
    segments: list[Segment] = []
    narration_lines: list[str] = []

    for line in strip_turn_metadata(content).splitlines():
        stripped = line.strip()
        if is_skippable_markdown_line(stripped):
            _flush_narration(narration_lines, segments)
            continue

        dialogue = _parse_tagged_dialogue(stripped) or _parse_bare_dialogue(stripped)
        if dialogue is not None:
            _flush_narration(narration_lines, segments)
            segments.append(dialogue)
            continue

        narration_lines.append(stripped)

    _flush_narration(narration_lines, segments)
    return segments


def strip_turn_metadata(content: str) -> str:
    marker_index = content.find(METADATA_MARKER)
    if marker_index == -1:
        return content

    before_marker = content[:marker_index]
    separator_matches = list(SEPARATOR_LINE_RE.finditer(before_marker))
    if not separator_matches:
        return content

    return before_marker[: separator_matches[-1].start()].rstrip("\n")


def is_skippable_markdown_line(line: str) -> bool:
    stripped = line.strip()
    return (
        not stripped
        or HEADING_RE.match(stripped) is not None
        or SEPARATOR_RE.match(stripped) is not None
    )


def _parse_tagged_dialogue(line: str) -> Segment | None:
    if not line.startswith("【"):
        return None

    tag_end = line.find("】")
    if tag_end == -1:
        return None

    tag = parse_speaker_tag(line[: tag_end + 1])
    if tag is None:
        return None

    text = _parse_dialogue_quote(line[tag_end + 1 :])
    if text is None:
        return None

    return Segment(
        kind=SegmentKind.DIALOGUE,
        text=text,
        speaker=tag.name,
        direction=tag.direction,
    )


def _parse_bare_dialogue(line: str) -> Segment | None:
    text = _parse_dialogue_quote(line)
    if text is None:
        return None
    return Segment(kind=SegmentKind.DIALOGUE, text=text)


def _parse_dialogue_quote(source: str) -> str | None:
    if not source.startswith("「") or not source.endswith("」"):
        return None

    text = source[1:-1]
    if not text:
        return None
    return text


def _flush_narration(narration_lines: list[str], segments: list[Segment]) -> None:
    if not narration_lines:
        return

    segments.append(Segment(kind=SegmentKind.NARRATION, text="".join(narration_lines)))
    narration_lines.clear()
