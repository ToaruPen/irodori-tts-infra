from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class SegmentKind(StrEnum):
    NARRATION = "narration"
    DIALOGUE = "dialogue"


@dataclass(frozen=True, slots=True)
class SpeakerTag:
    name: str
    direction: str = ""


@dataclass(frozen=True, slots=True)
class Segment:
    kind: SegmentKind
    text: str
    speaker: str | None = None
    direction: str = ""
