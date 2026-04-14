from __future__ import annotations

import re

from irodori_tts_infra.text.models import SpeakerTag

SPEAKER_TAG_RE = re.compile(r"^【([^:】]+)(?::([^】]+))?】$")


def parse_speaker_tag(source: str) -> SpeakerTag | None:
    match = SPEAKER_TAG_RE.fullmatch(source.strip())
    if match is None:
        return None

    name = match.group(1).strip()
    direction = (match.group(2) or "").strip()
    if not name:
        return None
    if ":" in source and not direction:
        return None

    return SpeakerTag(name=name, direction=direction)
