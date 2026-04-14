from __future__ import annotations

from irodori_tts_infra.voice_bank.captions import (
    DEFAULT_GENERIC_DIALOGUE_CAPTION,
    DEFAULT_NARRATOR_CAPTION,
    build_voicedesign_caption,
    load_characters_markdown,
    resolve_segment_caption,
)
from irodori_tts_infra.voice_bank.models import CharacterVoice, VoiceProfile
from irodori_tts_infra.voice_bank.repository import (
    find_characters_markdown,
    load_voice_profile,
)

__all__ = [
    "DEFAULT_GENERIC_DIALOGUE_CAPTION",
    "DEFAULT_NARRATOR_CAPTION",
    "CharacterVoice",
    "VoiceProfile",
    "build_voicedesign_caption",
    "find_characters_markdown",
    "load_characters_markdown",
    "load_voice_profile",
    "resolve_segment_caption",
]
