from __future__ import annotations

from irodori_tts_infra.voice_bank.captions import (
    DEFAULT_GENERIC_DIALOGUE_CAPTION,
    DEFAULT_NARRATOR_CAPTION,
    build_voicedesign_caption,
    load_characters_markdown,
    resolve_segment_caption,
)
from irodori_tts_infra.voice_bank.models import CharacterVoice, RVCProfile, VoiceProfile
from irodori_tts_infra.voice_bank.repository import (
    find_characters_markdown,
    find_rvc_manifest,
    load_voice_profile,
)

__all__ = [
    "DEFAULT_GENERIC_DIALOGUE_CAPTION",
    "DEFAULT_NARRATOR_CAPTION",
    "CharacterVoice",
    "RVCProfile",
    "VoiceProfile",
    "build_voicedesign_caption",
    "find_characters_markdown",
    "find_rvc_manifest",
    "load_characters_markdown",
    "load_voice_profile",
    "resolve_segment_caption",
]
