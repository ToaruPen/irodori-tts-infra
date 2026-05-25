from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
from types import MappingProxyType

import pytest

from irodori_tts_infra.voice_bank import CharacterVoice, SpeakerEmbeddingProfile, VoiceProfile

pytestmark = pytest.mark.unit


def test_speaker_embedding_profile_compares_by_value() -> None:
    left = SpeakerEmbeddingProfile(ref_embed=Path("speakers/mika.speaker.safetensors"))
    right = SpeakerEmbeddingProfile(ref_embed=Path("speakers/mika.speaker.safetensors"))

    assert left == right


def test_speaker_embedding_profile_is_frozen() -> None:
    profile = SpeakerEmbeddingProfile(ref_embed=Path("speakers/mika.speaker.safetensors"))

    with pytest.raises(FrozenInstanceError):
        profile.ref_embed = Path("speakers/other.speaker.safetensors")  # type: ignore[misc]


def test_speaker_embedding_profile_normalizes_path_values() -> None:
    profile = SpeakerEmbeddingProfile(
        ref_embed="speakers/mika.speaker.safetensors",  # type: ignore[arg-type]
    )

    assert profile.ref_embed == Path("speakers/mika.speaker.safetensors")


def test_voice_profile_requires_narrator_and_character_speakers() -> None:
    narrator = SpeakerEmbeddingProfile(Path("speakers/narrator.speaker.safetensors"))
    mika = CharacterVoice(
        name="ミカ",
        speaker=SpeakerEmbeddingProfile(Path("speakers/mika.speaker.safetensors")),
    )

    profile = VoiceProfile(characters={"ミカ": mika}, narrator=narrator)

    assert profile.narrator is narrator
    assert profile.characters["ミカ"] is mika


def test_voice_profile_copies_characters_into_read_only_mapping() -> None:
    narrator = SpeakerEmbeddingProfile(Path("speakers/narrator.speaker.safetensors"))
    mika = CharacterVoice(
        name="ミカ",
        speaker=SpeakerEmbeddingProfile(Path("speakers/mika.speaker.safetensors")),
    )
    characters = {"ミカ": mika}

    profile = VoiceProfile(characters=characters, narrator=narrator)
    characters.clear()

    assert isinstance(profile.characters, MappingProxyType)
    assert profile.characters["ミカ"] is mika
    with pytest.raises(TypeError):
        profile.characters["別名"] = mika  # type: ignore[index]
