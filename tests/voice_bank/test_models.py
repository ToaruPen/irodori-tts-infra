from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
from types import MappingProxyType

import pytest

from irodori_tts_infra.voice_bank import (
    CharacterVoice,
    PortableVoice,
    SpeakerEmbeddingProfile,
    VoiceProfile,
)

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


def _portable_voices(count: int) -> tuple[PortableVoice, ...]:
    return tuple(
        PortableVoice(
            id=f"fixture-voice-{index}",
            label=f"Fixture voice {index}",
            aliases=(f"fixture-alias-{index}",),
            default=index == 0,
            speaker=SpeakerEmbeddingProfile(Path(f"speakers/fixture-{index}.speaker.safetensors")),
        )
        for index in range(count)
    )


@pytest.mark.parametrize("count", [0, 1, 4])
def test_voice_profile_preserves_runtime_catalog_without_fixed_names_or_count(
    count: int,
) -> None:
    voices = _portable_voices(count)
    profile = VoiceProfile(
        characters={},
        narrator=SpeakerEmbeddingProfile(Path("speakers/narrator.speaker.safetensors")),
        catalog=voices,
    )

    assert profile.catalog == voices
    for voice in voices:
        assert profile.resolve_voice_id(voice.id) is voice
        assert profile.resolve_voice_id(voice.aliases[0]) is voice


def test_voice_profile_copies_catalog_and_keeps_lookup_stable() -> None:
    voice = _portable_voices(1)[0]
    source = [voice]
    profile = VoiceProfile(
        characters={},
        narrator=voice.speaker,
        catalog=source,  # type: ignore[arg-type]
    )
    source.clear()

    assert profile.catalog == (voice,)
    assert profile.resolve_voice_id(voice.id) is voice


@pytest.mark.parametrize(
    "voices",
    [
        (
            PortableVoice(
                id="voice-a",
                label="A",
                aliases=(),
                default=False,
                speaker=SpeakerEmbeddingProfile(Path("speakers/a.speaker.safetensors")),
            ),
            PortableVoice(
                id="voice-a",
                label="B",
                aliases=(),
                default=False,
                speaker=SpeakerEmbeddingProfile(Path("speakers/b.speaker.safetensors")),
            ),
        ),
        (
            PortableVoice(
                id="voice-a",
                label="A",
                aliases=("shared",),
                default=False,
                speaker=SpeakerEmbeddingProfile(Path("speakers/a.speaker.safetensors")),
            ),
            PortableVoice(
                id="voice-b",
                label="B",
                aliases=("shared",),
                default=False,
                speaker=SpeakerEmbeddingProfile(Path("speakers/b.speaker.safetensors")),
            ),
        ),
        (
            PortableVoice(
                id="voice-a",
                label="A",
                aliases=("voice-b",),
                default=False,
                speaker=SpeakerEmbeddingProfile(Path("speakers/a.speaker.safetensors")),
            ),
            PortableVoice(
                id="voice-b",
                label="B",
                aliases=(),
                default=False,
                speaker=SpeakerEmbeddingProfile(Path("speakers/b.speaker.safetensors")),
            ),
        ),
        (
            PortableVoice(
                id="voice-a",
                label="A",
                aliases=(),
                default=True,
                speaker=SpeakerEmbeddingProfile(Path("speakers/a.speaker.safetensors")),
            ),
            PortableVoice(
                id="voice-b",
                label="B",
                aliases=(),
                default=True,
                speaker=SpeakerEmbeddingProfile(Path("speakers/b.speaker.safetensors")),
            ),
        ),
    ],
)
def test_voice_profile_rejects_ambiguous_catalog(
    voices: tuple[PortableVoice, ...],
) -> None:
    with pytest.raises(ValueError, match="catalog"):
        VoiceProfile(
            characters={},
            narrator=voices[0].speaker,
            catalog=voices,
        )


def test_portable_voice_normalizes_and_validates_metadata() -> None:
    speaker = SpeakerEmbeddingProfile(Path("speakers/fixture.speaker.safetensors"))
    voice = PortableVoice(
        id="  fixture-id  ",
        label="  Fixture label  ",
        aliases=("  fixture-alias  ",),
        default=False,
        speaker=speaker,
    )

    assert voice.id == "fixture-id"
    assert voice.label == "Fixture label"
    assert voice.aliases == ("fixture-alias",)

    with pytest.raises(ValueError, match="aliases"):
        PortableVoice(
            id="fixture-id",
            label="Fixture label",
            aliases=("same", " same "),
            default=False,
            speaker=speaker,
        )


@pytest.mark.parametrize(
    ("overrides", "match", "error_type"),
    [
        ({"id": " "}, "id", ValueError),
        ({"id": 1}, "id", ValueError),
        ({"label": " "}, "label", ValueError),
        ({"label": 1}, "label", ValueError),
        ({"aliases": "alias"}, "aliases", TypeError),
        ({"aliases": (" ",)}, "aliases", ValueError),
        ({"aliases": (1,)}, "aliases", ValueError),
        ({"default": 1}, "default", TypeError),
        ({"speaker": object()}, "speaker", TypeError),
    ],
)
def test_portable_voice_rejects_invalid_field_types_and_blank_text(
    overrides: dict[str, object],
    match: str,
    error_type: type[Exception],
) -> None:
    values: dict[str, object] = {
        "id": "fixture-id",
        "label": "Fixture label",
        "aliases": (),
        "default": False,
        "speaker": SpeakerEmbeddingProfile(Path("speakers/fixture.speaker.safetensors")),
    }
    values.update(overrides)

    with pytest.raises(error_type, match=match):
        PortableVoice(**values)  # type: ignore[arg-type]
