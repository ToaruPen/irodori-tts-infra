from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from irodori_tts_infra.voice_bank import (
    PortableVoice,
    SpeakerEmbeddingProfile,
    find_characters_markdown,
    find_speaker_manifest,
    load_voice_profile,
)

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.unit


def _catalog_manifest(
    count: int, *, explicit_metadata: bool
) -> tuple[str, list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    sections = [
        "[narrator]",
        'ref_embed = "speakers/fixture-narrator.speaker.safetensors"',
    ]
    if explicit_metadata:
        sections.extend(
            [
                'voice_id = "fixture-voice-0"',
                'label = "Fixture voice 0"',
                'aliases = ["fixture-alias-0"]',
                "default = true",
            ]
        )
    rows.append(
        {
            "id": "fixture-voice-0" if explicit_metadata else "narrator",
            "label": "Fixture voice 0" if explicit_metadata else "Narrator",
            "aliases": ("fixture-alias-0",) if explicit_metadata else (),
            "default": True,
        }
    )

    for index in range(1, count + 1):
        name = f"fixture-character-{index}"
        sections.extend(
            [
                "",
                f'[characters."{name}"]',
                f'ref_embed = "speakers/fixture-{index}.speaker.safetensors"',
            ]
        )
        if explicit_metadata:
            sections.extend(
                [
                    f'voice_id = "fixture-voice-{index}"',
                    f'label = "Fixture voice {index}"',
                    f'aliases = ["fixture-alias-{index}"]',
                    "default = false",
                ]
            )
        rows.append(
            {
                "id": f"fixture-voice-{index}" if explicit_metadata else name,
                "label": f"Fixture voice {index}" if explicit_metadata else name,
                "aliases": (f"fixture-alias-{index}",) if explicit_metadata else (),
                "default": False,
            }
        )
    return "\n".join(sections) + "\n", rows


def test_find_characters_markdown_finds_file_next_to_turn(tmp_path: Path) -> None:
    story_dir = tmp_path / "chat" / "storyA"
    story_dir.mkdir(parents=True)
    turn_file = story_dir / "turn.md"
    turn_file.write_text("本文", encoding="utf-8")
    characters_md = story_dir / "characters.md"
    characters_md.write_text("# characters", encoding="utf-8")

    assert find_characters_markdown(turn_file) == characters_md


def test_find_characters_markdown_stops_at_chat_directory(tmp_path: Path) -> None:
    story_dir = tmp_path / "chat" / "storyA"
    story_dir.mkdir(parents=True)
    turn_file = story_dir / "turn.md"
    turn_file.write_text("本文", encoding="utf-8")
    (tmp_path / "characters.md").write_text("# outside", encoding="utf-8")

    assert find_characters_markdown(turn_file) is None


def test_find_characters_markdown_walks_up_to_chat_directory(tmp_path: Path) -> None:
    turn_dir = tmp_path / "chat" / "storyA" / "turns"
    turn_dir.mkdir(parents=True)
    turn_file = turn_dir / "turn.md"
    turn_file.write_text("本文", encoding="utf-8")
    characters_md = tmp_path / "chat" / "characters.md"
    characters_md.write_text("# characters", encoding="utf-8")

    assert find_characters_markdown(turn_file) == characters_md


def test_find_characters_markdown_skips_directory_and_walks_up(
    tmp_path: Path,
) -> None:
    story_dir = tmp_path / "chat" / "storyA"
    story_dir.mkdir(parents=True)
    turn_file = story_dir / "turn.md"
    turn_file.write_text("本文", encoding="utf-8")

    (story_dir / "characters.md").mkdir()

    parent_characters = tmp_path / "chat" / "characters.md"
    parent_characters.write_text("# characters", encoding="utf-8")

    assert find_characters_markdown(turn_file) == parent_characters


def test_find_characters_markdown_returns_none_when_not_found(tmp_path: Path) -> None:
    story_dir = tmp_path / "chat" / "storyA"
    story_dir.mkdir(parents=True)
    turn_file = story_dir / "turn.md"
    turn_file.write_text("本文", encoding="utf-8")

    assert find_characters_markdown(turn_file) is None


def test_find_speaker_manifest_walks_up_to_chat_directory(tmp_path: Path) -> None:
    turn_dir = tmp_path / "chat" / "storyA" / "turns"
    turn_dir.mkdir(parents=True)
    turn_file = turn_dir / "turn.md"
    turn_file.write_text("本文", encoding="utf-8")
    manifest = tmp_path / "chat" / "voice_bank_speakers.toml"
    manifest.write_text(
        "[narrator]\nref_embed = 'speakers/narrator.speaker.safetensors'\n", encoding="utf-8"
    )

    assert find_speaker_manifest(turn_file) == manifest


def test_find_speaker_manifest_stops_at_chat_directory(tmp_path: Path) -> None:
    story_dir = tmp_path / "chat" / "storyA"
    story_dir.mkdir(parents=True)
    turn_file = story_dir / "turn.md"
    turn_file.write_text("本文", encoding="utf-8")
    (tmp_path / "voice_bank_speakers.toml").write_text(
        "[narrator]\nref_embed = 'speakers/narrator.speaker.safetensors'\n",
        encoding="utf-8",
    )

    assert find_speaker_manifest(turn_file) is None


def test_load_voice_profile_requires_speaker_manifest() -> None:
    with pytest.raises(ValueError, match="speaker manifest is required"):
        load_voice_profile(None)


@pytest.mark.parametrize("count", [0, 1, 4])
@pytest.mark.parametrize("explicit_metadata", [False, True])
def test_load_voice_profile_builds_catalog_from_runtime_manifest(
    tmp_path: Path,
    count: int,
    explicit_metadata: bool,  # noqa: FBT001
) -> None:
    manifest_content, expected = _catalog_manifest(
        count,
        explicit_metadata=explicit_metadata,
    )
    manifest = tmp_path / "voice_bank_speakers.toml"
    manifest.write_text(manifest_content, encoding="utf-8")

    profile = load_voice_profile(None, speaker_manifest=manifest)

    assert len(profile.catalog) == len(expected)
    for voice, expected_voice in zip(profile.catalog, expected, strict=True):
        assert isinstance(voice, PortableVoice)
        assert voice.id == expected_voice["id"]
        assert voice.label == expected_voice["label"]
        assert voice.aliases == expected_voice["aliases"]
        assert voice.default is expected_voice["default"]
        assert profile.resolve_voice_id(voice.id) is voice
        for alias in voice.aliases:
            assert profile.resolve_voice_id(alias) is voice


@pytest.mark.parametrize(
    "metadata",
    [
        """
voice_id = "shared-id"

[characters."fixture-character"]
ref_embed = "speakers/fixture-character.speaker.safetensors"
voice_id = "shared-id"
""",
        """
aliases = ["shared-alias"]

[characters."fixture-character"]
ref_embed = "speakers/fixture-character.speaker.safetensors"
aliases = ["shared-alias"]
""",
        """
voice_id = "fixture-narrator"
aliases = ["fixture-character"]

[characters."fixture-character"]
ref_embed = "speakers/fixture-character.speaker.safetensors"
voice_id = "fixture-character"
""",
        """
default = true

[characters."fixture-character"]
ref_embed = "speakers/fixture-character.speaker.safetensors"
default = true
""",
    ],
)
def test_load_voice_profile_rejects_ambiguous_catalog_metadata(
    tmp_path: Path,
    metadata: str,
) -> None:
    manifest = tmp_path / "voice_bank_speakers.toml"
    manifest.write_text(
        """
[narrator]
ref_embed = "speakers/fixture-narrator.speaker.safetensors"
"""
        + metadata,
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="catalog"):
        load_voice_profile(None, speaker_manifest=manifest)


def test_load_voice_profile_loads_speaker_embeddings_relative_to_manifest(
    tmp_path: Path,
) -> None:
    characters_md = tmp_path / "characters.md"
    characters_md.write_text(
        """
## チヅル
- **性格**: クール

## ミカ
- **性格**: 明るい
""",
        encoding="utf-8",
    )
    manifest = tmp_path / "voice_bank_speakers.toml"
    manifest.write_text(
        """
[narrator]
ref_embed = "speakers/narrator.speaker.safetensors"

[characters."チヅル"]
ref_embed = "speakers/chizuru.speaker.safetensors"
""",
        encoding="utf-8",
    )

    profile = load_voice_profile(characters_md, speaker_manifest=manifest)

    assert profile.narrator == SpeakerEmbeddingProfile(
        tmp_path / "speakers/narrator.speaker.safetensors",
    )
    assert profile.characters["チヅル"].speaker == SpeakerEmbeddingProfile(
        tmp_path / "speakers/chizuru.speaker.safetensors",
    )
    assert "ミカ" not in profile.characters


def test_load_voice_profile_allows_manifest_without_characters_markdown(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "voice_bank_speakers.toml"
    manifest.write_text(
        """
[narrator]
ref_embed = "speakers/narrator.speaker.safetensors"

[characters."チヅル"]
ref_embed = "speakers/chizuru.speaker.safetensors"
""",
        encoding="utf-8",
    )

    profile = load_voice_profile(None, speaker_manifest=manifest)

    assert set(profile.characters) == {"チヅル"}


def test_load_voice_profile_rejects_missing_characters_markdown_path(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "voice_bank_speakers.toml"
    manifest.write_text(
        "[narrator]\nref_embed = 'speakers/narrator.speaker.safetensors'\n",
        encoding="utf-8",
    )
    missing = tmp_path / "missing.md"

    with pytest.raises(ValueError, match=r"characters_md path does not exist: .*missing\.md"):
        load_voice_profile(missing, speaker_manifest=manifest)


def test_load_voice_profile_rejects_characters_markdown_directory(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "voice_bank_speakers.toml"
    manifest.write_text(
        "[narrator]\nref_embed = 'speakers/narrator.speaker.safetensors'\n",
        encoding="utf-8",
    )
    directory = tmp_path / "characters.md"
    directory.mkdir()

    with pytest.raises(ValueError, match=r"characters_md path is not a file: .*characters\.md"):
        load_voice_profile(directory, speaker_manifest=manifest)


def test_load_voice_profile_rejects_manifest_character_missing_from_markdown(
    tmp_path: Path,
) -> None:
    characters_md = tmp_path / "characters.md"
    characters_md.write_text("## チヅル\n- **性格**: クール\n", encoding="utf-8")
    manifest = tmp_path / "voice_bank_speakers.toml"
    manifest.write_text(
        """
[narrator]
ref_embed = "speakers/narrator.speaker.safetensors"

[characters."いない"]
ref_embed = "speakers/missing.speaker.safetensors"
""",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=r"speaker manifest contains characters not present in characters\.md: いない",
    ):
        load_voice_profile(characters_md, speaker_manifest=manifest)


def test_load_voice_profile_rejects_manifest_character_when_markdown_has_no_names(
    tmp_path: Path,
) -> None:
    characters_md = tmp_path / "characters.md"
    characters_md.write_text("# Characters\n", encoding="utf-8")
    manifest = tmp_path / "voice_bank_speakers.toml"
    manifest.write_text(
        """
[narrator]
ref_embed = "speakers/narrator.speaker.safetensors"

[characters."TYPO"]
ref_embed = "speakers/typo.speaker.safetensors"
""",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=r"speaker manifest contains characters not present in characters\.md: TYPO",
    ):
        load_voice_profile(characters_md, speaker_manifest=manifest)


@pytest.mark.parametrize(
    "ref_embed",
    [
        "/speakers/narrator.speaker.safetensors",
        "C:/speakers/narrator.speaker.safetensors",
        r"C:\speakers\narrator.speaker.safetensors",
    ],
)
def test_load_voice_profile_rejects_absolute_speaker_paths(
    tmp_path: Path,
    ref_embed: str,
) -> None:
    manifest = tmp_path / "voice_bank_speakers.toml"
    manifest.write_text(
        f"""
[narrator]
ref_embed = '{ref_embed}'
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="manifest path values must be relative paths"):
        load_voice_profile(None, speaker_manifest=manifest)


def test_load_voice_profile_rejects_non_speaker_safetensors_ref_embed(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "voice_bank_speakers.toml"
    manifest.write_text(
        """
[narrator]
ref_embed = "speakers/narrator.safetensors"
""",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=r"narrator\.ref_embed must end with \.speaker\.safetensors",
    ):
        load_voice_profile(None, speaker_manifest=manifest)


def test_load_voice_profile_allows_missing_embedding_file_by_default(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "voice_bank_speakers.toml"
    manifest.write_text(
        """
[narrator]
ref_embed = "speakers/narrator.speaker.safetensors"
""",
        encoding="utf-8",
    )

    profile = load_voice_profile(None, speaker_manifest=manifest)

    assert profile.narrator == SpeakerEmbeddingProfile(
        tmp_path / "speakers/narrator.speaker.safetensors",
    )


def test_load_voice_profile_rejects_missing_speaker_embedding_file_when_required(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "voice_bank_speakers.toml"
    manifest.write_text(
        """
[narrator]
ref_embed = "speakers/narrator.speaker.safetensors"
""",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=r"speaker embedding file does not exist: .*narrator\.speaker\.safetensors",
    ):
        load_voice_profile(None, speaker_manifest=manifest, require_embedding_files=True)


def test_load_voice_profile_accepts_existing_speaker_embedding_file_when_required(
    tmp_path: Path,
) -> None:
    speakers_dir = tmp_path / "speakers"
    speakers_dir.mkdir()
    (speakers_dir / "narrator.speaker.safetensors").write_bytes(b"placeholder")
    manifest = tmp_path / "voice_bank_speakers.toml"
    manifest.write_text(
        """
[narrator]
ref_embed = "speakers/narrator.speaker.safetensors"
""",
        encoding="utf-8",
    )

    profile = load_voice_profile(None, speaker_manifest=manifest, require_embedding_files=True)

    assert profile.narrator == SpeakerEmbeddingProfile(
        tmp_path / "speakers/narrator.speaker.safetensors",
    )


def test_load_voice_profile_rejects_missing_character_embedding_file_when_required(
    tmp_path: Path,
) -> None:
    speakers_dir = tmp_path / "speakers"
    speakers_dir.mkdir()
    (speakers_dir / "narrator.speaker.safetensors").write_bytes(b"placeholder")
    characters_md = tmp_path / "characters.md"
    characters_md.write_text("## チヅル\n- **性格**: クール\n", encoding="utf-8")
    manifest = tmp_path / "voice_bank_speakers.toml"
    manifest.write_text(
        """
[narrator]
ref_embed = "speakers/narrator.speaker.safetensors"

[characters."チヅル"]
ref_embed = "speakers/chizuru.speaker.safetensors"
""",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=r"speaker embedding file does not exist: .*chizuru\.speaker\.safetensors",
    ):
        load_voice_profile(characters_md, speaker_manifest=manifest, require_embedding_files=True)


def test_load_voice_profile_accepts_existing_character_embedding_file_when_required(
    tmp_path: Path,
) -> None:
    speakers_dir = tmp_path / "speakers"
    speakers_dir.mkdir()
    (speakers_dir / "narrator.speaker.safetensors").write_bytes(b"placeholder")
    (speakers_dir / "chizuru.speaker.safetensors").write_bytes(b"placeholder")
    characters_md = tmp_path / "characters.md"
    characters_md.write_text("## チヅル\n- **性格**: クール\n", encoding="utf-8")
    manifest = tmp_path / "voice_bank_speakers.toml"
    manifest.write_text(
        """
[narrator]
ref_embed = "speakers/narrator.speaker.safetensors"

[characters."チヅル"]
ref_embed = "speakers/chizuru.speaker.safetensors"
""",
        encoding="utf-8",
    )

    profile = load_voice_profile(
        characters_md, speaker_manifest=manifest, require_embedding_files=True
    )

    assert profile.characters["チヅル"].speaker == SpeakerEmbeddingProfile(
        tmp_path / "speakers/chizuru.speaker.safetensors",
    )


def test_load_voice_profile_rejects_non_speaker_safetensors_character_ref_embed(
    tmp_path: Path,
) -> None:
    characters_md = tmp_path / "characters.md"
    characters_md.write_text("## チヅル\n- **性格**: クール\n", encoding="utf-8")
    manifest = tmp_path / "voice_bank_speakers.toml"
    manifest.write_text(
        """
[narrator]
ref_embed = "speakers/narrator.speaker.safetensors"

[characters."チヅル"]
ref_embed = "speakers/chizuru.safetensors"
""",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=r"characters\.チヅル\.ref_embed must end with \.speaker\.safetensors",
    ):
        load_voice_profile(characters_md, speaker_manifest=manifest)


@pytest.mark.parametrize(
    ("manifest_content", "match"),
    [
        ("[characters]\n", r"narrator is required"),
        ("[narrator]\n", r"narrator\.ref_embed is required"),
        ("[narrator]\nref_embed = 123\n", r"narrator\.ref_embed must be a string"),
        ("[narrator]\nref_embed = ''\n", r"narrator\.ref_embed must not be blank"),
        ("[narrator]\nref_embed = '   '\n", r"narrator\.ref_embed must not be blank"),
        ("narrator = 'bad'\n", r"narrator must be a TOML table"),
        (
            """
[narrator]
ref_embed = "speakers/narrator.speaker.safetensors"
characters = "bad"
""",
            r"narrator\.characters is invalid; define characters at top level",
        ),
        (
            """
[narrator]
ref_embed = "speakers/narrator.speaker.safetensors"

[narrator.characters."ミカ"]
ref_embed = "speakers/mika.speaker.safetensors"
""",
            r"narrator\.characters is invalid; define characters at top level",
        ),
        (
            """
[narrator]
ref_embed = "speakers/narrator.speaker.safetensors"

[characters."ミカ"]
ref_embed = 123
""",
            r"characters\.ミカ\.ref_embed must be a string",
        ),
        (
            """
[narrator]
ref_embed = "speakers/narrator.speaker.safetensors"

[characters."ミカ"]
ref_embed = ""
""",
            r"characters\.ミカ\.ref_embed must not be blank",
        ),
        (
            """
[narrator]
ref_embed = "speakers/narrator.speaker.safetensors"
voice_id = 123
""",
            r"narrator\.voice_id must be a string",
        ),
        (
            """
[narrator]
ref_embed = "speakers/narrator.speaker.safetensors"
label = "   "
""",
            r"narrator\.label must not be blank",
        ),
        (
            """
[narrator]
ref_embed = "speakers/narrator.speaker.safetensors"
aliases = "scalar"
""",
            r"narrator\.aliases must be an array",
        ),
        (
            """
[narrator]
ref_embed = "speakers/narrator.speaker.safetensors"
default = "true"
""",
            r"narrator\.default must be a boolean",
        ),
    ],
)
def test_load_voice_profile_rejects_invalid_speaker_manifest_shape(
    tmp_path: Path,
    manifest_content: str,
    match: str,
) -> None:
    manifest = tmp_path / "voice_bank_speakers.toml"
    manifest.write_text(manifest_content, encoding="utf-8")

    with pytest.raises((TypeError, ValueError), match=match):
        load_voice_profile(None, speaker_manifest=manifest)
