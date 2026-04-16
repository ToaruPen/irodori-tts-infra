from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from irodori_tts_infra.voice_bank import (
    DEFAULT_GENERIC_DIALOGUE_CAPTION,
    DEFAULT_NARRATOR_CAPTION,
    RVCProfile,
    find_characters_markdown,
    find_rvc_manifest,
    load_voice_profile,
)

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.unit


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


def test_find_rvc_manifest_walks_up_to_chat_directory(tmp_path: Path) -> None:
    turn_dir = tmp_path / "chat" / "storyA" / "turns"
    turn_dir.mkdir(parents=True)
    turn_file = turn_dir / "turn.md"
    turn_file.write_text("本文", encoding="utf-8")
    manifest = tmp_path / "chat" / "voice_bank_rvc.toml"
    manifest.write_text("[characters]\n", encoding="utf-8")

    assert find_rvc_manifest(turn_file) == manifest


def test_find_rvc_manifest_stops_at_chat_directory(tmp_path: Path) -> None:
    story_dir = tmp_path / "chat" / "storyA"
    story_dir.mkdir(parents=True)
    turn_file = story_dir / "turn.md"
    turn_file.write_text("本文", encoding="utf-8")
    (tmp_path / "voice_bank_rvc.toml").write_text("[characters]\n", encoding="utf-8")

    assert find_rvc_manifest(turn_file) is None


def test_load_voice_profile_returns_defaults_for_none_path() -> None:
    profile = load_voice_profile(None)

    assert profile.characters == {}
    assert profile.narrator_caption == DEFAULT_NARRATOR_CAPTION
    assert profile.generic_dialogue_caption == DEFAULT_GENERIC_DIALOGUE_CAPTION


def test_load_voice_profile_returns_defaults_for_missing_file(tmp_path: Path) -> None:
    profile = load_voice_profile(tmp_path / "missing.md")

    assert profile.characters == {}
    assert profile.narrator_caption == DEFAULT_NARRATOR_CAPTION
    assert profile.generic_dialogue_caption == DEFAULT_GENERIC_DIALOGUE_CAPTION


def test_load_voice_profile_returns_defaults_for_directory_path(
    tmp_path: Path,
) -> None:
    characters_dir = tmp_path / "characters.md"
    characters_dir.mkdir()

    profile = load_voice_profile(characters_dir)

    assert profile.characters == {}
    assert profile.narrator_caption == DEFAULT_NARRATOR_CAPTION
    assert profile.generic_dialogue_caption == DEFAULT_GENERIC_DIALOGUE_CAPTION


def test_load_voice_profile_loads_valid_characters_file(tmp_path: Path) -> None:
    characters_md = tmp_path / "characters.md"
    characters_md.write_text(
        "## チヅル\n- **性格**: クール\n- **年齢/外見**: 高校生の女子\n",
        encoding="utf-8",
    )

    profile = load_voice_profile(characters_md)

    assert set(profile.characters) == {"チヅル"}
    assert (
        profile.characters["チヅル"].caption
        == "若い女性が、落ち着いたクールな調子で話している。若々しい声。"
    )
    assert profile.characters["チヅル"].rvc is None


def test_load_voice_profile_uses_custom_narrator_caption(tmp_path: Path) -> None:
    characters_md = tmp_path / "characters.md"
    characters_md.write_text("## チヅル\n- **性格**: 明るい\n", encoding="utf-8")

    profile = load_voice_profile(characters_md, narrator_caption="独自の語り。")

    assert profile.narrator_caption == "独自の語り。"
    assert profile.generic_dialogue_caption == DEFAULT_GENERIC_DIALOGUE_CAPTION


def test_load_voice_profile_merges_rvc_manifest_paths_relative_to_manifest(
    tmp_path: Path,
) -> None:
    characters_md = tmp_path / "characters.md"
    characters_md.write_text(
        "## チヅル\n- **性格**: クール\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "voice_bank_rvc.toml"
    manifest.write_text(
        """
[characters."チヅル"]
model_path = "models/chizuru.pth"
sample_rate = 40000
neutral_prototype = "prototypes/chizuru-neutral.npy"

[characters."チヅル".state_prototypes]
happy = "prototypes/chizuru-happy.npy"
whisper = "prototypes/chizuru-whisper.npy"
""",
        encoding="utf-8",
    )

    profile = load_voice_profile(characters_md, rvc_manifest=manifest)

    assert profile.characters["チヅル"].rvc == RVCProfile(
        model_path=tmp_path / "models/chizuru.pth",
        sample_rate=40000,
        neutral_prototype=tmp_path / "prototypes/chizuru-neutral.npy",
        state_prototypes={
            "happy": tmp_path / "prototypes/chizuru-happy.npy",
            "whisper": tmp_path / "prototypes/chizuru-whisper.npy",
        },
    )


def test_load_voice_profile_keeps_markdown_character_without_manifest_entry(
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
    manifest = tmp_path / "voice_bank_rvc.toml"
    manifest.write_text(
        """
[characters."チヅル"]
model_path = "models/chizuru.pth"
sample_rate = 40000
""",
        encoding="utf-8",
    )

    profile = load_voice_profile(characters_md, rvc_manifest=manifest)

    assert profile.characters["チヅル"].rvc is not None
    assert profile.characters["ミカ"].rvc is None


def test_load_voice_profile_rejects_manifest_character_missing_from_markdown(
    tmp_path: Path,
) -> None:
    characters_md = tmp_path / "characters.md"
    characters_md.write_text("## チヅル\n- **性格**: クール\n", encoding="utf-8")
    manifest = tmp_path / "voice_bank_rvc.toml"
    manifest.write_text(
        """
[characters."いない"]
model_path = "models/missing.pth"
sample_rate = 40000
""",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=r"RVC manifest contains characters not present in characters\.md: いない",
    ):
        load_voice_profile(characters_md, rvc_manifest=manifest)


@pytest.mark.parametrize(
    "model_path",
    [
        "/models/chizuru.pth",
        "C:/models/chizuru.pth",
        r"C:\models\chizuru.pth",
    ],
)
def test_load_voice_profile_rejects_absolute_rvc_model_path(
    tmp_path: Path,
    model_path: str,
) -> None:
    characters_md = tmp_path / "characters.md"
    characters_md.write_text("## チヅル\n- **性格**: クール\n", encoding="utf-8")
    manifest = tmp_path / "voice_bank_rvc.toml"
    manifest.write_text(
        f"""
[characters."チヅル"]
model_path = '{model_path}'
sample_rate = 40000
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="manifest path values must be relative paths"):
        load_voice_profile(characters_md, rvc_manifest=manifest)


@pytest.mark.parametrize(
    ("field_line", "match"),
    [
        ('sample_rate = "40000"', r"characters\.チヅル\.sample_rate must be an integer"),
        ("sample_rate = true", r"characters\.チヅル\.sample_rate must be an integer"),
    ],
)
def test_load_voice_profile_rejects_invalid_rvc_sample_rate_type(
    tmp_path: Path,
    field_line: str,
    match: str,
) -> None:
    characters_md = tmp_path / "characters.md"
    characters_md.write_text("## チヅル\n- **性格**: クール\n", encoding="utf-8")
    manifest = tmp_path / "voice_bank_rvc.toml"
    manifest.write_text(
        f"""
[characters."チヅル"]
model_path = "models/chizuru.pth"
{field_line}
""",
        encoding="utf-8",
    )

    with pytest.raises(TypeError, match=match):
        load_voice_profile(characters_md, rvc_manifest=manifest)


def test_load_voice_profile_rejects_invalid_rvc_neutral_prototype_type(
    tmp_path: Path,
) -> None:
    characters_md = tmp_path / "characters.md"
    characters_md.write_text("## チヅル\n- **性格**: クール\n", encoding="utf-8")
    manifest = tmp_path / "voice_bank_rvc.toml"
    manifest.write_text(
        """
[characters."チヅル"]
model_path = "models/chizuru.pth"
sample_rate = 40000
neutral_prototype = 123
""",
        encoding="utf-8",
    )

    with pytest.raises(
        TypeError,
        match=r"characters\.チヅル\.neutral_prototype must be a string",
    ):
        load_voice_profile(characters_md, rvc_manifest=manifest)


def test_load_voice_profile_rejects_invalid_rvc_state_prototypes_type(
    tmp_path: Path,
) -> None:
    characters_md = tmp_path / "characters.md"
    characters_md.write_text("## チヅル\n- **性格**: クール\n", encoding="utf-8")
    manifest = tmp_path / "voice_bank_rvc.toml"
    manifest.write_text(
        """
[characters."チヅル"]
model_path = "models/chizuru.pth"
sample_rate = 40000
state_prototypes = "prototypes/chizuru-happy.npy"
""",
        encoding="utf-8",
    )

    with pytest.raises(
        TypeError,
        match=r"characters\.チヅル\.state_prototypes must be a TOML table",
    ):
        load_voice_profile(characters_md, rvc_manifest=manifest)
