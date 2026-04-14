from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from irodori_tts_infra.voice_bank import (
    DEFAULT_GENERIC_DIALOGUE_CAPTION,
    DEFAULT_NARRATOR_CAPTION,
    find_characters_markdown,
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


def test_find_characters_markdown_returns_none_when_not_found(tmp_path: Path) -> None:
    story_dir = tmp_path / "chat" / "storyA"
    story_dir.mkdir(parents=True)
    turn_file = story_dir / "turn.md"
    turn_file.write_text("本文", encoding="utf-8")

    assert find_characters_markdown(turn_file) is None


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


def test_load_voice_profile_uses_custom_narrator_caption(tmp_path: Path) -> None:
    characters_md = tmp_path / "characters.md"
    characters_md.write_text("## チヅル\n- **性格**: 明るい\n", encoding="utf-8")

    profile = load_voice_profile(characters_md, narrator_caption="独自の語り。")

    assert profile.narrator_caption == "独自の語り。"
    assert profile.generic_dialogue_caption == DEFAULT_GENERIC_DIALOGUE_CAPTION
