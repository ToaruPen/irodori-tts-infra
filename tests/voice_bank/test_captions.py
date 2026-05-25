from __future__ import annotations

from textwrap import dedent

import pytest

from irodori_tts_infra.voice_bank import load_characters_markdown

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("content", "expected_names"),
    [
        ("## 名前\uff08なまえ\uff09\n- **性格**: 明るい\n", {"名前"}),
        ("### 名前\n- **性格**: 明るい\n", {"名前"}),
        ("# 名前\n- **性格**: 明るい\n", set()),
    ],
)
def test_load_characters_markdown_parses_supported_heading_levels(
    content: str,
    expected_names: set[str],
) -> None:
    assert load_characters_markdown(content) == expected_names


def test_load_characters_markdown_keeps_heading_with_zero_attrs() -> None:
    content = """
## 空の見出し

## ミカ
- **性格**: 未設定
"""

    assert load_characters_markdown(content) == {"空の見出し", "ミカ"}


def test_load_characters_markdown_emits_multiple_blocks() -> None:
    content = dedent("""\
        ## 名前A
        - **性格**: 明るい

        ## 名前B
        - **性格**: 静か
    """)

    assert load_characters_markdown(content) == {"名前A", "名前B"}


@pytest.mark.parametrize("content", ["", "  \n\t  "])
def test_load_characters_markdown_returns_empty_set_for_blank_content(content: str) -> None:
    assert load_characters_markdown(content) == set()


def test_load_characters_markdown_ignores_attrs_without_heading() -> None:
    content = "- **性格**: 明るい\n"

    assert load_characters_markdown(content) == set()


@pytest.mark.parametrize("name", ["設定", "行動者", "主人公", "ヒロイン"])
def test_load_characters_markdown_keeps_previously_skipped_heading_names(name: str) -> None:
    content = f"## {name}\n- **性格**: 明るい\n"

    assert load_characters_markdown(content) == {name}
