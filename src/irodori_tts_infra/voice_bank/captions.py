from __future__ import annotations

import re

_HEADING_RE = re.compile(r"^#{2,3}\s+(.+?)(?:\uff08.+?\uff09)?$", re.MULTILINE)
_ATTR_RE = re.compile(r"^-\s+\*\*(.+?)\*\*:\s*(.+)$")


def load_characters_markdown(content: str) -> set[str]:
    if not content.strip():
        return set()

    return set(_split_character_blocks(content))


def _split_character_blocks(content: str) -> dict[str, dict[str, str]]:
    blocks: dict[str, dict[str, str]] = {}
    current_name: str | None = None
    current_attrs: dict[str, str] = {}

    for line in content.splitlines():
        stripped = line.strip()
        heading_match = _HEADING_RE.match(stripped)
        if heading_match:
            if current_name and current_attrs:
                blocks[current_name] = current_attrs
            current_name = heading_match.group(1).strip()
            current_attrs = {}
            continue

        attr_match = _ATTR_RE.match(stripped)
        if attr_match and current_name:
            current_attrs[attr_match.group(1)] = attr_match.group(2)

    if current_name and current_attrs:
        blocks[current_name] = current_attrs

    return blocks
