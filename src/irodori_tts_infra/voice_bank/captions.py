from __future__ import annotations

import re

_HEADING_RE = re.compile(r"^#{2,3}\s+(.+?)(?:\uff08.+?\uff09)?$", re.MULTILINE)


def load_characters_markdown(content: str) -> set[str]:
    if not content.strip():
        return set()

    return {
        match.group(1).strip()
        for line in content.splitlines()
        if (match := _HEADING_RE.match(line.strip())) is not None
    }
