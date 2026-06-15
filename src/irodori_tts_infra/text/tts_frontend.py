from __future__ import annotations

import re
import unicodedata
from typing import TYPE_CHECKING

from irodori_tts_infra.text.models import Segment

if TYPE_CHECKING:
    from collections.abc import Iterable

DEFAULT_TTS_MAX_CHARS = 120
DEFAULT_TTS_MAX_SEGMENTS = 500
DEFAULT_TURN_TEXT_MAX_CHARS = 60_000
SENTENCE_ENDS = frozenset("\u3002\uff01\uff1f!?")
TRAILING_CLOSERS = frozenset("\u300d\u300f\uff09\u3011\u300b\u201d\u2019")
SOFT_BREAKS = frozenset("\u3001\uff0c,\uff1b;\uff1a:")
SPACE_TRIM_PUNCTUATION = "\u3002\uff01\uff1f\u3001\uff0c\uff1b\uff1a\u2026"
IDEOGRAPHIC_SPACE = "\u3000"
FULLWIDTH_DIGIT_START = ord("\uff10")
FULLWIDTH_DIGIT_END = ord("\uff19")
FULLWIDTH_UPPER_START = ord("\uff21")
FULLWIDTH_UPPER_END = ord("\uff3a")
FULLWIDTH_LOWER_START = ord("\uff41")
FULLWIDTH_LOWER_END = ord("\uff5a")
DECORATIVE_RE = re.compile(r"[♡♥]+")
WHITESPACE_RE = re.compile(r"\s+")
SPACE_BEFORE_PUNCTUATION_RE = re.compile(r" +([" + SPACE_TRIM_PUNCTUATION + r"])")
SPACE_AFTER_PUNCTUATION_RE = re.compile(r"([" + SPACE_TRIM_PUNCTUATION + r"]) +")


def normalize_tts_text(text: str) -> str:
    normalized = "".join(_normalize_tts_char(char) for char in text)
    normalized = DECORATIVE_RE.sub("", normalized)
    normalized = WHITESPACE_RE.sub(" ", normalized)
    normalized = SPACE_BEFORE_PUNCTUATION_RE.sub(r"\1", normalized)
    normalized = SPACE_AFTER_PUNCTUATION_RE.sub(r"\1", normalized)
    return normalized.strip()


def split_tts_text(text: str, *, max_chars: int = DEFAULT_TTS_MAX_CHARS) -> list[str]:
    if max_chars <= 0:
        msg = "max_chars must be a positive integer"
        raise ValueError(msg)

    normalized = normalize_tts_text(text)
    if not normalized:
        return []

    chunks: list[str] = []
    for sentence in _sentence_chunks(normalized):
        chunks.extend(_split_long_sentence(sentence, max_chars=max_chars))
    return chunks


def prepare_tts_segments(
    segments: Iterable[Segment],
    *,
    max_chars: int = DEFAULT_TTS_MAX_CHARS,
    max_segments: int = DEFAULT_TTS_MAX_SEGMENTS,
    max_total_chars: int = DEFAULT_TURN_TEXT_MAX_CHARS,
) -> list[Segment]:
    if max_segments <= 0:
        msg = "max_segments must be a positive integer"
        raise ValueError(msg)
    if max_total_chars <= 0:
        msg = "max_total_chars must be a positive integer"
        raise ValueError(msg)

    prepared: list[Segment] = []
    total_chars = 0
    for segment in segments:
        total_chars += len(segment.text)
        if total_chars > max_total_chars:
            msg = "turn file is too large for read-aloud synthesis"
            raise ValueError(msg)

        for text in split_tts_text(segment.text, max_chars=max_chars):
            prepared.append(
                Segment(
                    kind=segment.kind,
                    text=text,
                    speaker=segment.speaker,
                    direction=segment.direction,
                )
            )
            if len(prepared) > max_segments:
                msg = "too many TTS segments; reduce input size or chunk manually"
                raise ValueError(msg)
    return prepared


def _sentence_chunks(text: str) -> list[str]:
    chunks: list[str] = []
    start = 0
    index = 0
    while index < len(text):
        if text[index] not in SENTENCE_ENDS:
            index += 1
            continue

        end = index + 1
        while end < len(text) and text[end] in SENTENCE_ENDS:
            end += 1
        while end < len(text) and text[end] in TRAILING_CLOSERS:
            end += 1
        chunk = text[start:end].strip()
        chunks.append(chunk)
        start = end
        index = end

    remainder = text[start:].strip()
    if remainder:
        chunks.append(remainder)
    return chunks


def _split_long_sentence(text: str, *, max_chars: int) -> list[str]:
    chunks: list[str] = []
    remaining = text
    while len(remaining) > max_chars:
        cut = _last_soft_break(remaining, max_chars) or max_chars
        cut = _extend_cut_through_closers(remaining, cut)
        chunk = remaining[:cut].strip()
        chunks.append(chunk)
        remaining = remaining[cut:].strip()

    if remaining:
        chunks.append(remaining)
    return chunks


def _last_soft_break(text: str, max_chars: int) -> int | None:
    limit = min(max_chars, len(text))
    for index in range(limit - 1, -1, -1):
        if text[index] in SOFT_BREAKS:
            return index + 1
    return None


def _extend_cut_through_closers(text: str, cut: int) -> int:
    if text[cut - 1] not in SENTENCE_ENDS:
        return cut

    while cut < len(text) and text[cut] in TRAILING_CLOSERS:
        cut += 1
    return cut


def _normalize_tts_char(char: str) -> str:
    codepoint = ord(char)
    if (
        char == IDEOGRAPHIC_SPACE
        or FULLWIDTH_DIGIT_START <= codepoint <= FULLWIDTH_DIGIT_END
        or FULLWIDTH_UPPER_START <= codepoint <= FULLWIDTH_UPPER_END
        or FULLWIDTH_LOWER_START <= codepoint <= FULLWIDTH_LOWER_END
    ):
        return unicodedata.normalize("NFKC", char)
    return char
