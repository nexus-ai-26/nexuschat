"""Safety cleanup for text that is visible to WhatsApp users."""

from __future__ import annotations

import re


_SOURCE_BLOCK_RE = re.compile(r"(?is)\bsources?\s*:.*$")
_SOURCE_LABEL_RE = re.compile(r"\[\d+\]")
_PHONE_NUMBER_RE = re.compile(r"(?<!\w)\+?\d[\d ()-]{6,}\d(?!\w)")
_INTERNAL_MENTION_RE = re.compile(r"(?<!\w)@(?:\d{6,}|user_\d+)(?!\w)")
_TIMESTAMP_RE = re.compile(
    r"\b(?:20\d{2}[-/]\d{1,2}[-/]\d{1,2}(?:[ T]\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|\s*UTC)?)?"
    r"|\d{1,2}[-/]\d{1,2}[-/]20\d{2}(?:[ T]\d{1,2}:\d{2})?)\b",
    re.IGNORECASE,
)
_SENTENCE_SEPARATOR_RE = re.compile(r"(?<=[.!?])\s+")
_RUN_ON_BULLET_RE = re.compile(r"\s+-\s+(?=\S)")
_LINK_RE = re.compile(r"https?://[^\s<>]+", re.IGNORECASE)
_LINK_LABEL_RE = re.compile(r"(?i)\b(links?)\s*:\s*(?=\S)")


def _format_links_on_separate_lines(text: str) -> str:
    matches = list(_LINK_RE.finditer(text))
    if not matches:
        return text

    parts: list[str] = []
    cursor = 0
    for match in matches:
        before = text[cursor : match.start()]
        has_section_gap = bool(re.search(r"\n\s*\n", before))
        before = before.strip(" \t\r\n;")
        if before:
            before = _LINK_LABEL_RE.sub(r"\1:\n", before, count=1)
            if has_section_gap and parts:
                before = "\n\n" + before
            parts.append(before)
        parts.append(match.group(0).rstrip(";,."))
        cursor = match.end()

    raw_tail = text[cursor:]
    has_section_gap = bool(re.match(r"\s*\n\s*\n", raw_tail))
    tail = raw_tail.strip(" \t\r\n;")
    if tail:
        if has_section_gap and parts:
            tail = "\n\n" + tail
        parts.append(tail)
    return "\n".join(parts)


def clean_visible_reply(text: str, *, max_sentences: int | None = None) -> str:
    """Remove structured citations and identifier-shaped data from user replies."""

    cleaned = _SOURCE_BLOCK_RE.split(text, maxsplit=1)[0]
    cleaned = _SOURCE_LABEL_RE.sub("", cleaned)
    cleaned = _PHONE_NUMBER_RE.sub("", cleaned)
    cleaned = _INTERNAL_MENTION_RE.sub("", cleaned)
    cleaned = _TIMESTAMP_RE.sub("", cleaned)
    cleaned = cleaned.replace("**", "*")
    cleaned = _RUN_ON_BULLET_RE.sub("\n- ", cleaned)

    if max_sentences is not None:
        sentences = _SENTENCE_SEPARATOR_RE.split(cleaned)
        cleaned = " ".join(sentences[:max_sentences]).strip()

    # Keep every URL visually separate so WhatsApp never hides it in a paragraph.
    cleaned = _format_links_on_separate_lines(cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = "\n".join(line.strip() for line in cleaned.splitlines()).strip()

    return cleaned
