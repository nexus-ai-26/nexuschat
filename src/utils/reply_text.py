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
DEFAULT_MAX_REPLY_CHARS = 5000
WHATSAPP_MAX_CHUNK_CHARS = 4000
_REPLY_LIMIT_NOTICE = (
    "\n\nI shortened this answer to fit the response limit. "
    "Ask for a specific section if you want more detail."
)


def _is_inside_code_fence(text: str) -> bool:
    return text.count("```") % 2 == 1


def _split_boundary(text: str, max_chars: int) -> int:
    """Find the largest readable split point without losing source characters."""

    if len(text) <= max_chars:
        return len(text)

    candidates: list[int] = []
    for match in re.finditer(r"\n\s*\n|\n|(?<=[.!?])\s+|\s+", text[:max_chars]):
        position = match.end()
        if not _is_inside_code_fence(text[:position]):
            candidates.append(position)
    return max(candidates, default=max_chars)


def limit_reply_text(
    text: str,
    *,
    max_chars: int = DEFAULT_MAX_REPLY_CHARS,
) -> str:
    """Keep a complete logical reply within its configured budget."""

    if len(text) <= max_chars:
        return text

    content_budget = max(1, max_chars - len(_REPLY_LIMIT_NOTICE))
    split_at = _split_boundary(text, content_budget)
    prefix = text[:split_at].rstrip()
    return f"{prefix}{_REPLY_LIMIT_NOTICE}"[:max_chars]


def chunk_reply_text(
    text: str,
    *,
    max_total_chars: int = DEFAULT_MAX_REPLY_CHARS,
    max_chunk_chars: int = WHATSAPP_MAX_CHUNK_CHARS,
) -> list[str]:
    """Limit a logical reply, then split it into readable WhatsApp chunks."""

    limited = limit_reply_text(text, max_chars=max_total_chars)
    chunks: list[str] = []
    remaining = limited
    while remaining:
        if len(remaining) <= max_chunk_chars:
            chunks.append(remaining)
            break
        split_at = _split_boundary(remaining, max_chunk_chars)
        chunks.append(remaining[:split_at])
        remaining = remaining[split_at:]
    return chunks


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


def clean_visible_reply(
    text: str,
    *,
    max_sentences: int | None = None,
    allowed_phone_numbers: tuple[str, ...] = (),
) -> str:
    """Remove structured citations and identifier-shaped data from user replies."""

    cleaned = _SOURCE_BLOCK_RE.split(text, maxsplit=1)[0]
    cleaned = _SOURCE_LABEL_RE.sub("", cleaned)
    allowed_digits = {re.sub(r"\D", "", number) for number in allowed_phone_numbers}

    def remove_untrusted_phone(match: re.Match[str]) -> str:
        candidate = re.sub(r"\D", "", match.group(0))
        return match.group(0) if candidate in allowed_digits else ""

    cleaned = _PHONE_NUMBER_RE.sub(remove_untrusted_phone, cleaned)
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
