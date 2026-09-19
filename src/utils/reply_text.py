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


def clean_visible_reply(text: str, *, max_sentences: int | None = None) -> str:
    """Remove structured citations and identifier-shaped data from user replies."""

    cleaned = _SOURCE_BLOCK_RE.split(text, maxsplit=1)[0]
    cleaned = _SOURCE_LABEL_RE.sub("", cleaned)
    cleaned = _PHONE_NUMBER_RE.sub("", cleaned)
    cleaned = _INTERNAL_MENTION_RE.sub("", cleaned)
    cleaned = _TIMESTAMP_RE.sub("", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip()

    if max_sentences is not None:
        sentences = _SENTENCE_SEPARATOR_RE.split(cleaned)
        cleaned = " ".join(sentences[:max_sentences]).strip()

    return cleaned
