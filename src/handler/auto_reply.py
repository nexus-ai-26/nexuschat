"""Gating for unprompted group replies: retrieval confidence + per-group rate limits."""

from __future__ import annotations

import time
from collections import deque
import re
from typing import Iterable

from models import Message
from search.hybrid_search import SearchResult

# pgvector cosine_distance is 1 - cosine_similarity for normalized embeddings
# (Voyage). 0.32 ≈ similarity 0.68. Strict on purpose: only one KB topic is
# indexed, so a looser cutoff would auto-reply to too much casual chat.
# Loosen later if real questions are being missed.
AUTO_REPLY_MAX_VECTOR_DISTANCE = 0.32

# Keep automatic replies bounded independently for each sender and group.
AUTO_REPLY_USER_COOLDOWN_SECONDS = 20
AUTO_REPLY_GROUP_WINDOW_SECONDS = 60
AUTO_REPLY_GROUP_MAX_REPLIES = 10
AUTO_REPLY_DUPLICATE_WINDOW_SECONDS = 10 * 60

AUTO_REPLY_MIN_WORDS = 3

_QUESTION_WORDS = (
    "what",
    "when",
    "where",
    "how",
    "why",
    "who",
    "can",
    "does",
    "is",
    "help",
            "please",
            "give",
            "send",
            "share",
            "forward",
            "need",
            "recap",
            "summary",
            "happened",
    "tell",
    "explain",
    "describe",
    "could",
    "would",
    "should",
    "do",
    "are",
    "quel",
    "quelle",
    "quels",
    "quelles",
    "quand",
    "comment",
    "pourquoi",
    "qui",
    "quoi",
    "pouvez",
    "peut",
    "aidez",
    "inscription",
)
_AMHARIC_QUESTION_MARKERS = ("ምን", "መቼ", "የት", "እንዴት", "ለምን", "ማን", "እባክ")
_GREETING_OR_CHITCHAT = (
    re.compile(r"^(?:hi|hello|hey)(?:[ ,!]+(?:everyone|all|there))?[!.]*$"),
    re.compile(r"^(?:good morning|good afternoon|good evening)[!.]*$"),
    re.compile(r"^how are you[?!\.]*$"),
    re.compile(r"^what(?:'s| is) up[?!\.]*$"),
    re.compile(r"^(?:thanks|thank you|okay|ok|lol)[!.]*$"),
)

NO_ANSWER_SENTINEL = "NO_ANSWER"

_PROGRAMME_REQUEST_RE = re.compile(
    r"\b(?:give|send|share|forward|need|recap|summary|summarize|happened)\b",
    re.IGNORECASE,
)


def has_confident_match(
    results: Iterable[SearchResult],
    max_distance: float = AUTO_REPLY_MAX_VECTOR_DISTANCE,
) -> bool:
    """True if at least one retrieved topic is closer than the distance cutoff."""
    return any(result.vector_distance <= max_distance for result in results)


def best_vector_distance(results: Iterable[SearchResult]) -> float | None:
    distances = [result.vector_distance for result in results]
    return min(distances) if distances else None


def thread_key(message: Message) -> str:
    """WhatsApp reply-chain id, or the message itself when it starts a thread."""
    return message.reply_to_id or message.message_id


def normalize_question(text: str | None) -> str:
    """Normalize question text for the short-lived per-group dedupe."""
    if not text:
        return ""
    without_punctuation = re.sub(r"[^\w\s]", " ", text.casefold())
    return " ".join(without_punctuation.split())


def is_too_short_for_auto_reply(text: str | None) -> bool:
    return not text or len(text.split()) < AUTO_REPLY_MIN_WORDS


def auto_reply_question_rule(text: str | None) -> str | None:
    """Return the heuristic rule that qualifies text for an automatic reply."""
    if is_too_short_for_auto_reply(text):
        return None

    normalized = " ".join(text.casefold().split())
    if any(pattern.fullmatch(normalized) for pattern in _GREETING_OR_CHITCHAT):
        return None
    if "?" in text or "؟" in text:
        return "question-mark"
    if any(re.search(rf"\b{word}\b", normalized) for word in _QUESTION_WORDS):
        return "question-word"
    if any(marker in text for marker in _AMHARIC_QUESTION_MARKERS):
        return "amharic-question-word"
    return None


def auto_reply_text_skip_reason(text: str | None) -> str:
    if is_too_short_for_auto_reply(text):
        return "fewer than 3 words"
    if auto_reply_question_rule(text) is None:
        return "not a question or help request"
    return "not eligible"


def is_clear_banter(text: str | None) -> bool:
    """Recognize only obvious banter; ambiguous text remains on the normal path."""
    if not text:
        return False
    normalized = text.casefold().strip()
    if any(
        marker in normalized
        for marker in (
            "lol",
            "lmao",
            "haha",
            "knock knock",
            "just kidding",
            "it's a joke",
            "its a joke",
            "joke",
        )
    ):
        return True
    return any(emoji in text for emoji in ("😂", "🤣", "😅"))


_BOT_META_RE = re.compile(
    r"\b(?:which|what|are there|other|another)\s+(?:\w+\s+)?(?:bots?|agents?|models?|assistants?)\b"
    r"|\b(?:bots?|agents?|models?)\b.*\b(?:running|used|available|version|power(?:ing|ed))\b",
    re.IGNORECASE,
)
_POLL_OR_VOTE_RE = re.compile(
    r"\b(?:poll|polls|vote|votes|voting|survey|ballot)\b", re.IGNORECASE
)
_TESTING_RE = re.compile(r"\b(?:test|testing|tested|tester|testers)\b", re.IGNORECASE)
_BOT_TESTING_RE = re.compile(
    r"\b(?:bot|bots|assistant|chatbot|ai)\b.*\b(?:test|testing|tested|tester|testers)\b"
    r"|\b(?:test|testing|tested|tester|testers)\b.*\b(?:bot|bots|assistant|chatbot|ai)\b",
    re.IGNORECASE,
)
_OTHER_CLAIM_RE = re.compile(
    r"\b(?:someone|somebody|they|he|she|my\s+(?:friend|team(?:mate)?))\s+"
    r"(?:said|says|claim(?:ed|s)?|mentioned|thinks?|told)\b"
    r"|\baccording\s+to\s+(?:someone|somebody|them|him|her|[a-z][\w-]*)\b",
    re.IGNORECASE,
)


def silent_message_reason(text: str | None) -> str | None:
    """Return the policy reason a message must not trigger a public reply."""
    if not text or not text.strip():
        return "no text"
    if is_clear_banter(text):
        return "banter/insult/joke"
    normalized = text.casefold().strip()
    if any(pattern.fullmatch(normalized) for pattern in _GREETING_OR_CHITCHAT):
        return "banter/insult/joke"
    if re.search(
        r"\b(?:idiot|stupid|dumb|fool|moron|clown|useless|sucks?)\b|\bshut\s+up\b",
        normalized,
    ):
        return "banter/insult/joke"
    if _BOT_META_RE.search(text):
        return "bot or model meta-question"
    if _POLL_OR_VOTE_RE.search(text):
        return "vote/poll"
    if _TESTING_RE.search(text) and not _PROGRAMME_REQUEST_RE.search(text):
        if _BOT_TESTING_RE.search(text) or "slot" not in text.casefold():
            return "testing"
    if _OTHER_CLAIM_RE.search(text):
        return "other person's claim"
    if auto_reply_question_rule(text) is None and not _PROGRAMME_REQUEST_RE.search(text):
        return "not a question"
    return None


def is_programme_request(text: str | None) -> bool:
    """Recognize direct requests that should reach the KB even without '?'."""
    return bool(text and _PROGRAMME_REQUEST_RE.search(text))


def is_no_answer(response: str) -> bool:
    stripped = response.strip()
    return stripped == NO_ANSWER_SENTINEL or stripped.startswith(
        f"{NO_ANSWER_SENTINEL}\n"
    )


class AutoReplyLimiter:
    """Process-local sender and group limits for automatic replies.

    MessageHandler is constructed per webhook, so this must live at module
    scope (see ``auto_reply_limiter`` below), not on the handler instance.
    """

    def __init__(
        self,
        cooldown_seconds: int = AUTO_REPLY_USER_COOLDOWN_SECONDS,
        group_window_seconds: int = AUTO_REPLY_GROUP_WINDOW_SECONDS,
        group_max_replies: int = AUTO_REPLY_GROUP_MAX_REPLIES,
    ):
        self.cooldown_seconds = cooldown_seconds
        self.group_window_seconds = group_window_seconds
        self.group_max_replies = group_max_replies
        self._last_sent_at: dict[str, float] = {}
        self._group_sent_at: dict[str, deque[float]] = {}
        self._last_thread: dict[str, str] = {}
        self._question_sent_at: dict[tuple[str, str], float] = {}

    def skip_reason(self, group_jid: str, message: Message) -> str | None:
        """Return why this auto-reply should be skipped, or None to proceed."""
        key = thread_key(message)
        if self._last_thread.get(group_jid) == key:
            return "same-thread back-to-back"
        last = self._last_sent_at.get(message.sender_jid)
        if last is not None and (time.monotonic() - last) < self.cooldown_seconds:
            remaining = self.cooldown_seconds - (time.monotonic() - last)
            return f"cooldown ({remaining:.0f}s remaining)"

        now = time.monotonic()
        sent_at = self._group_sent_at.setdefault(group_jid, deque())
        while sent_at and (now - sent_at[0]) >= self.group_window_seconds:
            sent_at.popleft()
        if len(sent_at) >= self.group_max_replies:
            return f"group cap ({self.group_max_replies} replies/{self.group_window_seconds}s)"

        normalized_question = normalize_question(message.text)
        question_key = (group_jid, normalized_question)
        question_sent_at = self._question_sent_at.get(question_key)
        if (
            normalized_question
            and question_sent_at is not None
            and (now - question_sent_at) < AUTO_REPLY_DUPLICATE_WINDOW_SECONDS
        ):
            return "duplicate question"
        return None

    def record(self, group_jid: str, message: Message) -> None:
        now = time.monotonic()
        self._last_sent_at[message.sender_jid] = now
        sent_at = self._group_sent_at.setdefault(group_jid, deque())
        while sent_at and (now - sent_at[0]) >= self.group_window_seconds:
            sent_at.popleft()
        sent_at.append(now)
        self._last_thread[group_jid] = thread_key(message)
        normalized_question = normalize_question(message.text)
        if normalized_question:
            self._question_sent_at[(group_jid, normalized_question)] = now

    def reset(self) -> None:
        self._last_sent_at.clear()
        self._group_sent_at.clear()
        self._last_thread.clear()
        self._question_sent_at.clear()


# Shared across webhook-scoped handler instances.
auto_reply_limiter = AutoReplyLimiter()
