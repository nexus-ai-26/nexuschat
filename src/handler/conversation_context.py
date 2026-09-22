"""Lightweight conversational context resolution for programme questions."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from models import Message


class ConversationKind(str, Enum):
    standalone = "standalone"
    follow_up = "follow_up"
    expansion = "expansion"
    clarification = "clarification"
    continuation = "continuation"
    more_items = "more_items"
    file_action = "file_action"
    social = "social"
    ambiguous = "ambiguous"


class RetrievalMode(str, Enum):
    direct_fact = "direct_fact"
    recent_update = "recent_update"
    conversational_followup = "conversational_followup"
    summary = "summary"
    admin_organizer = "admin_organizer"
    file_request = "file_request"
    live_data = "live_data"
    general_programme = "general_programme"
    unknown = "unknown"


class AnswerDepth(str, Enum):
    concise = "concise"
    normal = "normal"
    detailed = "detailed"
    comprehensive = "comprehensive"
    summary = "summary"
    comparison = "comparison"


@dataclass(frozen=True)
class ConversationResolution:
    """The interpretation passed from routing through retrieval and generation."""

    current_message: str
    resolved_query: str
    kind: ConversationKind
    retrieval_mode: RetrievalMode
    answer_depth: AnswerDepth
    is_follow_up: bool = False
    previous_user_message: str | None = None
    previous_assistant_message: str | None = None
    context_message_ids: tuple[str, ...] = ()

    @property
    def has_context(self) -> bool:
        return bool(self.previous_user_message or self.previous_assistant_message)


_SOCIAL_RE = re.compile(
    r"^\s*(?:hi+|hello+|hey+|hola|bonjour|salut|greetings|yo|thanks?|"
    r"thank\s+you|thx|ok(?:ay)?|fine|great|nice|good)\b[\s\W]*$",
    re.IGNORECASE,
)
_EXPANSION_RE = re.compile(
    r"^\s*(?:in\s+detail(?:s)?|more\s+detail(?:s)?|tell\s+me\s+more|"
    r"tell\s+me\s+about\s+all(?:\s+then)?|give\s+me\s+more|"
    r"give\s+me\s+everything|more\s+information|more\s+info|"
    r"expand(?:\s+that)?|explain\s+more|elaborate|what\s+else|"
    r"is\s+that\s+all)\s*[.!?]*\s*$",
    re.IGNORECASE,
)
_REFERENCE_RE = re.compile(
    r"(?:\bwhat\s+about\b|\bafter\s+that\b|\b(?:that|it|this|those|these)\b|"
    r"\b(?:the\s+)?(?:first|second|third|last|next)\s+one\b|\beach\s+one\b|"
    r"\banother\b|^\s*(?:and|also)\b|\bsend\s+it\b)",
    re.IGNORECASE,
)
_SUMMARY_RE = re.compile(
    r"\b(?:recap|summary|summarize|whole\s+programme|everything\s+important|"
    r"what\s+happened|recently|last\s+(?:few\s+)?days?|last\s+two\s+weeks?|"
    r"past\s+(?:week|two\s+weeks)|this\s+week|fortnight)\b",
    re.IGNORECASE,
)
_RECENT_UPDATE_RE = re.compile(
    r"\b(?:correction|corrected|update|updated|announcement|extended|changed|"
    r"what\s+changed|deadline\s+changed)\b",
    re.IGNORECASE,
)
_ADMIN_RE = re.compile(
    r"\b(?:admin(?:istrator)?s?|organizers?|facilitator(?:s)?|who\s+handles?|"
    r"responsib(?:le|ility)|who\s+should\s+i\s+(?:contact|ask))\b",
    re.IGNORECASE,
)
_LIVE_RE = re.compile(
    r"\b(?:current(?:ly)?|now)\b.*\b(?:members?|participants?|group\s+size)\b|"
    r"\b(?:how\s+many|number\s+of|count\s+of)\b.*\b(?:members?|participants?)\b",
    re.IGNORECASE,
)
_FILE_RE = re.compile(
    r"\b(?:send|share|forward|resend|download)\b.*\b(?:file|document|pdf|"
    r"guideline\w*|recording\w*|transcript\w*|slide\w*|presentation)\b|"
    r"\b(?:file|document|pdf|guideline\w*|recording\w*|transcript\w*|"
    r"slide\w*|presentation)\b.*\b(?:send|share|forward|resend|download)\b",
    re.IGNORECASE,
)
_DETAIL_RE = re.compile(
    r"\b(?:detail(?:ed|s)?|explain|elaborate|expand|examples?|step(?:s)?|"
    r"everything|all)\b",
    re.IGNORECASE,
)
_COMPARISON_RE = re.compile(
    r"\b(?:compare|comparison|difference|differentiate|versus|vs\.?|between)\b",
    re.IGNORECASE,
)
_QUESTION_RE = re.compile(
    r"[?]|\b(?:what|when|where|how|why|who|can|does|is|are|tell|explain|"
    r"give|send|share|need|describe)\b",
    re.IGNORECASE,
)


def _message_timestamp(message: Message) -> datetime:
    timestamp = message.timestamp
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc)


def _truncate(value: str | None, limit: int = 1800) -> str:
    text = " ".join((value or "").split())
    if len(text) <= limit:
        return text
    return f"{text[: limit - 3].rstrip()}..."


def is_contextual_follow_up_candidate(text: str | None) -> bool:
    """Return whether a message needs recent turns before it can be routed."""

    if not text or _SOCIAL_RE.fullmatch(text):
        return False
    return bool(_EXPANSION_RE.fullmatch(text) or _REFERENCE_RE.search(text))


def _answer_depth(text: str, *, is_follow_up: bool) -> AnswerDepth:
    if _SUMMARY_RE.search(text):
        return AnswerDepth.summary
    if _COMPARISON_RE.search(text):
        return AnswerDepth.comparison
    if re.search(r"\b(?:everything|all|whole|comprehensive|complete)\b", text, re.I):
        return AnswerDepth.comprehensive
    if _DETAIL_RE.search(text) or is_follow_up and _EXPANSION_RE.fullmatch(text):
        return AnswerDepth.detailed
    if (
        len(text.split()) <= 12
        and _QUESTION_RE.search(text)
        and not re.search(
            r"\b(?:tell\s+me|explain|describe|how\s+(?:can|do|should)|"
            r"give\s+me)\b",
            text,
            re.I,
        )
    ):
        return AnswerDepth.concise
    return AnswerDepth.normal


def _retrieval_mode(text: str, *, is_follow_up: bool) -> RetrievalMode:
    if _LIVE_RE.search(text):
        return RetrievalMode.live_data
    if _FILE_RE.search(text):
        return RetrievalMode.file_request
    if _SUMMARY_RE.search(text):
        return RetrievalMode.summary
    if _RECENT_UPDATE_RE.search(text):
        return RetrievalMode.recent_update
    if _ADMIN_RE.search(text):
        return RetrievalMode.admin_organizer
    if is_follow_up:
        return RetrievalMode.conversational_followup
    if re.search(
        r"\b(?:what(?:'s|\s+is)|when(?:\s+is|\s+does)?|where(?:\s+is|\s+are)?|"
        r"who(?:\s+is|\s+are)?)\b",
        text,
        re.I,
    ):
        return RetrievalMode.direct_fact
    if _QUESTION_RE.search(text):
        return RetrievalMode.general_programme
    return RetrievalMode.unknown


def _previous_turns(
    history: Sequence[Message],
    *,
    current_message: str,
    current_message_id: str | None,
    current_sender: str,
    bot_jid: str,
) -> tuple[Message | None, Message | None]:
    ordered = sorted(history, key=_message_timestamp)
    prior = [
        item
        for item in ordered
        if item.text
        and item.message_id != current_message_id
        and item.text.strip() != current_message.strip()
    ]
    if not prior:
        return None, None

    normalized_bot = bot_jid.casefold()
    assistant = next(
        (
            item
            for item in reversed(prior)
            if item.sender_jid.casefold() == normalized_bot
        ),
        None,
    )
    same_sender = [item for item in prior if item.sender_jid == current_sender]
    user = (same_sender or prior)[-1]
    if assistant is not None:
        before_assistant = [
            item
            for item in prior
            if _message_timestamp(item) <= _message_timestamp(assistant)
            and item.sender_jid.casefold() != normalized_bot
        ]
        if before_assistant:
            user = before_assistant[-1]
    return user, assistant


def _resolved_query(
    current: str,
    *,
    kind: ConversationKind,
    previous_user: str | None,
    previous_assistant: str | None,
    depth: AnswerDepth,
) -> str:
    if kind == ConversationKind.standalone:
        return current

    previous_question = _truncate(previous_user)
    previous_answer = _truncate(previous_assistant)
    context = (
        f"Previous user request: {previous_question or 'not available'}. "
        f"Previous Nexus answer: {previous_answer or 'not available'}."
    )
    if kind == ConversationKind.expansion:
        return (
            f"{context} The user now asks: {current!r}. "
            f"Provide a {depth.value} expansion of the previous subject, using only "
            "supported programme evidence."
        )
    if kind == ConversationKind.more_items:
        return (
            f"{context} The user now asks: {current!r}. "
            "Identify the additional items or remaining parts requested, grounded in "
            "the available programme evidence."
        )
    if kind == ConversationKind.file_action:
        return (
            f"{context} Resolve the referenced file or resource for this request: "
            f"{current!r}."
        )
    return (
        f"{context} Answer the contextual follow-up: {current!r}. "
        "Resolve its reference to the previous subject before retrieving evidence."
    )


def resolve_conversation_context(
    current_text: str,
    history: Sequence[Message],
    *,
    current_sender: str,
    bot_jid: str,
    current_message_id: str | None = None,
) -> ConversationResolution:
    """Resolve a short follow-up without treating every short message as one."""

    current = current_text.strip()
    previous_user, previous_assistant = _previous_turns(
        history,
        current_message=current,
        current_message_id=current_message_id,
        current_sender=current_sender,
        bot_jid=bot_jid,
    )
    candidate = is_contextual_follow_up_candidate(current)
    has_context = previous_user is not None or previous_assistant is not None
    is_follow_up = candidate and has_context

    if not is_follow_up:
        kind = (
            ConversationKind.social
            if _SOCIAL_RE.fullmatch(current)
            else ConversationKind.standalone
        )
        return ConversationResolution(
            current_message=current,
            resolved_query=current,
            kind=kind,
            retrieval_mode=_retrieval_mode(current, is_follow_up=False),
            answer_depth=_answer_depth(current, is_follow_up=False),
        )

    if _FILE_RE.search(current):
        kind = ConversationKind.file_action
    elif _EXPANSION_RE.fullmatch(current):
        kind = ConversationKind.expansion
    elif re.search(
        r"\b(?:each\s+one|more|what\s+else|another|everything|all)\b", current, re.I
    ):
        kind = ConversationKind.more_items
    elif re.search(
        r"\b(?:that|it|this|those|these|first|second|third|last|next|after)\b",
        current,
        re.I,
    ):
        kind = ConversationKind.clarification
    else:
        kind = ConversationKind.continuation

    resolved = _resolved_query(
        current,
        kind=kind,
        previous_user=previous_user.text if previous_user else None,
        previous_assistant=previous_assistant.text if previous_assistant else None,
        depth=_answer_depth(current, is_follow_up=True),
    )
    context_ids = tuple(
        item.message_id
        for item in (previous_user, previous_assistant)
        if item is not None
    )
    return ConversationResolution(
        current_message=current,
        resolved_query=resolved,
        kind=kind,
        retrieval_mode=_retrieval_mode(resolved, is_follow_up=True),
        answer_depth=_answer_depth(current, is_follow_up=True),
        is_follow_up=True,
        previous_user_message=previous_user.text if previous_user else None,
        previous_assistant_message=(
            previous_assistant.text if previous_assistant else None
        ),
        context_message_ids=context_ids,
    )


def format_conversation_context(resolution: ConversationResolution) -> str:
    """Format interpretation context separately from factual evidence."""

    if not resolution.is_follow_up:
        return "No contextual follow-up was detected."
    return (
        f"Conversation kind: {resolution.kind.value}\n"
        f"Current user message: {resolution.current_message}\n"
        f"Previous user request: {resolution.previous_user_message or 'not available'}\n"
        f"Previous Nexus answer: {resolution.previous_assistant_message or 'not available'}\n"
        f"Resolved retrieval query: {resolution.resolved_query}\n"
        f"Answer depth: {resolution.answer_depth.value}"
    )
