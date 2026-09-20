"""Classify human/escalation requests without generating public replies."""

from __future__ import annotations

import logging
import re

from models import Message

logger = logging.getLogger(__name__)


def is_human_request(text: str | None) -> bool:
    if not text:
        return False
    normalized = re.sub(r"[^a-z0-9 ]", " ", text.casefold())
    normalized = " ".join(normalized.split())
    return bool(
        re.search(
            r"\b(?:need|want|speak|talk|connect|contact|reach)\b.*\b(?:human|person|organizer|admin|organiser)\b",
            normalized,
        )
        or re.search(
            r"\b(?:human|organizer|organiser|admin)\s+(?:please|help)\b", normalized
        )
    )


def reset_pending_escalations() -> None:
    """Retained as a compatibility no-op for callers clearing old state."""


async def offer_escalation(handler: object, message: Message) -> None:
    logger.info(
        "reply skipped chat=%s message=%s reason=human request",
        message.chat_jid,
        message.message_id,
    )


async def handle_pending_confirmation(handler: object, message: Message) -> bool:
    logger.info(
        "reply skipped chat=%s message=%s reason=escalation flow removed",
        message.chat_jid,
        message.message_id,
    )
    return True


def is_escalation_status_question(text: str | None) -> bool:
    return False


def pending_status_message(message: Message) -> str:
    return ""
