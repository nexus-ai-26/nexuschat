"""Human-escalation offers and confirmations for one WhatsApp conversation."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

from models import Message

if TYPE_CHECKING:
    from .base_handler import BaseHandler


logger = logging.getLogger(__name__)

ESCALATION_OFFER = (
    "I can't answer that confidently — want me to flag this to an organizer?"
)
_PENDING_TTL = timedelta(minutes=15)
_pending: dict[str, "PendingEscalation"] = {}
_ESCALATION_STATUS_RE = re.compile(
    r"(?:\b(?:have|has|did|was|is|are|do|does)\b.*\b(?:escalat\w*|flag\w*)\b)"
    r"|(?:\b(?:escalat\w*|flag\w*)\b.*\b(?:sent|gone|through|done|status|happen|receive)\b)"
    r"|\b(?:did that get sent|was this flagged|did this get flagged)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PendingEscalation:
    chat_jid: str
    sender_jid: str
    question: str
    group_label: str
    timestamp: datetime


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


def is_escalation_confirmation(text: str | None) -> bool:
    if not text:
        return False
    normalized = re.sub(r"[^a-z0-9 ]", " ", text.casefold())
    normalized = " ".join(normalized.split())
    return any(
        re.search(pattern, normalized)
        for pattern in (
            r"\byes\b",
            r"\byeah\b",
            r"\byep\b",
            r"\bsure\b",
            r"\bok(?:ay)?\b",
            r"\bplease\b",
            r"\bgo ahead\b",
            r"\bdo it\b",
            r"\bflag it\b",
        )
    )


def is_escalation_decline(text: str | None) -> bool:
    if not text:
        return False
    normalized = re.sub(r"[^a-z0-9 ]", " ", text.casefold())
    normalized = " ".join(normalized.split())
    return any(
        re.search(pattern, normalized)
        for pattern in (
            r"\bno\b",
            r"\bnah\b",
            r"\bnot now\b",
            r"\bcancel\b",
            r"\bdon t\b",
            r"\bdont\b",
        )
    )


def is_escalation_status_question(text: str | None) -> bool:
    return bool(text and _ESCALATION_STATUS_RE.search(text))


def _group_label(message: Message) -> str:
    if message.group and message.group.group_name:
        return message.group.group_name
    return message.group_jid or "direct message"


def _prune_pending(now: datetime) -> None:
    expired = [
        chat_jid
        for chat_jid, pending in _pending.items()
        if now - pending.timestamp > _PENDING_TTL
    ]
    for chat_jid in expired:
        _pending.pop(chat_jid, None)


async def offer_escalation(handler: BaseHandler, message: Message) -> None:
    now = datetime.now(timezone.utc)
    _prune_pending(now)
    _pending[message.chat_jid] = PendingEscalation(
        chat_jid=message.chat_jid,
        sender_jid=message.sender_jid,
        question=message.text or "",
        group_label=_group_label(message),
        timestamp=now,
    )
    await handler.send_message(message.chat_jid, ESCALATION_OFFER)


def reset_pending_escalations() -> None:
    _pending.clear()


def pending_status_message(message: Message) -> str:
    _prune_pending(datetime.now(timezone.utc))
    pending = _pending.get(message.chat_jid)
    if pending is not None and pending.sender_jid == message.sender_jid:
        return "The organizer flag is still waiting for your confirmation; it has not been sent yet."
    return "There is no open escalation for you right now."


async def handle_pending_confirmation(handler: BaseHandler, message: Message) -> bool:
    now = datetime.now(timezone.utc)
    _prune_pending(now)
    pending = _pending.get(message.chat_jid)
    if pending is None or pending.sender_jid != message.sender_jid:
        return False

    if is_escalation_decline(message.text):
        _pending.pop(message.chat_jid, None)
        await handler.send_message(message.chat_jid, "Okay — I won't flag it.")
        return True
    if not is_escalation_confirmation(message.text):
        await handler.send_message(message.chat_jid, ESCALATION_OFFER)
        return True

    _pending.pop(message.chat_jid, None)
    settings = handler.settings
    targets = list(getattr(settings, "escalation_primary_jids", []) or [])
    targets.extend(getattr(settings, "escalation_secondary_jids", []) or [])
    targets = list(
        dict.fromkeys(str(target).strip() for target in targets if str(target).strip())
    )
    if not targets:
        logger.warning(
            "Escalation requested but ESCALATION_PRIMARY_JIDS and "
            "ESCALATION_SECONDARY_JIDS are unset"
        )
        await handler.send_message(
            message.chat_jid,
            "I couldn't flag it because organizer contacts are not configured yet.",
        )
        return True

    alert = (
        "🚩 Organizer flag\n"
        f"Question: {pending.question}\n"
        f"Group: {pending.group_label}\n"
        f"Timestamp: {pending.timestamp.isoformat()}"
    )
    delivered = 0
    for target in targets:
        try:
            await handler.send_message(target, alert, sanitize=False)
            delivered += 1
        except Exception:
            logger.exception("Escalation delivery failed for configured target")

    if delivered:
        await handler.send_message(
            message.chat_jid, "It has been flagged to an organizer."
        )
    else:
        logger.warning(
            "Escalation requested but no configured target accepted the flag"
        )
        await handler.send_message(
            message.chat_jid,
            "I couldn't flag it because the organizer contacts did not accept the message.",
        )
    return True
