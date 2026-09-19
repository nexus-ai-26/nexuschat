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
        or re.search(r"\b(?:human|organizer|organiser|admin)\s+(?:please|help)\b", normalized)
    )


def is_escalation_confirmation(text: str | None) -> bool:
    if not text:
        return False
    normalized = re.sub(r"[^a-z0-9 ]", " ", text.casefold())
    normalized = " ".join(normalized.split())
    return normalized in {
        "yes",
        "y",
        "please",
        "please do",
        "go ahead",
        "yes please",
        "sure",
        "okay",
        "ok",
        "do it",
        "flag it",
    }


def is_escalation_decline(text: str | None) -> bool:
    if not text:
        return False
    normalized = re.sub(r"[^a-z0-9 ]", " ", text.casefold())
    normalized = " ".join(normalized.split())
    return normalized in {"no", "n", "no thanks", "not now", "cancel"}


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


async def handle_pending_confirmation(
    handler: BaseHandler, message: Message
) -> bool:
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
        return False

    _pending.pop(message.chat_jid, None)
    settings = handler.settings
    targets = list(getattr(settings, "escalation_primary_jids", []) or [])
    targets.extend(getattr(settings, "escalation_secondary_jids", []) or [])
    targets = list(dict.fromkeys(str(target).strip() for target in targets if str(target).strip()))
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
        await handler.send_message(message.chat_jid, "It has been flagged to an organizer.")
    else:
        logger.warning("Escalation requested but no configured target accepted the flag")
        await handler.send_message(
            message.chat_jid,
            "I couldn't flag it because the organizer contacts did not accept the message.",
        )
    return True
