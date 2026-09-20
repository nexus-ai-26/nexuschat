"""Human-escalation offers and confirmations for one WhatsApp conversation."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

from models import Message, Sender
from whatsapp.jid import normalize_jid, parse_jid

if TYPE_CHECKING:
    from .base_handler import BaseHandler


logger = logging.getLogger(__name__)

ESCALATION_OFFER = (
    "I can't answer that confidently. Would you like me to flag this to an organizer?"
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
    message_id: str
    requester_name: str
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
    timestamp = message.timestamp
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    requester_name = parse_jid(message.sender_jid).user
    session = getattr(handler, "session", None)
    if session is not None:
        sender = await session.get(Sender, message.sender_jid)
        await session.commit()
        if sender is not None and sender.push_name and sender.push_name.strip():
            requester_name = sender.push_name.strip()
    _prune_pending(timestamp)
    _pending[message.chat_jid] = PendingEscalation(
        chat_jid=message.chat_jid,
        sender_jid=message.sender_jid,
        message_id=message.message_id,
        requester_name=requester_name,
        question=message.text or "",
        group_label=_group_label(message),
        timestamp=timestamp,
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
        await handler.send_message(message.chat_jid, "Okay. I won't flag it.")
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

    return await _deliver_escalation(handler, message, pending, targets)


def _organizer_label(jid: str) -> str:
    try:
        return parse_jid(jid).user
    except Exception:
        return jid.split("@", 1)[0].split(":", 1)[0]


def _received_eat(timestamp: datetime) -> str:
    received = timestamp.astimezone(ZoneInfo("Africa/Nairobi"))
    day = received.strftime("%d").lstrip("0")
    hour = received.strftime("%I").lstrip("0")
    return f"{day} {received:%b %Y}, {hour}:{received:%M %p} (EAT)"


async def _deliver_escalation(
    handler: BaseHandler,
    message: Message,
    pending: PendingEscalation,
    targets: list[str],
) -> bool:
    alert = (
        "\U0001f6a9 *Organizer follow-up needed*\n\n"
        f"*From:* {pending.requester_name} ({_organizer_label(pending.sender_jid)})\n"
        f"*Group:* {pending.group_label}\n"
        f"*Received:* {_received_eat(pending.timestamp)}\n\n"
        "*Question:*\n"
        f"{pending.question}\n\n"
        "Please reply to the requester directly."
    )
    delivered = 0
    for target in targets:
        try:
            await handler.send_message(
                normalize_jid(target),
                alert,
                sanitize=False,
            )
            delivered += 1
        except Exception as error:
            logger.warning(
                "Escalation delivery failed target=%s error=%s",
                _organizer_label(target),
                type(error).__name__,
            )

    if not delivered:
        logger.warning("Escalation requested but no organizer accepted the flag")
        await handler.send_message(
            message.chat_jid,
            "I couldn't flag it because the organizer contacts did not accept the message.",
        )
        return True

    labels = [_organizer_label(target) for target in targets]
    mentions = [normalize_jid(target) for target in targets]
    organizer_text = " and ".join(f"@{label}" for label in labels)
    await handler.send_message(
        message.chat_jid,
        f"Thank you, {pending.requester_name}. I've passed your question to "
        f"{organizer_text}, who will contact you shortly.",
        in_reply_to=pending.message_id,
        mentions=mentions,
    )
    return True
