"""Recover and answer recent WhatsApp mentions dropped during an outage."""

from __future__ import annotations

import asyncio
import logging
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

from gowa_sdk import ChatMessagesParams
from gowa_sdk.webhooks import WebhookEnvelope
from sqlmodel import select

from config import Settings
from handler.base_handler import BaseHandler
from models import Group, Message
from services.webhook_processing import process_webhook_message
from whatsapp import WhatsAppClient
from whatsapp.jid import normalize_jid


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HistoryMessage:
    message: Message
    from_me: bool = False
    is_bot: bool = False


@dataclass
class CatchupStats:
    groups: int = 0
    recovered: int = 0
    candidates: int = 0
    answered: int = 0
    already_answered: int = 0
    skipped: int = 0
    errors: int = 0
    source: str = "none"

    def as_dict(self) -> dict[str, int | str]:
        return {
            "groups": self.groups,
            "recovered": self.recovered,
            "candidates": self.candidates,
            "answered": self.answered,
            "already_answered": self.already_answered,
            "skipped": self.skipped,
            "errors": self.errors,
            "source": self.source,
        }


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _is_bot_sender(
    sender_jid: str,
    *,
    bot_jid: str,
    other_bot_jids: Iterable[str] = (),
) -> bool:
    normalized = normalize_jid(sender_jid)
    known_bots = {normalize_jid(jid) for jid in other_bot_jids if jid}
    return normalized == bot_jid or normalized in known_bots


def find_unanswered_mentions(
    messages: Iterable[Message],
    *,
    bot_jid: str,
    answered_ids: set[str],
    window_start: datetime,
    now: datetime,
    other_bot_jids: Iterable[str] = (),
) -> tuple[list[Message], int]:
    """Return eligible messages oldest-first and count already answered rows."""
    candidates: list[Message] = []
    already_answered = 0
    for message in sorted(messages, key=lambda item: _as_utc(item.timestamp)):
        timestamp = _as_utc(message.timestamp)
        if timestamp < window_start or timestamp > now:
            continue
        if not message.text or _is_bot_sender(
            message.sender_jid,
            bot_jid=bot_jid,
            other_bot_jids=other_bot_jids,
        ):
            continue
        if message.message_id in answered_ids:
            already_answered += 1
            continue
        if message.has_mentioned(bot_jid):
            candidates.append(message)
    return candidates, already_answered


def select_catchup_candidates(
    candidates: Iterable[Message], max_replies: int
) -> tuple[list[Message], int]:
    """Apply the per-run cap and return the number intentionally skipped."""
    ordered = list(candidates)
    selected = ordered[: max(0, max_replies)]
    return selected, max(0, len(ordered) - len(selected))


def _pick(record: dict[str, Any], *names: str) -> Any:
    for name in names:
        if record.get(name) is not None:
            return record[name]
    return None


def _jid_value(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        nested = _pick(value, "jid", "id", "user", "participant")
        return str(nested) if nested else None
    return str(value) if value is not None else None


def _text_value(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        nested = _pick(value, "text", "conversation", "caption", "body", "message")
        return _text_value(nested)
    return str(value) if value is not None else None


def _history_records(results: Any) -> list[dict[str, Any]]:
    if isinstance(results, list):
        return [item for item in results if isinstance(item, dict)]
    if not isinstance(results, dict):
        return []
    for key in ("messages", "data", "items", "results"):
        value = results.get(key)
        records = _history_records(value)
        if records:
            return records
    if _pick(results, "id", "message_id") and _pick(
        results, "from", "sender", "sender_jid", "participant"
    ):
        return [results]
    return []


def history_message_from_record(
    record: dict[str, Any], group_jid: str
) -> HistoryMessage | None:
    message_id = _pick(record, "id", "message_id", "key_id")
    sender_jid = _jid_value(
        _pick(record, "from", "sender", "sender_jid", "participant")
    )
    if not message_id or not sender_jid:
        return None
    timestamp = _pick(record, "timestamp", "time", "sent_at", "created_at")
    if timestamp is None:
        timestamp = datetime.now(timezone.utc)
    try:
        payload = WebhookEnvelope.model_validate(
            {
                "event": "message",
                "payload": {
                    "id": str(message_id),
                    "chat_id": group_jid,
                    "from": sender_jid,
                    "timestamp": timestamp,
                    "body": _text_value(_pick(record, "text", "message", "body")),
                    "replied_to_id": _pick(
                        record, "replied_to_id", "reply_to_id", "quoted_message_id"
                    ),
                },
            }
        )
        message = Message.from_webhook(payload)
    except Exception:
        return None
    return HistoryMessage(
        message=message,
        from_me=bool(_pick(record, "from_me", "fromMe", "is_from_me")),
        is_bot=bool(_pick(record, "is_bot", "isBot", "bot")),
    )


def _message_envelope(message: Message) -> WebhookEnvelope:
    return WebhookEnvelope.model_validate(
        {
            "event": "message",
            "payload": {
                "id": message.message_id,
                "chat_id": message.chat_jid,
                "from": message.sender_jid,
                "timestamp": message.timestamp,
                "body": message.text,
                "replied_to_id": message.reply_to_id,
            },
        }
    )


class CatchupService:
    def __init__(self, app: Any):
        self.app = app

    @property
    def settings(self) -> Settings:
        return self.app.state.settings

    async def _served_groups(self, session) -> list[str]:
        configured = set(self.settings.active_groups or [])
        configured.update(self.settings.auto_reply_groups or [])
        result = await session.exec(select(Group).where(Group.managed == True))  # noqa: E712
        configured.update(group.group_jid for group in result.all())
        return sorted(configured)

    async def _bridge_history(
        self,
        whatsapp: WhatsAppClient,
        group_jid: str,
        start: datetime,
        end: datetime,
    ) -> list[HistoryMessage]:
        params = ChatMessagesParams(
            limit=500,
            start_time=start,
            end_time=end,
            is_from_me=False,
        )
        response = await asyncio.wait_for(
            whatsapp.get_chat_messages(group_jid, params=params), timeout=30
        )
        return [
            parsed
            for record in _history_records(response.results)
            if (parsed := history_message_from_record(record, group_jid)) is not None
        ]

    async def _answered_ids(self, session, bot_jid: str, start: datetime) -> set[str]:
        result = await session.exec(
            select(Message.reply_to_id)
            .where(Message.sender_jid == bot_jid)
            .where(Message.reply_to_id != None)  # noqa: E711
            .where(Message.timestamp >= start)
        )
        return {str(row) for row in result.all() if row}

    async def run(self) -> CatchupStats:
        stats = CatchupStats()
        now = datetime.now(timezone.utc)
        start = now - timedelta(hours=self.settings.catchup_window_hours)
        whatsapp: WhatsAppClient = self.app.state.whatsapp

        try:
            bot_jid = (await whatsapp.get_my_jid()).normalize_str()
        except Exception as error:
            stats.errors += 1
            logger.error(
                "Catch-up not run reason=bridge_identity_unavailable error=%s",
                type(error).__name__,
            )
            return stats

        async with self.app.state.async_session() as session:
            groups = await self._served_groups(session)
            stats.groups = len(groups)
            answered_ids = await self._answered_ids(session, bot_jid, start)
            db_messages: list[Message] = []
            for group_jid in groups:
                result = await session.exec(
                    select(Message)
                    .where(Message.group_jid == group_jid)
                    .where(Message.timestamp >= start)
                    .where(Message.timestamp <= now)
                )
                db_messages.extend(result.all())
            await session.commit()

        bridge_messages: list[HistoryMessage] = []
        for group_jid in groups:
            try:
                history = await self._bridge_history(whatsapp, group_jid, start, now)
                bridge_messages.extend(history)
                if history:
                    stats.source = "bridge"
            except Exception as error:
                stats.errors += 1
                logger.error(
                    "Catch-up history failed group=%s error=%s",
                    group_jid,
                    type(error).__name__,
                )

        if db_messages and stats.source == "none":
            stats.source = "database"

        if bridge_messages:
            async with self.app.state.async_session() as session:
                handler = BaseHandler(
                    session, whatsapp, self.app.state.embedding_client
                )
                known_ids = {row.message_id for row in db_messages}
                for recovered in bridge_messages:
                    message = recovered.message
                    if (
                        recovered.from_me
                        or recovered.is_bot
                        or _is_bot_sender(
                            message.sender_jid,
                            bot_jid=bot_jid,
                            other_bot_jids=self.settings.other_bot_jids,
                        )
                    ):
                        continue
                    if message.message_id not in known_ids:
                        stored = await handler.store_message(message)
                        if stored:
                            db_messages.append(stored)
                            known_ids.add(message.message_id)
                            stats.recovered += 1
                            logger.info(
                                "Catch-up recovered message_id=%s source=bridge",
                                message.message_id,
                            )
                await session.commit()

        candidates, stats.already_answered = find_unanswered_mentions(
            db_messages,
            bot_jid=bot_jid,
            answered_ids=answered_ids,
            window_start=start,
            now=now,
            other_bot_jids=self.settings.other_bot_jids,
        )
        stats.candidates = len(candidates)
        max_replies = max(0, int(self.settings.catchup_max_replies))
        to_answer, stats.skipped = select_catchup_candidates(candidates, max_replies)
        logger.info(
            "Catch-up scan source=%s groups=%s candidates=%s skipped=%s",
            stats.source,
            stats.groups,
            stats.candidates,
            stats.skipped,
        )

        for index, message in enumerate(to_answer):
            async with self.app.state.async_session() as check_session:
                if message.message_id in await self._answered_ids(
                    check_session, bot_jid, start
                ):
                    stats.already_answered += 1
                    logger.info(
                        "Catch-up message_id=%s answered=false reason=already_answered",
                        message.message_id,
                    )
                    continue

            logger.info("Catch-up message_id=%s answered=false", message.message_id)
            try:
                await process_webhook_message(self.app, _message_envelope(message))
                async with self.app.state.async_session() as check_session:
                    answered_now = message.message_id in await self._answered_ids(
                        check_session, bot_jid, start
                    )
                if answered_now:
                    stats.answered += 1
                else:
                    stats.errors += 1
                logger.info(
                    "Catch-up message_id=%s answered=%s",
                    message.message_id,
                    answered_now,
                )
            except Exception as error:
                stats.errors += 1
                logger.error(
                    "Catch-up message_id=%s answered=false error=%s",
                    message.message_id,
                    type(error).__name__,
                )
            if index + 1 < len(to_answer):
                await asyncio.sleep(max(0.0, self.settings.catchup_delay_seconds))

        return stats


async def run_startup_catchup(app: Any) -> None:
    try:
        await asyncio.sleep(app.state.settings.catchup_start_delay_seconds)
        async with app.state.catchup_lock:
            async with app.state.background_semaphore:
                await CatchupService(app).run()
    except asyncio.CancelledError:
        raise
    except Exception as error:
        logger.error("Startup catch-up failed error=%s", type(error).__name__)


def valid_admin_secret(provided: str | None, configured: str | None) -> bool:
    return bool(
        configured and provided and secrets.compare_digest(provided, configured)
    )
