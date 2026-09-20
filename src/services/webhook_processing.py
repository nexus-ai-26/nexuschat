"""Background processing for webhook messages."""

from __future__ import annotations

import asyncio
import logging
from time import monotonic
from typing import Any

from gowa_sdk.webhooks import WebhookEnvelope, WebhookMessagePayload
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlmodel.ext.asyncio.session import AsyncSession
from voyageai.client_async import AsyncClient

from config import Settings
from handler import MessageHandler
from handler.base_handler import BaseHandler
from models import Message
from whatsapp import WhatsAppClient


logger = logging.getLogger(__name__)
FALLBACK_REPLY = (
    "I'm having trouble answering right now, please try again in a moment."
)
_failure_reply_sent_at: dict[str, float] = {}
_failure_reply_lock = asyncio.Lock()


async def _claim_failure_reply(chat_key: str, cooldown_seconds: float) -> bool:
    """Claim the one failure notice allowed for a chat during the cooldown."""
    now = monotonic()
    async with _failure_reply_lock:
        last_sent = _failure_reply_sent_at.get(chat_key)
        if last_sent is not None and now - last_sent < cooldown_seconds:
            return False
        _failure_reply_sent_at[chat_key] = now
        return True


def webhook_chat_key(payload: WebhookEnvelope) -> str:
    data = WebhookMessagePayload.model_validate(payload.payload)
    return data.chat_id or data.from_ or "unknown"


async def _handle_payload(
    payload: WebhookEnvelope,
    *,
    settings: Settings,
    async_session: async_sessionmaker[AsyncSession],
    whatsapp: WhatsAppClient,
    embedding_client: AsyncClient,
) -> None:
    async with async_session() as session:
        handler = MessageHandler(session, whatsapp, embedding_client, settings)
        await handler(payload)
        await session.commit()


async def _send_failure_reply(
    payload: WebhookEnvelope,
    *,
    async_session: async_sessionmaker[AsyncSession],
    whatsapp: WhatsAppClient,
    embedding_client: AsyncClient,
) -> None:
    if payload.event.lower() != "message":
        return
    try:
        message = Message.from_webhook(payload)
    except Exception:
        logger.error("Could not build failure reply target error=invalid_payload")
        return

    async with async_session() as session:
        handler = BaseHandler(session, whatsapp, embedding_client)
        try:
            await handler.send_message(
                message.chat_jid,
                FALLBACK_REPLY,
                in_reply_to=message.message_id,
            )
            await session.commit()
        except Exception as error:
            await session.rollback()
            logger.error(
                "Failure reply could not be sent chat=%s error=%s",
                message.chat_jid,
                type(error).__name__,
            )


async def process_webhook_message(
    app: Any,
    payload: WebhookEnvelope,
    *,
    send_failure_reply: bool = True,
) -> None:
    """Process one queued webhook with bounded agent concurrency and a deadline."""
    settings: Settings = app.state.settings
    try:
        async with app.state.agent_semaphore:
            await asyncio.wait_for(
                _handle_payload(
                    payload,
                    settings=settings,
                    async_session=app.state.async_session,
                    whatsapp=app.state.whatsapp,
                    embedding_client=app.state.embedding_client,
                ),
                timeout=settings.reply_deadline_seconds,
            )
    except asyncio.CancelledError:
        raise
    except Exception as error:
        reason = "deadline" if isinstance(error, asyncio.TimeoutError) else type(error).__name__
        logger.error(
            "Webhook processing failed event=%s reason=%s; sending fallback",
            payload.event,
            reason,
        )
        if not send_failure_reply:
            logger.error(
                "Failure reply suppressed reason=catchup chat=%s",
                webhook_chat_key(payload),
            )
            return
        chat_key = webhook_chat_key(payload)
        cooldown = float(
            getattr(settings, "failure_reply_cooldown_seconds", 600.0)
        )
        if not await _claim_failure_reply(chat_key, cooldown):
            logger.warning(
                "Failure reply suppressed chat=%s reason=cooldown",
                chat_key,
            )
            return
        await _send_failure_reply(
            payload,
            async_session=app.state.async_session,
            whatsapp=app.state.whatsapp,
            embedding_client=app.state.embedding_client,
        )


async def process_group_sync(app: Any, payload: WebhookEnvelope) -> None:
    """Synchronize groups outside the request and below live-reply concurrency."""
    from whatsapp.init_groups import gather_groups

    try:
        async with app.state.background_semaphore:
            async with app.state.async_session() as session:
                await gather_groups(session, app.state.whatsapp)
                await session.commit()
    except asyncio.CancelledError:
        raise
    except Exception as error:
        logger.error(
            "Group synchronization failed event=%s error=%s",
            payload.event,
            type(error).__name__,
        )
