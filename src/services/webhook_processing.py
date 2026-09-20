"""Background processing for webhook messages."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from gowa_sdk.webhooks import WebhookEnvelope, WebhookMessagePayload
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlmodel.ext.asyncio.session import AsyncSession
from voyageai.client_async import AsyncClient

from config import Settings
from handler import MessageHandler
from whatsapp import WhatsAppClient


logger = logging.getLogger(__name__)
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


async def process_webhook_message(
    app: Any,
    payload: WebhookEnvelope,
    *,
    send_failure_reply: bool = True,
) -> bool:
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
            return True
    except asyncio.CancelledError:
        raise
    except Exception as error:
        reason = "deadline" if isinstance(error, asyncio.TimeoutError) else type(error).__name__
        logger.error(
            "Webhook processing failed event=%s reason=%s action=no_failure_notice",
            payload.event,
            reason,
        )
        return False


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
