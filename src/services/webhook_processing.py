"""Background processing for webhook messages."""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

from gowa_sdk.webhooks import WebhookEnvelope, WebhookMessagePayload
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlmodel.ext.asyncio.session import AsyncSession
from voyageai.client_async import AsyncClient

from config import Settings
from handler import MessageHandler
from whatsapp import WhatsAppClient


logger = logging.getLogger(__name__)

_SAFE_DB_CODE_RE = re.compile(r"^[A-Za-z0-9_.-]{1,32}$")


def _safe_db_failure_metadata(error: BaseException) -> dict[str, str]:
    """Return non-sensitive metadata suitable for a production log line.

    DBAPI exceptions often include SQL, bound values, connection details, or
    server error text in their string representation.  Keep the diagnostic
    useful without logging any of those fields.
    """
    metadata = {"error_class": type(error).__name__}
    if not isinstance(error, DBAPIError):
        return metadata

    original = error.orig
    metadata["orig_class"] = type(original).__name__
    for key in ("sqlstate", "pgcode", "code"):
        value = getattr(original, key, None)
        if value is None:
            value = getattr(error, key, None)
        if isinstance(value, (str, int)) and _SAFE_DB_CODE_RE.fullmatch(str(value)):
            metadata["sqlstate"] = str(value)
            break

    constraint = getattr(original, "constraint_name", None)
    if isinstance(constraint, str) and _SAFE_DB_CODE_RE.fullmatch(constraint):
        metadata["constraint"] = constraint
    return metadata


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
        try:
            handler = MessageHandler(session, whatsapp, embedding_client, settings)
            await handler(payload)
            await session.commit()
        except BaseException:
            await session.rollback()
            raise


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
        reason = (
            "deadline"
            if isinstance(error, asyncio.TimeoutError)
            else type(error).__name__
        )
        if isinstance(error, DBAPIError):
            metadata = _safe_db_failure_metadata(error)
            logger.error(
                "Webhook processing failed event=%s reason=db_error "
                "action=no_failure_notice error_class=%s orig_class=%s "
                "sqlstate=%s constraint=%s",
                payload.event,
                metadata.get("error_class", "unknown"),
                metadata.get("orig_class", "unknown"),
                metadata.get("sqlstate", "unknown"),
                metadata.get("constraint", "unknown"),
            )
        else:
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
