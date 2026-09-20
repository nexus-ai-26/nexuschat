import asyncio
from typing import Annotated

from fastapi import APIRouter, Depends, Request

from api.deps import get_whatsapp
from gowa_sdk.webhooks import WebhookEnvelope
from services.webhook_processing import process_group_sync, webhook_chat_key
from services.webhook_queue import PerChatQueue
from whatsapp import WhatsAppClient

# Create router for webhook endpoints
router = APIRouter(tags=["webhook"])

MESSAGE_EVENTS = {"message", "message.reaction"}


def is_group_sync_event(event: str) -> bool:
    return event.lower().startswith("group.")


@router.post("/webhook")
async def webhook(
    payload: WebhookEnvelope,
    request: Request,
    whatsapp: Annotated[WhatsAppClient, Depends(get_whatsapp)],
) -> str:
    """
    WhatsApp webhook endpoint for receiving incoming messages.
    Returns:
        Simple "ok" response to acknowledge receipt
    """
    event = payload.event.lower()

    # Queue message work after acknowledgement. The queue owns its DB session;
    # request dependencies never remain alive across LLM or bridge calls.
    if event in MESSAGE_EVENTS:
        queue: PerChatQueue[WebhookEnvelope] = request.app.state.webhook_queue
        queue.enqueue(webhook_chat_key(payload), payload)

    # Keep GROUPS table in sync when group-related events happen
    if is_group_sync_event(event):
        asyncio.create_task(process_group_sync(request.app, payload))

    return "ok"
