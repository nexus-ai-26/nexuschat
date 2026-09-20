import asyncio
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import Request

from api import webhook as webhook_api
from gowa_sdk.webhooks import WebhookEnvelope


def _request(queue: Mock) -> SimpleNamespace:
    return SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(webhook_queue=queue))
    )


@pytest.mark.asyncio
async def test_webhook_queues_message_without_waiting_for_handler():
    payload = WebhookEnvelope.model_validate(
        {
            "event": "message",
            "payload": {
                "id": "m1",
                "chat_id": "group@g.us",
                "from": "user@s.whatsapp.net",
            },
        }
    )
    queue = Mock()
    whatsapp = AsyncMock()

    result = await webhook_api.webhook(
        payload, cast(Request, _request(queue)), whatsapp
    )

    assert result == "ok"
    queue.enqueue.assert_called_once_with("group@g.us", payload)


@pytest.mark.asyncio
async def test_webhook_group_sync_is_backgrounded(
    monkeypatch: pytest.MonkeyPatch,
):
    payload = WebhookEnvelope.model_validate(
        {"event": "group.participants", "payload": {"chat_id": "group@g.us"}}
    )
    queue = Mock()
    whatsapp = AsyncMock()
    sync = AsyncMock()
    monkeypatch.setattr(webhook_api, "process_group_sync", sync)

    result = await webhook_api.webhook(
        payload, cast(Request, _request(queue)), whatsapp
    )
    await asyncio.sleep(0)

    assert result == "ok"
    sync.assert_awaited_once()
