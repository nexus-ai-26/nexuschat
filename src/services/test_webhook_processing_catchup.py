import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from gowa_sdk.webhooks import WebhookEnvelope

import services.webhook_processing as processing


@pytest.mark.asyncio
async def test_catchup_processing_failure_never_sends_failure_notice(
    monkeypatch: pytest.MonkeyPatch,
):
    payload = WebhookEnvelope.model_validate(
        {
            "event": "message",
            "payload": {
                "id": "catchup-message",
                "chat_id": "group@g.us",
                "from": "user@s.whatsapp.net",
                "body": "@999 question",
            },
        }
    )
    app = SimpleNamespace(
        state=SimpleNamespace(
            settings=SimpleNamespace(reply_deadline_seconds=1),
            agent_semaphore=asyncio.Semaphore(1),
        )
    )
    monkeypatch.setattr(
        processing,
        "_handle_payload",
        AsyncMock(side_effect=RuntimeError("provider failed")),
    )
    send_failure = AsyncMock()
    monkeypatch.setattr(processing, "_send_failure_reply", send_failure)

    result = await processing.process_webhook_message(
        app, payload, send_failure_reply=False
    )

    assert result is False
    send_failure.assert_not_awaited()
