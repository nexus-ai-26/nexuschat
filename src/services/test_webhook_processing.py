import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import services.webhook_processing as processing
from gowa_sdk.webhooks import WebhookEnvelope
from sqlalchemy.exc import DBAPIError


def test_failure_reply_cooldown_is_not_part_of_silent_processing():
    settings = SimpleNamespace()
    assert not hasattr(processing, "_failure_reply_sent_at")
    assert not hasattr(processing, "_send_failure_reply")
    assert not hasattr(settings, "failure_reply_cooldown_seconds")


@pytest.mark.asyncio
async def test_database_failure_logs_safe_metadata_only(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    payload = WebhookEnvelope.model_validate(
        {
            "event": "message",
            "payload": {
                "id": "db-error-message",
                "chat_id": "user@s.whatsapp.net",
                "from": "user@s.whatsapp.net",
                "body": "hello",
            },
        }
    )

    class FakeDriverError(Exception):
        sqlstate = "23505"
        constraint_name = "dm_greeting_pkey"

    error = DBAPIError(
        "INSERT with private user content",
        {"message": "private body"},
        FakeDriverError("driver failure"),
    )
    app = SimpleNamespace(
        state=SimpleNamespace(
            settings=SimpleNamespace(reply_deadline_seconds=1),
            agent_semaphore=asyncio.Semaphore(1),
            async_session=object(),
            whatsapp=object(),
            embedding_client=object(),
        )
    )
    monkeypatch.setattr(processing, "_handle_payload", AsyncMock(side_effect=error))

    with caplog.at_level(logging.ERROR):
        result = await processing.process_webhook_message(app, payload)

    assert result is False
    record = caplog.records[-1]
    assert record.message == (
        "Webhook processing failed event=message reason=db_error "
        "action=no_failure_notice error_class=DBAPIError "
        "orig_class=FakeDriverError sqlstate=23505 "
        "constraint=dm_greeting_pkey"
    )
    assert "private" not in record.message
    assert "driver failure" not in record.message


@pytest.mark.asyncio
async def test_payload_failure_rolls_back_session_before_propagating(
    monkeypatch: pytest.MonkeyPatch,
):
    class Session:
        def __init__(self):
            self.rollback = AsyncMock()

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_value, traceback):
            return False

    session = Session()

    class SessionFactory:
        def __call__(self):
            return session

    class FailingHandler:
        def __init__(self, *args):
            pass

        async def __call__(self, payload):
            raise RuntimeError("handler failure")

    monkeypatch.setattr(processing, "MessageHandler", FailingHandler)
    payload = WebhookEnvelope.model_validate(
        {
            "event": "message",
            "payload": {
                "id": "rollback-message",
                "chat_id": "user@s.whatsapp.net",
                "from": "user@s.whatsapp.net",
                "body": "hello",
            },
        }
    )

    with pytest.raises(RuntimeError, match="handler failure"):
        await processing._handle_payload(
            payload,
            settings=SimpleNamespace(),
            async_session=SessionFactory(),
            whatsapp=object(),
            embedding_client=object(),
        )

    session.rollback.assert_awaited_once()
