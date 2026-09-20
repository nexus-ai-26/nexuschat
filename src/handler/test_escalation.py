from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from handler.escalation import is_human_request, offer_escalation
from models import Message


def _message(text: str) -> Message:
    return Message(
        message_id="human-request",
        chat_jid="group@g.us",
        group_jid="group@g.us",
        sender_jid="user@s.whatsapp.net",
        text=text,
        timestamp=datetime.now(timezone.utc),
    )


def test_human_request_heuristic_is_conservative():
    assert is_human_request("I need a human please") is True
    assert is_human_request("What is the schedule?") is False


@pytest.mark.asyncio
async def test_human_request_is_silent_and_sends_nothing():
    handler = SimpleNamespace(send_message=AsyncMock())

    await offer_escalation(handler, _message("I need a human please"))

    handler.send_message.assert_not_awaited()
