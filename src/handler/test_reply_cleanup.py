from unittest.mock import AsyncMock

import pytest

from handler.base_handler import BaseHandler
from models import Message
from test_utils.mock_session import AsyncSessionMock
from whatsapp import SendMessageRequest
from whatsapp.jid import JID


@pytest.mark.asyncio
async def test_base_handler_sanitizes_every_visible_outbound_message():
    whatsapp = AsyncMock()
    response = AsyncMock()
    response.results.message_id = "sent-1"
    whatsapp.send_message.return_value = response
    whatsapp.get_my_jid.return_value = JID(user="bot", server="s.whatsapp.net")
    handler = BaseHandler(AsyncSessionMock(), whatsapp, AsyncMock())
    handler.store_message = AsyncMock(
        return_value=Message(
            message_id="sent-1",
            chat_jid="user@s.whatsapp.net",
            sender_jid="bot@s.whatsapp.net",
            text="clean",
        )
    )

    await handler.send_message(
        "user@s.whatsapp.net",
        "Answer [1]. Call +251 911 222 333 at 2026-09-20T10:00:00Z.\nSources:\n[1] raw",
    )

    request = whatsapp.send_message.await_args.args[0]
    assert isinstance(request, SendMessageRequest)
    assert "Sources" not in request.message
    assert "[1]" not in request.message
    assert "+251" not in request.message
    assert "2026-09-20" not in request.message
