from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest

from handler.base_handler import BaseHandler
from models import Message
from test_utils.mock_session import AsyncSessionMock
from whatsapp import SendFileRequest, SendMessageRequest
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


@pytest.mark.asyncio
async def test_base_handler_can_forward_a_file_via_gowa():
    whatsapp = AsyncMock()
    response = AsyncMock()
    response.results = object()
    whatsapp.send_file.return_value = response
    handler = BaseHandler(AsyncSessionMock(), whatsapp, AsyncMock())

    await handler.send_file("user@s.whatsapp.net", b"pdf-bytes", filename="guide.pdf")

    request, content = whatsapp.send_file.await_args.args
    assert isinstance(request, SendFileRequest)
    assert request.phone == "user@s.whatsapp.net"
    assert request.is_forwarded is True
    assert content == b"pdf-bytes"
    assert whatsapp.send_file.await_args.kwargs["filename"] == "guide.pdf"


@pytest.mark.asyncio
async def test_base_handler_retries_server_send_failures():
    whatsapp = AsyncMock()
    response = AsyncMock()
    response.results.message_id = "sent-2"
    error = RuntimeError("bridge unavailable")
    cast(Any, error).response = SimpleNamespace(status_code=500)
    whatsapp.send_message.side_effect = [error, response]
    whatsapp.get_my_jid.return_value = JID(user="bot", server="s.whatsapp.net")
    handler = BaseHandler(AsyncSessionMock(), whatsapp, AsyncMock())
    handler.store_message = AsyncMock(
        return_value=Message(
            message_id="sent-2",
            chat_jid="user@s.whatsapp.net",
            sender_jid="bot@s.whatsapp.net",
            text="clean",
        )
    )
    setattr(
        handler,
        "settings",
        SimpleNamespace(send_retry_attempts=2, send_retry_base_seconds=0),
    )

    await handler.send_message(
        "user@s.whatsapp.net",
        "answer",
        in_reply_to="question-1",
    )

    assert whatsapp.send_message.await_count == 2
    assert whatsapp.send_message.await_args.args[0].reply_message_id == "question-1"


@pytest.mark.asyncio
async def test_base_handler_chunks_long_replies_at_whatsapp_transport_limit():
    whatsapp = AsyncMock()
    response = AsyncMock()
    response.results.message_id = "sent-long"
    whatsapp.send_message.return_value = response
    whatsapp.get_my_jid.return_value = JID(user="bot", server="s.whatsapp.net")
    handler = BaseHandler(AsyncSessionMock(), whatsapp, AsyncMock())
    handler.settings = type("ReplySettings", (), {"max_reply_chars": 5000})()
    handler.store_message = AsyncMock(
        return_value=Message(
            message_id="sent-long",
            chat_jid="user@s.whatsapp.net",
            sender_jid="bot@s.whatsapp.net",
            text="clean",
        )
    )
    logical_reply = (
        ("Paragraph one with supported detail. " * 99)
        + ("Paragraph one with supported detail.")
        + "\n\n"
        + ("Paragraph two with supported detail. " * 24)
        + ("Paragraph two with supported detail.")
    )

    await handler.send_message(
        "user@s.whatsapp.net", logical_reply, in_reply_to="question-2"
    )

    requests = [call.args[0] for call in whatsapp.send_message.await_args_list]
    assert len(requests) > 1
    assert all(len(request.message) <= 4000 for request in requests)
    assert "".join(request.message for request in requests) == logical_reply
    assert requests[0].reply_message_id == "question-2"
    assert all(request.reply_message_id is None for request in requests[1:])
