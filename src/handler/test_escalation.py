from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from handler.escalation import (
    ESCALATION_OFFER,
    handle_pending_confirmation,
    is_human_request,
    offer_escalation,
    reset_pending_escalations,
)
from models import Group, Message


def _message(text: str) -> Message:
    message = Message(
        message_id=f"message-{text}",
        chat_jid="group@g.us",
        group_jid="group@g.us",
        sender_jid="user@s.whatsapp.net",
        text=text,
        timestamp=datetime.now(timezone.utc),
    )
    message.group = Group(group_jid="group@g.us", group_name="Test group")
    return message


@pytest.fixture(autouse=True)
def clear_pending():
    reset_pending_escalations()
    yield
    reset_pending_escalations()


@pytest.mark.asyncio
async def test_no_answer_offers_then_explicit_confirmation_notifies_admin():
    handler = SimpleNamespace(
        settings=SimpleNamespace(
            escalation_primary_jids=["diana@s.whatsapp.net"],
            escalation_secondary_jids=["munira@s.whatsapp.net"],
        ),
        send_message=AsyncMock(),
    )
    question = _message("What is not documented?")

    await offer_escalation(handler, question)
    assert handler.send_message.await_args.args[1] == ESCALATION_OFFER

    confirmation = _message("yes please")
    handled = await handle_pending_confirmation(handler, confirmation)

    assert handled is True
    assert handler.send_message.await_count == 4
    admin_alert = handler.send_message.await_args_list[1]
    assert "What is not documented?" in admin_alert.args[1]
    assert "Test group" in admin_alert.args[1]
    assert admin_alert.kwargs["sanitize"] is False
    assert handler.send_message.await_args_list[-1].args[1] == (
        "It has been flagged to an organizer."
    )


@pytest.mark.asyncio
async def test_confirmation_is_transparent_when_contacts_are_unset():
    handler = SimpleNamespace(
        settings=SimpleNamespace(
            escalation_primary_jids=[], escalation_secondary_jids=[]
        ),
        send_message=AsyncMock(),
    )
    await offer_escalation(handler, _message("Who can help?"))

    handled = await handle_pending_confirmation(handler, _message("go ahead"))

    assert handled is True
    assert "not configured" in handler.send_message.await_args_list[-1].args[1]


def test_human_request_heuristic_is_conservative():
    assert is_human_request("I need a human please") is True
    assert is_human_request("What is the schedule?") is False
