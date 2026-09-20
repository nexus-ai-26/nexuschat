from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from handler.escalation import (
    ESCALATION_OFFER,
    handle_pending_confirmation,
    is_escalation_confirmation,
    is_escalation_decline,
    is_escalation_status_question,
    is_human_request,
    offer_escalation,
    pending_status_message,
    reset_pending_escalations,
    _received_eat,
)
from models import Group, Message, Sender
from test_utils.mock_session import AsyncSessionMock


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
    session = AsyncSessionMock()
    session.get = AsyncMock(
        return_value=Sender(jid="254700000000@s.whatsapp.net", push_name="Amina")
    )
    handler = SimpleNamespace(
        session=session,
        settings=SimpleNamespace(
            escalation_primary_jids=["diana@s.whatsapp.net"],
            escalation_secondary_jids=["munira@s.whatsapp.net"],
        ),
        send_message=AsyncMock(),
    )
    question = _message("What is not documented?")
    question.sender_jid = "254700000000@s.whatsapp.net"
    question.timestamp = datetime.now(timezone.utc).replace(second=0, microsecond=0)

    await offer_escalation(handler, question)
    assert handler.send_message.await_args.args[1] == ESCALATION_OFFER

    confirmation = _message("yes please")
    confirmation.sender_jid = question.sender_jid
    confirmation.timestamp = question.timestamp
    handled = await handle_pending_confirmation(handler, confirmation)

    assert handled is True
    assert handler.send_message.await_count == 4
    admin_alert = handler.send_message.await_args_list[1]
    assert "What is not documented?" in admin_alert.args[1]
    assert "Test group" in admin_alert.args[1]
    assert "*Organizer follow-up needed*" in admin_alert.args[1]
    assert "*From:* Amina (254700000000)" in admin_alert.args[1]
    assert f"*Received:* {_received_eat(question.timestamp)}" in admin_alert.args[1]
    assert "T12:34" not in admin_alert.args[1]
    assert admin_alert.kwargs["sanitize"] is False
    group_reply = handler.send_message.await_args_list[-1]
    assert group_reply.args[1] == (
        "Thank you, Amina. I've passed your question to "
        "@diana and @munira, who will contact you shortly."
    )
    assert group_reply.kwargs["in_reply_to"] == question.message_id
    assert group_reply.kwargs["mentions"] == [
        "diana@s.whatsapp.net",
        "munira@s.whatsapp.net",
    ]


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


def test_escalation_confirmation_and_decline_are_loose_whole_word_matches():
    assert is_escalation_confirmation("Yeah I do") is True
    assert is_escalation_confirmation("okay, please flag it") is True
    assert is_escalation_confirmation("yesterday's schedule") is False
    assert is_escalation_decline("No, not now") is True
    assert is_escalation_decline("don't do that") is True
    assert is_escalation_decline("notebook") is False


def test_escalation_status_questions_are_guarded():
    assert is_escalation_status_question("Have I escalated this?") is True
    assert is_escalation_status_question("Did that get sent?") is True
    assert is_escalation_status_question("Was this flagged?") is True
    assert is_escalation_status_question("What is the schedule?") is False


@pytest.mark.asyncio
async def test_pending_smalltalk_reoffers_confirmation_instead_of_fallback():
    handler = SimpleNamespace(send_message=AsyncMock())
    await offer_escalation(handler, _message("Who can help?"))

    handled = await handle_pending_confirmation(handler, _message("Hi"))

    assert handled is True
    assert handler.send_message.await_args.args[1] == ESCALATION_OFFER


@pytest.mark.asyncio
async def test_pending_status_message_never_uses_generation():
    handler = SimpleNamespace(send_message=AsyncMock())
    question = _message("Who can help?")
    await offer_escalation(handler, question)

    assert "waiting" in pending_status_message(_message("Have I escalated?"))

    reset_pending_escalations()
    assert pending_status_message(_message("Did that get sent?")) == (
        "There is no open escalation for you right now."
    )
