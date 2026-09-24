from datetime import UTC, datetime
from unittest.mock import AsyncMock, Mock

import pytest
from config import Settings
from gowa_sdk.webhooks import WebhookEnvelope
from handler import MessageHandler
from models import Group, GroupMemberPermission, Message
from services.group_control import (
    GroupCommand,
    group_is_selected,
    group_state,
    parse_group_command,
)
from test_utils.mock_session import AsyncSessionMock
from whatsapp.jid import JID


@pytest.fixture
def mock_whatsapp():
    client = AsyncMock()
    client.get_my_jid = AsyncMock(return_value=JID(user="bot", server="s.whatsapp.net"))
    client.is_group_admin = AsyncMock(return_value=False)
    return client


@pytest.fixture
def mock_embedding_client():
    return AsyncMock()


def _payload(message_id: str, body: str, sender: str = "admin@s.whatsapp.net"):
    return WebhookEnvelope.model_validate(
        {
            "event": "message",
            "payload": {
                "id": message_id,
                "chat_id": "g@g.us",
                "from": sender,
                "timestamp": datetime.now(UTC),
                "body": body,
            },
        }
    )


def test_group_command_aliases_are_deterministic():
    assert parse_group_command("/pause").command == GroupCommand.pause
    assert parse_group_command("/stop").command == GroupCommand.pause
    assert parse_group_command("/resume").command == GroupCommand.resume
    assert parse_group_command("/start").command == GroupCommand.resume
    assert parse_group_command("nexus deny 123@s.whatsapp.net").argument == (
        "123@s.whatsapp.net"
    )
    assert parse_group_command("normal question") is None


def test_group_state_distinguishes_selection_and_pause():
    group = Group(group_jid="g@g.us")
    assert group_state(group) == "UNSELECTED"
    group.selected = True
    assert group_is_selected(group)
    assert group_state(group) == "SELECTED"
    group.managed = True
    assert group_state(group) == "ACTIVE"
    group.paused = True
    assert group_state(group) == "PAUSED"


@pytest.mark.asyncio
async def test_admin_pause_resume_is_persistent_and_does_not_use_llm(
    mock_whatsapp: AsyncMock,
    mock_embedding_client: AsyncMock,
):
    session = AsyncSessionMock()
    settings = Mock(
        spec=Settings,
        active_groups=[],
        auto_reply_groups=[],
        ai_enabled=True,
    )
    group = Group(group_jid="g@g.us", selected=True, managed=True)
    session._storage[("Group", "g@g.us")] = group
    mock_whatsapp.is_group_admin = AsyncMock(return_value=True)
    handler = MessageHandler(session, mock_whatsapp, mock_embedding_client, settings)
    handler.send_message = AsyncMock()
    handler.store_message = AsyncMock()

    await handler(_payload("pause-1", "/pause"))
    assert group.paused is True
    assert group.paused_by == "admin@s.whatsapp.net"
    handler.send_message.assert_awaited_once()
    assert "paused" in handler.send_message.await_args.args[1].casefold()
    handler.store_message.assert_not_awaited()

    handler.send_message.reset_mock()
    await handler(_payload("pause-2", "/stop"))
    assert group.paused is True
    handler.send_message.assert_awaited_once()

    handler.send_message.reset_mock()
    await handler(_payload("resume-1", "/resume"))
    assert group.paused is False
    assert group.resumed_at is not None
    assert "resumed" in handler.send_message.await_args.args[1].casefold()

    handler.send_message.reset_mock()
    await handler(_payload("resume-2", "/start"))
    assert group.paused is False
    handler.send_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_unauthorized_pause_is_denied_before_state_change(
    mock_whatsapp: AsyncMock,
    mock_embedding_client: AsyncMock,
):
    session = AsyncSessionMock()
    settings = Mock(
        spec=Settings, active_groups=[], auto_reply_groups=[], ai_enabled=True
    )
    group = Group(group_jid="g@g.us", selected=True, managed=True)
    session._storage[("Group", "g@g.us")] = group
    mock_whatsapp.is_group_admin = AsyncMock(return_value=False)
    handler = MessageHandler(session, mock_whatsapp, mock_embedding_client, settings)
    handler.send_message = AsyncMock()

    await handler(_payload("pause-unauthorized", "/pause", "member@s.whatsapp.net"))

    assert group.paused is False
    assert (
        "Only a WhatsApp group administrator" in handler.send_message.await_args.args[1]
    )


@pytest.mark.asyncio
async def test_paused_group_does_not_store_or_process_ordinary_messages(
    mock_whatsapp: AsyncMock,
    mock_embedding_client: AsyncMock,
):
    session = AsyncSessionMock()
    settings = Mock(
        spec=Settings, active_groups=[], auto_reply_groups=[], ai_enabled=True
    )
    session._storage[("Group", "g@g.us")] = Group(
        group_jid="g@g.us", selected=True, managed=True, paused=True
    )
    handler = MessageHandler(session, mock_whatsapp, mock_embedding_client, settings)
    handler.store_message = AsyncMock()

    await handler(
        _payload(
            "paused-question", "@Nexus what is the deadline?", "member@s.whatsapp.net"
        )
    )

    handler.store_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_denied_member_is_blocked_but_group_admin_override_remains(
    mock_whatsapp: AsyncMock,
    mock_embedding_client: AsyncMock,
):
    session = AsyncSessionMock()
    settings = Mock(
        spec=Settings, active_groups=[], auto_reply_groups=[], ai_enabled=True
    )
    group = Group(group_jid="g@g.us", selected=True, managed=True)
    session._storage[("Group", "g@g.us")] = group
    session._storage[("GroupMemberPermission", ("g@g.us", "member@s.whatsapp.net"))] = (
        GroupMemberPermission(
            group_jid="g@g.us",
            member_jid="member@s.whatsapp.net",
            allowed=False,
            updated_by="admin@s.whatsapp.net",
        )
    )
    handler = MessageHandler(session, mock_whatsapp, mock_embedding_client, settings)
    handler.store_message = AsyncMock()
    mock_whatsapp.is_group_admin = AsyncMock(return_value=False)

    await handler(
        _payload("denied-1", "@Nexus what is the deadline?", "member@s.whatsapp.net")
    )
    handler.store_message.assert_not_awaited()

    mock_whatsapp.is_group_admin = AsyncMock(return_value=True)
    allowed_message = Message(
        message_id="admin-override-1",
        chat_jid="g@g.us",
        sender_jid="member@s.whatsapp.net",
        group_jid="g@g.us",
        text="@Nexus what is the deadline?",
        timestamp=datetime.now(UTC),
    )
    allowed_message.group = group
    handler.store_message = AsyncMock(return_value=allowed_message)
    handler.router = AsyncMock()

    await handler(
        _payload("admin-override-1", allowed_message.text, "member@s.whatsapp.net")
    )
    handler.router.assert_awaited_once_with(allowed_message)


@pytest.mark.asyncio
async def test_unselected_group_isolated_and_selected_group_is_unaffected(
    mock_whatsapp: AsyncMock,
    mock_embedding_client: AsyncMock,
):
    session = AsyncSessionMock()
    settings = Mock(
        spec=Settings, active_groups=[], auto_reply_groups=[], ai_enabled=True
    )
    session._storage[("Group", "selected@g.us")] = Group(
        group_jid="selected@g.us", selected=True, managed=True
    )
    handler = MessageHandler(session, mock_whatsapp, mock_embedding_client, settings)
    handler.store_message = AsyncMock()

    await handler(_payload("unselected", "@Nexus what is the deadline?"))
    handler.store_message.assert_not_awaited()

    # A direct unit assertion covers the persisted state boundary for another group.
    assert group_state(session._storage[("Group", "selected@g.us")]) == "ACTIVE"


@pytest.mark.asyncio
async def test_selected_group_allows_explicit_mention_without_auto_reply_mode(
    mock_whatsapp: AsyncMock,
    mock_embedding_client: AsyncMock,
):
    session = AsyncSessionMock()
    settings = Mock(
        spec=Settings, active_groups=[], auto_reply_groups=[], ai_enabled=True
    )
    group = Group(group_jid="g@g.us", selected=True, managed=False)
    session._storage[("Group", "g@g.us")] = group
    handler = MessageHandler(session, mock_whatsapp, mock_embedding_client, settings)
    handler.store_message = AsyncMock(
        return_value=Message(
            message_id="selected-mention",
            chat_jid="g@g.us",
            sender_jid="member@s.whatsapp.net",
            group_jid="g@g.us",
            text="@Nexus what is the deadline?",
            timestamp=datetime.now(UTC),
            group=group,
        )
    )
    handler.router = AsyncMock()

    await handler(
        _payload(
            "selected-mention",
            "@Nexus what is the deadline?",
            "member@s.whatsapp.net",
        )
    )

    handler.router.assert_awaited_once()


@pytest.mark.asyncio
async def test_dm_remains_active_when_a_group_is_paused(
    mock_whatsapp: AsyncMock,
    mock_embedding_client: AsyncMock,
):
    session = AsyncSessionMock()
    settings = Mock(
        spec=Settings, active_groups=[], auto_reply_groups=[], ai_enabled=True
    )
    session._storage[("Group", "g@g.us")] = Group(
        group_jid="g@g.us", selected=True, managed=True, paused=True
    )
    handler = MessageHandler(session, mock_whatsapp, mock_embedding_client, settings)
    handler.store_message = AsyncMock(
        return_value=Message(
            message_id="dm-1",
            chat_jid="user@s.whatsapp.net",
            sender_jid="user@s.whatsapp.net",
            text="hello",
            timestamp=datetime.now(UTC),
        )
    )
    handler.router = AsyncMock()
    handler._send_private_opening_once = AsyncMock(return_value=False)

    await handler(
        WebhookEnvelope.model_validate(
            {
                "event": "message",
                "payload": {
                    "id": "dm-1",
                    "chat_id": "user@s.whatsapp.net",
                    "from": "user@s.whatsapp.net",
                    "body": "hello",
                },
            }
        )
    )

    handler.store_message.assert_awaited_once()
    handler.router.assert_awaited_once()
