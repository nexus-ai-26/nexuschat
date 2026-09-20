from datetime import UTC, datetime
from unittest.mock import AsyncMock, Mock

import pytest
from config import Settings
from handler.auto_reply import auto_reply_limiter
from handler.escalation import offer_escalation, reset_pending_escalations
from gowa_sdk.webhooks import WebhookEnvelope
from models import Group, Message
from handler import MessageHandler
from test_utils.mock_session import AsyncSessionMock
from whatsapp import SendMessageRequest
from whatsapp.jid import JID


@pytest.fixture
def mock_whatsapp():
    client = AsyncMock()
    client.send_message = AsyncMock()
    client.get_my_jid = AsyncMock(return_value=JID(user="bot", server="s.whatsapp.net"))
    return client


@pytest.fixture
def mock_embedding_client():
    client = AsyncMock()
    return client


@pytest.fixture
def mock_settings():
    return Mock(
        spec=Settings,
        model_name="test-model",
        dm_autoreply_enabled=False,
        auto_reply_groups=[],
        ai_enabled=True,
    )


@pytest.fixture(autouse=True)
def reset_auto_reply_state():
    auto_reply_limiter.reset()
    reset_pending_escalations()
    yield
    auto_reply_limiter.reset()
    reset_pending_escalations()


@pytest.mark.asyncio
async def test_message_handler_dm_opt_out(
    mock_session: AsyncSessionMock,
    mock_whatsapp: AsyncMock,
    mock_embedding_client: AsyncMock,
    mock_settings: Mock,
):
    # Create handler instance
    handler = MessageHandler(
        mock_session, mock_whatsapp, mock_embedding_client, mock_settings
    )

    # Mock store_message to return our test message
    test_message = Message(
        message_id="1",
        chat_jid="user@s.whatsapp.net",
        sender_jid="user@s.whatsapp.net",  # DM: sender == chat (usually, but logic checks message.group)
        text="opt-out",
        timestamp=datetime.now(UTC),
    )
    # Ensure message.group is None for DM check
    # In the code: if message and not message.group:
    # Message model has a 'group' relationship. We can just set it to None or rely on default.
    # But wait, store_message returns a Message object.

    # We need to mock store_message because __call__ calls it.
    # However, store_message is an async method on the instance.
    handler.store_message = AsyncMock(return_value=test_message)

    # Create a dummy payload
    payload = WebhookEnvelope.model_validate(
        {
            "event": "message",
            "payload": {
                "id": "msg_opt_out",
                "chat_id": "user@s.whatsapp.net",
                "from": "user@s.whatsapp.net",
                "from_name": "User",
                "timestamp": datetime.now(UTC),
                "body": "opt-out",
            },
        }
    )

    # Fix mock response for send_message
    mock_response = AsyncMock()
    mock_response.results.message_id = "response_id"
    mock_whatsapp.send_message.return_value = mock_response

    await handler(payload)

    # Verify upsert was called (which calls execute)
    mock_session.execute.assert_called()

    # Verify confirmation message
    mock_whatsapp.send_message.assert_called_with(
        SendMessageRequest(
            phone="user@s.whatsapp.net",
            message="You have been opted out. You will no longer be tagged in summaries and answers.",
            reply_message_id=None,
        )
    )


@pytest.mark.asyncio
async def test_message_handler_dm_opt_in(
    mock_session: AsyncSessionMock,
    mock_whatsapp: AsyncMock,
    mock_embedding_client: AsyncMock,
    mock_settings: Mock,
):
    handler = MessageHandler(
        mock_session, mock_whatsapp, mock_embedding_client, mock_settings
    )

    test_message = Message(
        message_id="1",
        chat_jid="user@s.whatsapp.net",
        sender_jid="user@s.whatsapp.net",
        text="opt-in",
        timestamp=datetime.now(UTC),
    )
    handler.store_message = AsyncMock(return_value=test_message)

    payload = WebhookEnvelope.model_validate(
        {
            "event": "message",
            "payload": {
                "id": "msg_opt_in",
                "chat_id": "user@s.whatsapp.net",
                "from": "user@s.whatsapp.net",
                "from_name": "User",
                "timestamp": datetime.now(UTC),
                "body": "opt-in",
            },
        }
    )

    # Mock existing opt-out record
    from models import OptOut

    opt_out = OptOut(jid="user@s.whatsapp.net")
    mock_session._storage[("OptOut", "user@s.whatsapp.net")] = opt_out

    mock_response = AsyncMock()
    mock_response.results.message_id = "response_id"
    mock_whatsapp.send_message.return_value = mock_response

    await handler(payload)

    # Verify delete was called
    mock_session.delete.assert_called_with(opt_out)
    mock_session.commit.assert_called()

    # Verify confirmation message
    mock_whatsapp.send_message.assert_called_with(
        SendMessageRequest(
            phone="user@s.whatsapp.net",
            message="You have been opted in. You will now be tagged in summaries and answers.",
            reply_message_id=None,
        )
    )


@pytest.mark.asyncio
async def test_message_handler_dm_status(
    mock_session: AsyncSessionMock,
    mock_whatsapp: AsyncMock,
    mock_embedding_client: AsyncMock,
    mock_settings: Mock,
):
    handler = MessageHandler(
        mock_session, mock_whatsapp, mock_embedding_client, mock_settings
    )

    test_message = Message(
        message_id="1",
        chat_jid="user@s.whatsapp.net",
        sender_jid="user@s.whatsapp.net",
        text="status",
        timestamp=datetime.now(UTC),
    )
    handler.store_message = AsyncMock(return_value=test_message)

    payload = WebhookEnvelope.model_validate(
        {
            "event": "message",
            "payload": {
                "id": "msg_status",
                "chat_id": "user@s.whatsapp.net",
                "from": "user@s.whatsapp.net",
                "from_name": "User",
                "timestamp": datetime.now(UTC),
                "body": "status",
            },
        }
    )

    # Mock get to return None (opted in)
    mock_session.get.return_value = None

    mock_response = AsyncMock()
    mock_response.results.message_id = "response_id"
    mock_whatsapp.send_message.return_value = mock_response

    await handler(payload)

    # Verify status message
    mock_whatsapp.send_message.assert_called_with(
        SendMessageRequest(
            phone="user@s.whatsapp.net",
            message="You are currently opted in.",
            reply_message_id=None,
        )
    )


def _group_payload(
    message_id: str, body: str, *, from_me: bool = False
) -> WebhookEnvelope:
    return WebhookEnvelope.model_validate(
        {
            "event": "message",
            "payload": {
                "id": message_id,
                "chat_id": "g@g.us",
                "from": "user@s.whatsapp.net",
                "from_name": "User",
                "timestamp": datetime.now(UTC),
                "body": body,
                "from_me": from_me,
            },
        }
    )


def _managed_group_message(
    message_id: str, text: str, *, managed: bool = True
) -> Message:
    group = Group(group_jid="g@g.us", group_name="test", managed=managed)
    message = Message(
        message_id=message_id,
        chat_jid="g@g.us",
        sender_jid="user@s.whatsapp.net",
        group_jid="g@g.us",
        text=text,
        timestamp=datetime.now(UTC),
    )
    message.group = group
    return message


@pytest.mark.asyncio
async def test_managed_group_mention_uses_router(
    mock_session: AsyncSessionMock,
    mock_whatsapp: AsyncMock,
    mock_embedding_client: AsyncMock,
    mock_settings: Mock,
):
    handler = MessageHandler(
        mock_session, mock_whatsapp, mock_embedding_client, mock_settings
    )
    handler.router = AsyncMock()
    handler.router.ask_knowledge_base = AsyncMock()

    test_message = _managed_group_message(
        "mention-1", "@bot how do I fix MIT blank pages?"
    )
    handler.store_message = AsyncMock(return_value=test_message)

    await handler(_group_payload("mention-1", test_message.text or ""))

    handler.router.assert_awaited_once_with(test_message)
    handler.router.ask_knowledge_base.assert_not_called()


@pytest.mark.asyncio
async def test_managed_group_without_mention_uses_auto_reply(
    mock_session: AsyncSessionMock,
    mock_whatsapp: AsyncMock,
    mock_embedding_client: AsyncMock,
    mock_settings: Mock,
):
    mock_settings.auto_reply_groups = ["g@g.us"]
    handler = MessageHandler(
        mock_session, mock_whatsapp, mock_embedding_client, mock_settings
    )
    handler.router = AsyncMock()
    handler.router.ask_knowledge_base = AsyncMock(return_value=True)

    test_message = _managed_group_message(
        "auto-1", "How do I fix blank pages on the MIT platform?"
    )
    handler.store_message = AsyncMock(return_value=test_message)

    await handler(_group_payload("auto-1", test_message.text or ""))

    handler.router.assert_not_awaited()
    handler.router.ask_knowledge_base.assert_awaited_once_with(
        test_message, auto_reply=True
    )


@pytest.mark.asyncio
async def test_active_unmanaged_group_allows_mention_only_reply(
    mock_session: AsyncSessionMock,
    mock_whatsapp: AsyncMock,
    mock_embedding_client: AsyncMock,
    mock_settings: Mock,
):
    mock_settings.active_groups = ["g@g.us"]
    handler = MessageHandler(
        mock_session, mock_whatsapp, mock_embedding_client, mock_settings
    )
    handler.router = AsyncMock()
    test_message = _managed_group_message(
        "active-mention-1", "@bot What is the schedule?", managed=False
    )
    handler.store_message = AsyncMock(return_value=test_message)

    await handler(_group_payload("active-mention-1", test_message.text or ""))

    handler.router.assert_awaited_once_with(test_message)


@pytest.mark.asyncio
async def test_unmanaged_group_skips_auto_reply(
    mock_session: AsyncSessionMock,
    mock_whatsapp: AsyncMock,
    mock_embedding_client: AsyncMock,
    mock_settings: Mock,
):
    mock_settings.auto_reply_groups = ["other@g.us"]
    handler = MessageHandler(
        mock_session, mock_whatsapp, mock_embedding_client, mock_settings
    )
    handler.router = AsyncMock()
    handler.router.ask_knowledge_base = AsyncMock()

    test_message = _managed_group_message(
        "unmanaged-1",
        "How do I fix blank pages on the MIT platform?",
        managed=False,
    )
    handler.store_message = AsyncMock(return_value=test_message)

    await handler(_group_payload("unmanaged-1", test_message.text or ""))

    handler.router.assert_not_awaited()
    handler.router.ask_knowledge_base.assert_not_called()


@pytest.mark.asyncio
async def test_unlisted_group_without_mention_keeps_mention_only_behavior(
    mock_session: AsyncSessionMock,
    mock_whatsapp: AsyncMock,
    mock_embedding_client: AsyncMock,
    mock_settings: Mock,
):
    handler = MessageHandler(
        mock_session, mock_whatsapp, mock_embedding_client, mock_settings
    )
    handler.router = AsyncMock()
    handler.router.ask_knowledge_base = AsyncMock()

    test_message = _managed_group_message(
        "unlisted-1", "How do I fix blank pages on the MIT platform?"
    )
    handler.store_message = AsyncMock(return_value=test_message)

    await handler(_group_payload("unlisted-1", test_message.text or ""))

    handler.router.assert_not_awaited()
    handler.router.ask_knowledge_base.assert_not_called()


@pytest.mark.asyncio
async def test_empty_auto_reply_groups_replies_nowhere(
    mock_session: AsyncSessionMock,
    mock_whatsapp: AsyncMock,
    mock_embedding_client: AsyncMock,
    mock_settings: Mock,
):
    handler = MessageHandler(
        mock_session, mock_whatsapp, mock_embedding_client, mock_settings
    )
    handler.router = AsyncMock()
    handler.router.ask_knowledge_base = AsyncMock()

    test_message = _managed_group_message(
        "empty-1", "Please help me access the MIT platform"
    )
    handler.store_message = AsyncMock(return_value=test_message)

    await handler(_group_payload("empty-1", test_message.text or ""))

    handler.router.assert_not_awaited()
    handler.router.ask_knowledge_base.assert_not_called()


@pytest.mark.asyncio
async def test_from_me_group_message_is_ignored(
    mock_session: AsyncSessionMock,
    mock_whatsapp: AsyncMock,
    mock_embedding_client: AsyncMock,
    mock_settings: Mock,
):
    mock_settings.auto_reply_groups = ["g@g.us"]
    handler = MessageHandler(
        mock_session, mock_whatsapp, mock_embedding_client, mock_settings
    )
    handler.router = AsyncMock()
    handler.router.ask_knowledge_base = AsyncMock()

    test_message = _managed_group_message(
        "from-me-1", "Please help me access the MIT platform"
    )
    handler.store_message = AsyncMock(return_value=test_message)

    await handler(_group_payload("from-me-1", test_message.text or "", from_me=True))

    handler.router.assert_not_awaited()
    handler.router.ask_knowledge_base.assert_not_called()


@pytest.mark.asyncio
async def test_greeting_is_skipped_in_listed_group(
    mock_session: AsyncSessionMock,
    mock_whatsapp: AsyncMock,
    mock_embedding_client: AsyncMock,
    mock_settings: Mock,
):
    mock_settings.auto_reply_groups = ["g@g.us"]
    handler = MessageHandler(
        mock_session, mock_whatsapp, mock_embedding_client, mock_settings
    )
    handler.router = AsyncMock()
    handler.router.ask_knowledge_base = AsyncMock()

    test_message = _managed_group_message("greeting-1", "How are you?")
    handler.store_message = AsyncMock(return_value=test_message)

    await handler(_group_payload("greeting-1", test_message.text or ""))

    handler.router.assert_not_awaited()
    handler.router.ask_knowledge_base.assert_not_called()


@pytest.mark.asyncio
async def test_per_user_rate_limit_is_enforced(
    mock_session: AsyncSessionMock,
    mock_whatsapp: AsyncMock,
    mock_embedding_client: AsyncMock,
    mock_settings: Mock,
):
    mock_settings.auto_reply_groups = ["g@g.us"]
    handler = MessageHandler(
        mock_session, mock_whatsapp, mock_embedding_client, mock_settings
    )
    handler.router = AsyncMock()
    handler.router.ask_knowledge_base = AsyncMock(return_value=True)

    first = _managed_group_message("rate-1", "Please help me access the platform")
    second = _managed_group_message("rate-2", "Please help me access the schedule")
    handler.store_message = AsyncMock(side_effect=[first, second])

    await handler(_group_payload("rate-1", first.text or ""))
    await handler(_group_payload("rate-2", second.text or ""))

    handler.router.ask_knowledge_base.assert_awaited_once_with(first, auto_reply=True)


@pytest.mark.asyncio
async def test_private_chat_uses_the_full_router_pipeline(
    mock_session: AsyncSessionMock,
    mock_whatsapp: AsyncMock,
    mock_embedding_client: AsyncMock,
    mock_settings: Mock,
):
    mock_settings.auto_reply_groups = ["g@g.us"]
    handler = MessageHandler(
        mock_session, mock_whatsapp, mock_embedding_client, mock_settings
    )
    handler.router = AsyncMock()

    private_message = Message(
        message_id="private-1",
        chat_jid="user@s.whatsapp.net",
        sender_jid="user@s.whatsapp.net",
        text="How do I access the platform?",
        timestamp=datetime.now(UTC),
    )
    handler.store_message = AsyncMock(return_value=private_message)

    await handler(
        WebhookEnvelope.model_validate(
            {
                "event": "message",
                "payload": {
                    "id": "private-1",
                    "chat_id": "user@s.whatsapp.net",
                    "from": "user@s.whatsapp.net",
                    "timestamp": datetime.now(UTC),
                    "body": private_message.text,
                },
            }
        )
    )

    handler.router.assert_awaited_once_with(private_message)
    mock_whatsapp.send_message.assert_not_called()


@pytest.mark.asyncio
async def test_escalation_status_question_never_reaches_router(
    mock_session: AsyncSessionMock,
    mock_whatsapp: AsyncMock,
    mock_embedding_client: AsyncMock,
    mock_settings: Mock,
):
    handler = MessageHandler(
        mock_session, mock_whatsapp, mock_embedding_client, mock_settings
    )
    handler.send_message = AsyncMock()
    original = Message(
        message_id="pending-question",
        chat_jid="user@s.whatsapp.net",
        sender_jid="user@s.whatsapp.net",
        text="Who can help?",
    )
    await offer_escalation(handler, original)

    handler.router = AsyncMock()
    status_message = Message(
        message_id="pending-status",
        chat_jid="user@s.whatsapp.net",
        sender_jid="user@s.whatsapp.net",
        text="Did that get sent?",
    )
    handler.store_message = AsyncMock(return_value=status_message)
    payload = WebhookEnvelope.model_validate(
        {
            "event": "message",
            "payload": {
                "id": "pending-status",
                "chat_id": "user@s.whatsapp.net",
                "from": "user@s.whatsapp.net",
                "timestamp": datetime.now(UTC),
                "body": status_message.text,
            },
        }
    )

    await handler(payload)

    handler.router.assert_not_awaited()
    assert "waiting" in handler.send_message.await_args_list[-1].args[1]


@pytest.mark.asyncio
async def test_ai_disabled_still_stores_group_messages(
    mock_session, mock_whatsapp, mock_embedding_client, mock_settings
):
    mock_settings.ai_enabled = False
    handler = MessageHandler(
        mock_session, mock_whatsapp, mock_embedding_client, mock_settings
    )
    message = Message(
        message_id="disabled-ai-test",
        chat_jid="123@g.us",
        sender_jid="user@s.whatsapp.net",
        text="@bot summarize this",
        timestamp=datetime.now(UTC),
    )
    message.group = Group(group_jid="123@g.us", managed=True)
    handler.store_message = AsyncMock(return_value=message)
    handler.router = AsyncMock()
    handler.kb_qa_handler = AsyncMock()
    handler.whatsapp_group_link_spam = AsyncMock()
    payload = WebhookEnvelope.model_validate(
        {"event": "message", "payload": {"id": "disabled-ai-test"}}
    )

    await handler(payload)

    handler.store_message.assert_awaited_once_with(payload)
    mock_session.commit.assert_awaited()
    handler.router.assert_not_awaited()
    handler.kb_qa_handler.assert_not_awaited()
    handler.whatsapp_group_link_spam.assert_not_awaited()
    mock_whatsapp.send_message.assert_not_awaited()
