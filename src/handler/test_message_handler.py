from datetime import UTC, datetime
from unittest.mock import AsyncMock, Mock

import pytest
from config import Settings
from handler.auto_reply import auto_reply_limiter
from gowa_sdk.webhooks import WebhookEnvelope
from sqlalchemy.exc import IntegrityError
from models import DMGreeting, Group, Message
from handler import DM_OPENING, MessageHandler
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
    yield
    auto_reply_limiter.reset()


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
async def test_group_contextual_follow_up_reaches_auto_reply_path(
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

    test_message = _managed_group_message("auto-follow-up-1", "In details")
    handler.store_message = AsyncMock(return_value=test_message)

    await handler(_group_payload("auto-follow-up-1", test_message.text or ""))

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
    handler.send_message = AsyncMock()

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
async def test_private_chat_sends_durable_opening_once_then_routes_question(
    mock_session: AsyncSessionMock,
    mock_whatsapp: AsyncMock,
    mock_embedding_client: AsyncMock,
    mock_settings: Mock,
):
    handler = MessageHandler(
        mock_session, mock_whatsapp, mock_embedding_client, mock_settings
    )
    handler.router = AsyncMock()
    handler.send_message = AsyncMock()
    private_message = Message(
        message_id="private-opening-1",
        chat_jid="opening-user@s.whatsapp.net",
        sender_jid="opening-user@s.whatsapp.net",
        text="What are the deadlines?",
        timestamp=datetime.now(UTC),
    )
    handler.store_message = AsyncMock(return_value=private_message)
    mock_session.get = AsyncMock(
        side_effect=[None, DMGreeting(sender_jid=private_message.sender_jid)]
    )
    payload = WebhookEnvelope.model_validate(
        {
            "event": "message",
            "payload": {
                "id": private_message.message_id,
                "chat_id": private_message.chat_jid,
                "from": private_message.sender_jid,
                "body": private_message.text,
            },
        }
    )

    await handler(payload)
    await handler(payload)

    assert handler.send_message.await_args_list[0].args == (
        private_message.chat_jid,
        DM_OPENING,
    )
    assert handler.send_message.await_count == 1
    assert handler.router.await_count == 2


@pytest.mark.asyncio
async def test_duplicate_greeting_claim_does_not_send_second_opening(
    mock_session: AsyncSessionMock,
    mock_whatsapp: AsyncMock,
    mock_embedding_client: AsyncMock,
    mock_settings: Mock,
):
    handler = MessageHandler(
        mock_session, mock_whatsapp, mock_embedding_client, mock_settings
    )
    message = Message(
        message_id="duplicate-opening-1",
        chat_jid="duplicate-user@s.whatsapp.net",
        sender_jid="duplicate-user@s.whatsapp.net",
        text="hello",
        timestamp=datetime.now(UTC),
    )
    original = type(
        "DuplicateGreetingError",
        (Exception,),
        {"constraint_name": "dm_greeting_pkey"},
    )("duplicate")
    mock_session.get.return_value = None
    mock_session.flush.side_effect = IntegrityError("insert", {}, original)
    handler.send_message = AsyncMock()

    claimed = await handler._send_private_opening_once(message)

    assert claimed is False
    handler.send_message.assert_not_awaited()
    mock_session.commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_non_greeting_integrity_error_is_not_swallowed(
    mock_session: AsyncSessionMock,
    mock_whatsapp: AsyncMock,
    mock_embedding_client: AsyncMock,
    mock_settings: Mock,
):
    handler = MessageHandler(
        mock_session, mock_whatsapp, mock_embedding_client, mock_settings
    )
    message = Message(
        message_id="integrity-error-opening-1",
        chat_jid="integrity-user@s.whatsapp.net",
        sender_jid="integrity-user@s.whatsapp.net",
        text="hello",
        timestamp=datetime.now(UTC),
    )
    original = type("OtherConstraintError", (Exception,), {})("duplicate")
    mock_session.get.return_value = None
    mock_session.flush.side_effect = IntegrityError("insert", {}, original)

    with pytest.raises(IntegrityError):
        await handler._send_private_opening_once(message)


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
