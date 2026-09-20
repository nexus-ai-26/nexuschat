from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic_ai.agent import AgentRunResult

from config import Settings
from handler.knowledge_base_answers import KnowledgeBaseAnswers
from models import Group, KBTopic, Message
from search.hybrid_search import SearchResult
from test_utils.mock_session import AsyncSessionMock
from whatsapp.jid import JID


def _message(**overrides) -> Message:
    values = dict(
        message_id="question-1",
        chat_jid="group@g.us",
        sender_jid="user@s.whatsapp.net",
        group_jid="group@g.us",
        text="What is the schedule for the hackathon?",
        timestamp=datetime.now(timezone.utc),
    )
    values.update(overrides)
    message = Message(**values)
    message.group = Group(group_jid="group@g.us", managed=True)
    return message


def _result(distance: float) -> SearchResult:
    topic = KBTopic(
        id="topic-1",
        group_jid="group@g.us",
        subject="Hackathon schedule",
        summary="The schedule is in the group announcement.",
        speakers="faq",
        embedding=[0.1] * 1024,
    )
    return SearchResult(topic=topic, messages=[], vector_distance=distance)


def _empty_result():
    result = MagicMock()
    result.all.return_value = []
    return result


@pytest.mark.asyncio
async def test_auto_reply_loose_match_retrieves_recent_context_and_answers():
    session = AsyncSessionMock()
    session.exec = AsyncMock(side_effect=[_empty_result(), _empty_result()])
    whatsapp = AsyncMock()
    whatsapp.get_my_jid = AsyncMock(
        return_value=JID(user="bot", server="s.whatsapp.net")
    )
    settings = SimpleNamespace(spec=Settings)
    handler = KnowledgeBaseAnswers(session, whatsapp, AsyncMock(), settings)
    handler.rephrasing_agent = AsyncMock(return_value=AgentRunResult(output="schedule"))
    handler.generation_agent = AsyncMock(
        return_value=AgentRunResult(
            output="The schedule is in the announcement.\nSources:\n[1] 5555555555"
        )
    )

    with (
        patch(
            "handler.knowledge_base_answers.get_opt_out_map",
            new=AsyncMock(return_value={}),
        ),
        patch(
            "handler.knowledge_base_answers.voyage_embed_text",
            new=AsyncMock(return_value=[[0.1] * 1024]),
        ),
        patch(
            "search.hybrid_search.hybrid_search",
            new=AsyncMock(side_effect=[[_result(0.8)], [_result(0.55)]]),
        ) as search,
    ):
        handler.send_message = AsyncMock()
        answered = await handler(_message(), auto_reply=True)

    assert answered is True
    assert search.await_count == 2
    assert search.await_args_list[1].kwargs["max_vector_distance"] == 0.6
    handler.generation_agent.assert_awaited_once()
    assert handler.generation_agent.await_args.kwargs["weak_match"] is True
    handler.send_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_file_request_forwards_known_attachment_reference():
    handler = KnowledgeBaseAnswers(
        AsyncSessionMock(), AsyncMock(), AsyncMock(), SimpleNamespace(spec=Settings)
    )
    source_message = _message(
        text="[[Attached Document]] hackathon-guidelines.pdf",
        media_url="statics/media/hackathon-guidelines.pdf",
    )
    result = _result(0.1)
    result.messages = [source_message]
    handler._download_file_reference = AsyncMock(return_value=b"pdf-bytes")
    handler.send_file = AsyncMock()

    forwarded = await handler._try_forward_file(
        _message(text="Please send the guidelines PDF"), [result]
    )

    assert forwarded is True
    handler.send_file.assert_awaited_once_with(
        "group@g.us", b"pdf-bytes", filename="hackathon-guidelines.pdf"
    )


@pytest.mark.asyncio
async def test_auto_reply_confident_match_uses_normal_context_path():
    session = AsyncSessionMock()
    session.exec = AsyncMock(return_value=_empty_result())
    whatsapp = AsyncMock()
    whatsapp.get_my_jid = AsyncMock(
        return_value=JID(user="bot", server="s.whatsapp.net")
    )
    handler = KnowledgeBaseAnswers(
        session, whatsapp, AsyncMock(), SimpleNamespace(spec=Settings)
    )
    handler.rephrasing_agent = AsyncMock(return_value=AgentRunResult(output="schedule"))
    handler.generation_agent = AsyncMock(
        return_value=AgentRunResult(output="The schedule is in the announcement.")
    )

    with (
        patch(
            "handler.knowledge_base_answers.get_opt_out_map",
            new=AsyncMock(return_value={}),
        ),
        patch(
            "handler.knowledge_base_answers.voyage_embed_text",
            new=AsyncMock(return_value=[[0.1] * 1024]),
        ),
        patch(
            "search.hybrid_search.hybrid_search",
            new=AsyncMock(return_value=[_result(0.2)]),
        ) as search,
    ):
        handler.send_message = AsyncMock()
        answered = await handler(_message(), auto_reply=True)

    assert answered is True
    search.assert_awaited_once()
    assert "max_vector_distance" not in search.await_args.kwargs
    assert handler.generation_agent.await_args.kwargs["weak_match"] is False
    handler.send_message.assert_awaited_once()
    sent_text = handler.send_message.await_args.args[1]
    assert "Sources" not in sent_text
    assert "5555555555" not in sent_text


def test_auto_reply_context_filters_question_bot_and_test_topic_messages():
    handler = KnowledgeBaseAnswers(
        AsyncSessionMock(), AsyncMock(), AsyncMock(), SimpleNamespace()
    )
    current = _message()
    current.timestamp = datetime.now(timezone.utc)
    bot_message = _message(
        message_id="bot-message",
        sender_jid="bot@s.whatsapp.net",
        text="Bot answer",
    )
    short_question = _message(
        message_id="old-test-question",
        sender_jid="other@s.whatsapp.net",
        text="What is it?",
    )
    useful = _message(
        message_id="useful",
        sender_jid="other@s.whatsapp.net",
        text="The organizers will publish the full schedule in the announcement.",
    )

    filtered = handler._filter_auto_reply_messages(
        [current, bot_message, short_question, useful],
        current_question=current.text or "",
        current_sender=current.sender_jid,
        group_jid=current.group_jid,
        bot_jid="bot@s.whatsapp.net",
    )

    assert [message.message_id for message in filtered] == ["useful"]


def test_auto_reply_does_not_send_sources_or_phone_numbers():
    cleaned = KnowledgeBaseAnswers._clean_auto_reply_text(
        "Answer [1]. Call +251 911 222 333.\nSources:\n[1] raw member message"
    )

    assert "Sources" not in cleaned
    assert "[1]" not in cleaned
    assert "+251" not in cleaned


@pytest.mark.asyncio
async def test_mentioned_reply_does_not_append_sources_or_identifiers():
    session = AsyncSessionMock()
    session.exec = AsyncMock(return_value=_empty_result())
    whatsapp = AsyncMock()
    whatsapp.get_my_jid = AsyncMock(
        return_value=JID(user="bot", server="s.whatsapp.net")
    )
    handler = KnowledgeBaseAnswers(session, whatsapp, AsyncMock(), SimpleNamespace())
    handler.rephrasing_agent = AsyncMock(return_value=AgentRunResult(output="schedule"))
    handler.generation_agent = AsyncMock(
        return_value=AgentRunResult(
            output="*The schedule is in the announcement.* [1]\nSources:\n[1] @251911222333"
        )
    )

    with (
        patch(
            "handler.knowledge_base_answers.get_opt_out_map",
            new=AsyncMock(return_value={}),
        ),
        patch(
            "handler.knowledge_base_answers.voyage_embed_text",
            new=AsyncMock(return_value=[[0.1] * 1024]),
        ),
        patch(
            "search.hybrid_search.hybrid_search",
            new=AsyncMock(return_value=[_result(0.2)]),
        ),
    ):
        handler.send_message = AsyncMock()
        answered = await handler(_message(), auto_reply=False)

    assert answered is True
    sent_text = handler.send_message.await_args.args[1]
    assert "Sources" not in sent_text
    assert "[1]" not in sent_text
    assert "251911222333" not in sent_text
