from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic_ai.agent import AgentRunResult

from config import Settings
from handler.knowledge_base_answers import KnowledgeBaseAnswers
from handler.conversation_context import (
    AnswerDepth,
    ConversationKind,
    ConversationResolution,
    RetrievalMode,
)
from models import Group, KBTopic, Message
from search.hybrid_search import SearchResult
from test_utils.mock_session import AsyncSessionMock
from utils.chat_text import chat2text
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


def test_recent_context_is_chronological_and_repeated_copies_are_deduplicated():
    handler = KnowledgeBaseAnswers(
        AsyncSessionMock(), AsyncMock(), AsyncMock(), SimpleNamespace()
    )
    older = _message(
        message_id="older",
        text="Correction: the submission deadline is Friday.",
        timestamp=datetime(2026, 9, 20, tzinfo=timezone.utc),
    )
    newer_copy = _message(
        message_id="newer-copy",
        text="Correction: the submission deadline is Friday.",
        timestamp=datetime(2026, 9, 21, tzinfo=timezone.utc),
    )
    latest = _message(
        message_id="latest",
        text="The organizer proposed a Saturday extension, not yet confirmed.",
        timestamp=datetime(2026, 9, 22, tzinfo=timezone.utc),
    )

    filtered = handler._filter_auto_reply_messages(
        [latest, older, newer_copy],
        current_question="What is the deadline?",
        current_sender="questioner@s.whatsapp.net",
        group_jid="group@g.us",
        bot_jid="bot@s.whatsapp.net",
    )

    assert [message.message_id for message in filtered] == ["newer-copy", "latest"]
    assert "proposed" in (filtered[-1].text or "")


def test_recent_context_limit_is_bounded_for_non_settings_test_doubles():
    handler = KnowledgeBaseAnswers(
        AsyncSessionMock(),
        AsyncMock(),
        AsyncMock(),
        SimpleNamespace(recent_message_context_limit=500),
    )

    assert handler._recent_message_context_limit() == 100


def test_recent_context_default_is_twenty_messages():
    handler = KnowledgeBaseAnswers(
        AsyncSessionMock(), AsyncMock(), AsyncMock(), SimpleNamespace()
    )

    assert handler._recent_message_context_limit() == 20


def test_recent_context_preserves_speculation_and_correction_qualifiers():
    handler = KnowledgeBaseAnswers(
        AsyncSessionMock(), AsyncMock(), AsyncMock(), SimpleNamespace()
    )
    ordinary_speculation = _message(
        message_id="speculation",
        sender_jid="member@s.whatsapp.net",
        text="I think the deadline may be Friday, but nobody has verified this information yet.",
    )
    organizer_correction = _message(
        message_id="correction",
        sender_jid="250783188655@s.whatsapp.net",
        text="Correction: the organizer said submissions are extended to Friday.",
    )

    filtered = handler._filter_auto_reply_messages(
        [ordinary_speculation, organizer_correction],
        current_question="What is the deadline?",
        current_sender="questioner@s.whatsapp.net",
        group_jid="group@g.us",
        bot_jid="bot@s.whatsapp.net",
    )

    assert [message.message_id for message in filtered] == [
        "speculation",
        "correction",
    ]
    context = chat2text(
        filtered,
        {},
        {"250783188655@s.whatsapp.net": "Diana"},
    )
    assert "I think" in context
    assert "nobody has verified" in context
    assert "Diana: Correction" in context


@pytest.mark.asyncio
async def test_summary_path_loads_a_broader_chronological_context_window():
    session = AsyncSessionMock()
    result = _empty_result()
    result.all.return_value = [
        _message(
            message_id="summary-update",
            text="Correction: the deadline moved to Thursday.",
        )
    ]
    session.exec = AsyncMock(return_value=result)
    handler = KnowledgeBaseAnswers(session, AsyncMock(), AsyncMock(), SimpleNamespace())

    messages = await handler._recent_summary_messages(
        _message(text="Give me the whole programme journey from the last two weeks."),
        query_text="Give me the whole programme journey from the last two weeks.",
        bot_jid="bot@s.whatsapp.net",
    )

    assert [message.message_id for message in messages] == ["summary-update"]
    assert handler._summary_window_days("last two weeks") == 14


def test_auto_reply_does_not_send_sources_or_phone_numbers():
    cleaned = KnowledgeBaseAnswers._clean_auto_reply_text(
        "Answer [1]. Call +251 911 222 333.\nSources:\n[1] raw member message"
    )

    assert "Sources" not in cleaned
    assert "[1]" not in cleaned
    assert "+251" not in cleaned


def test_generated_answer_is_not_forcibly_reduced_to_three_sentences():
    cleaned = KnowledgeBaseAnswers._clean_auto_reply_text(
        "First point. Second point. Third point. Fourth point with useful detail."
    )

    assert cleaned.count(".") == 4
    assert "Fourth point" in cleaned


@pytest.mark.asyncio
async def test_rephrasing_prompt_contains_resolved_follow_up_context():
    handler = KnowledgeBaseAnswers(
        AsyncSessionMock(), AsyncMock(), AsyncMock(), SimpleNamespace()
    )
    handler.settings.model_name = "test"
    resolution = ConversationResolution(
        current_message="In details",
        resolved_query="Expand the previous chatbot improvement question.",
        kind=ConversationKind.expansion,
        retrieval_mode=RetrievalMode.conversational_followup,
        answer_depth=AnswerDepth.detailed,
        is_follow_up=True,
        previous_user_message="How can I improve my chatbot?",
        previous_assistant_message="Start with clear answers.",
    )
    with patch(
        "handler.knowledge_base_answers.run_with_provider_fallback",
        new=AsyncMock(return_value=AgentRunResult(output="chatbot improvements")),
    ) as run:
        await handler.rephrasing_agent(
            "bot",
            _message(text="In details"),
            [],
            {},
            resolution=resolution,
        )

    prompt = run.await_args.kwargs["prompt"]
    assert "How can I improve my chatbot?" in prompt
    assert "Expand the previous chatbot improvement question." in prompt


def test_file_request_matching_supports_common_document_words():
    for request in (
        "Please send the PDF",
        "Can you share the transcript?",
        "Please send the slides",
        "Share the presentation",
        "Could you resend the recording?",
    ):
        assert KnowledgeBaseAnswers._is_file_request(request)


def test_file_request_matching_does_not_claim_unsupported_implicit_files():
    assert not KnowledgeBaseAnswers._is_file_request("Can you send it?")


@pytest.mark.asyncio
async def test_live_group_member_count_is_answered_from_the_sdk_result():
    handler = KnowledgeBaseAnswers(
        AsyncSessionMock(), AsyncMock(), AsyncMock(), SimpleNamespace()
    )
    handler.whatsapp.get_my_jid = AsyncMock(
        return_value=JID(user="bot", server="s.whatsapp.net")
    )
    handler.whatsapp.get_group_member_count = AsyncMock(return_value=42)
    handler.send_message = AsyncMock()

    answered = await handler(_message(text="How many members are in the group?"))

    assert answered is True
    handler.whatsapp.get_group_member_count.assert_awaited_once_with("group@g.us")
    handler.send_message.assert_awaited_once_with(
        "group@g.us",
        "This group currently has 42 members.",
        in_reply_to="question-1",
    )


@pytest.mark.asyncio
async def test_live_group_member_count_unavailable_does_not_generate_a_number():
    handler = KnowledgeBaseAnswers(
        AsyncSessionMock(), AsyncMock(), AsyncMock(), SimpleNamespace()
    )
    handler.whatsapp.get_my_jid = AsyncMock(
        return_value=JID(user="bot", server="s.whatsapp.net")
    )
    handler.whatsapp.get_group_member_count = AsyncMock(return_value=None)
    handler.send_message = AsyncMock()
    handler.generation_agent = AsyncMock()

    answered = await handler(_message(text="What is the current group size?"))

    assert answered is False
    handler.send_message.assert_not_awaited()
    handler.generation_agent.assert_not_awaited()


@pytest.mark.asyncio
async def test_generation_prompt_separates_trusted_facts_and_recent_chat():
    handler = KnowledgeBaseAnswers(
        AsyncSessionMock(), AsyncMock(), AsyncMock(), SimpleNamespace()
    )
    handler.settings.model_name = "test"
    with patch(
        "handler.knowledge_base_answers.run_with_provider_fallback",
        new=AsyncMock(return_value=AgentRunResult(output="ok")),
    ) as run:
        await handler.generation_agent(
            "Who handles the MIT course?",
            "official course context",
            "user@s.whatsapp.net",
            [_message(text="Correction: the link changed.")],
            {},
            trusted_application_facts="Munira handles course support.",
            live_tool_results="No live tool results are available.",
        )

    prompt = run.await_args.kwargs["prompt"]
    assert "# TRUSTED_APPLICATION_FACTS" in prompt
    assert "# LIVE_TOOL_RESULTS" in prompt
    assert "# RECENT_CHAT_CONTEXT" in prompt
    assert "# CURATED_KB_CONTEXT" in prompt
    assert prompt.index("# LIVE_TOOL_RESULTS") < prompt.index(
        "# TRUSTED_APPLICATION_FACTS"
    )
    assert prompt.index("# TRUSTED_APPLICATION_FACTS") < prompt.index(
        "# CURATED_KB_CONTEXT"
    )
    assert prompt.index("# CURATED_KB_CONTEXT") < prompt.index("# RECENT_CHAT_CONTEXT")


@pytest.mark.asyncio
async def test_generation_prompt_deduplicates_recent_organizer_context():
    handler = KnowledgeBaseAnswers(
        AsyncSessionMock(), AsyncMock(), AsyncMock(), SimpleNamespace()
    )
    handler.settings.model_name = "test"
    correction = _message(
        message_id="correction",
        sender_jid="250783188655@s.whatsapp.net",
        text="Correction: the deadline is Friday.",
    )
    with patch(
        "handler.knowledge_base_answers.run_with_provider_fallback",
        new=AsyncMock(return_value=AgentRunResult(output="ok")),
    ) as run:
        await handler.generation_agent(
            "What is the deadline?",
            "official context",
            "user@s.whatsapp.net",
            [correction],
            {},
            organizer_history=[correction],
        )

    prompt = run.await_args.kwargs["prompt"]
    assert prompt.count("Correction: the deadline is Friday.") == 1


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


@pytest.mark.asyncio
async def test_quoted_document_question_uses_document_content_first():
    session = AsyncSessionMock()
    session.exec = AsyncMock(return_value=_empty_result())
    source = _message(
        message_id="document-1",
        text="[[Attached Document]] UniPods Hackathon Guidelines.pdf\n\n"
        "Submit a two-page proposal by 30 September.",
        media_url="/media/document-1.pdf",
    )
    session.get = AsyncMock(return_value=source)
    whatsapp = AsyncMock()
    whatsapp.get_my_jid = AsyncMock(
        return_value=JID(user="bot", server="s.whatsapp.net")
    )
    handler = KnowledgeBaseAnswers(
        session, whatsapp, AsyncMock(), SimpleNamespace(spec=Settings)
    )
    handler.generation_agent = AsyncMock(
        return_value=AgentRunResult(
            output="Submit a two-page proposal by 30 September."
        )
    )
    question = _message(
        message_id="question-about-document",
        reply_to_id="document-1",
        text="What do we submit, and when is it due?",
    )
    document_result = _result(0.0)
    document_result.topic.summary = "Submit a two-page proposal by 30 September."

    with (
        patch(
            "handler.knowledge_base_answers.get_opt_out_map",
            new=AsyncMock(return_value={}),
        ),
        patch(
            "handler.knowledge_base_answers.document_topics_for_message",
            new=AsyncMock(return_value=[document_result.topic]),
        ),
    ):
        handler.send_message = AsyncMock()
        answered = await handler(question)

    assert answered is True
    handler.generation_agent.assert_awaited_once()
    assert "two-page proposal" in handler.generation_agent.await_args.args[1]
    handler.send_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_distinct_group_questions_produce_distinct_answers():
    session = AsyncSessionMock()
    session.exec = AsyncMock(side_effect=[_empty_result(), _empty_result()])
    whatsapp = AsyncMock()
    whatsapp.get_my_jid = AsyncMock(
        return_value=JID(user="bot", server="s.whatsapp.net")
    )
    handler = KnowledgeBaseAnswers(
        session, whatsapp, AsyncMock(), SimpleNamespace(spec=Settings)
    )
    first = _message(message_id="admins-question", text="Who are the admins?")
    second = _message(message_id="deadline-question", text="When does it start?")
    handler.rephrasing_agent = AsyncMock(
        side_effect=[
            AgentRunResult(output="admins"),
            AgentRunResult(output="start date"),
        ]
    )
    handler.generation_agent = AsyncMock(
        side_effect=[
            AgentRunResult(
                output="1. The admins are listed in the programme materials."
            ),
            AgentRunResult(output="2. The programme starts on 1 October."),
        ]
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
        assert await handler(first) is True
        assert await handler(second) is True

    answers = [call.args[1] for call in handler.send_message.await_args_list]
    assert answers == [
        "1. The admins are listed in the programme materials.",
        "2. The programme starts on 1 October.",
    ]


@pytest.mark.asyncio
async def test_realistic_detail_follow_up_reaches_generation_with_previous_subject():
    session = AsyncSessionMock()
    first = _message(
        message_id="chatbot-question",
        text="How can I improve my chatbot?",
    )
    previous_answer = _message(
        message_id="chatbot-answer",
        sender_jid="bot@s.whatsapp.net",
        text="Start with clear answers.",
    )
    second = _message(message_id="chatbot-follow-up", text="In details")
    first_history = _empty_result()
    first_history.all.return_value = [first]
    second_history = _empty_result()
    second_history.all.return_value = [first, previous_answer, second]
    session.exec = AsyncMock(side_effect=[first_history, second_history])
    whatsapp = AsyncMock()
    whatsapp.get_my_jid = AsyncMock(
        return_value=JID(user="bot", server="s.whatsapp.net")
    )
    handler = KnowledgeBaseAnswers(
        session, whatsapp, AsyncMock(), SimpleNamespace(spec=Settings)
    )
    handler.rephrasing_agent = AsyncMock(
        side_effect=[
            AgentRunResult(output="chatbot improvement"),
            AgentRunResult(output="How can I improve my chatbot in detail?"),
        ]
    )
    handler.generation_agent = AsyncMock(
        side_effect=[
            AgentRunResult(output="Start with clear answers."),
            AgentRunResult(
                output="First point. Second point. Third point. Fourth point."
            ),
        ]
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
        assert await handler(first) is True
        assert await handler(second) is True

    second_call = handler.generation_agent.await_args_list[1]
    assert "How can I improve my chatbot?" in second_call.args[0]
    assert second_call.kwargs["conversation_resolution"].is_follow_up is True
    assert second_call.kwargs["conversation_resolution"].answer_depth == (
        AnswerDepth.detailed
    )
    assert (
        "How can I improve my chatbot in detail?"
        in search.await_args_list[1].kwargs["query"]
    )
    assert handler.send_message.await_args_list[1].args[1].count(".") == 4
