from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, Mock, MagicMock, patch

import pytest
from pydantic_ai import Agent
from pydantic_ai.agent import AgentRunResult

from handler.router import Router, IntentEnum, Intent
from handler.conversation_context import (
    AnswerDepth,
    ConversationKind,
    ConversationResolution,
    RetrievalMode,
)
from models import Message
from test_utils.mock_session import AsyncSessionMock
from whatsapp import SendMessageRequest
from whatsapp.jid import JID
from config import Settings


@pytest.fixture
def mock_whatsapp():
    client = AsyncMock()
    client.send_message = AsyncMock()
    client.get_my_jid = AsyncMock(return_value=JID(user="bot", server="s.whatsapp.net"))
    return client


@pytest.fixture
def mock_embedding_client():
    client = AsyncMock()
    # Mock the embed method to return an object with embeddings and total_tokens attributes
    mock_response = AsyncMock()
    mock_response.embeddings = [[0.1, 0.2, 0.3, 0.4, 0.5]]
    mock_response.total_tokens = 4
    client.embed = AsyncMock(return_value=mock_response)
    return client


@pytest.fixture
def test_message():
    return Message(
        message_id="test_id",
        text="Could you assist",
        chat_jid="user@s.whatsapp.net",
        sender_jid="user@s.whatsapp.net",
        timestamp=datetime.now(timezone.utc),
    )


@pytest.fixture
def mock_settings():
    return Mock(
        spec=Settings,
        model_name="anthropic:test-model",
        llm_provider_order="anthropic",
        anthropic_api_key="test-anthropic-key",
    )


def MockAgent(return_value: Any):
    mock = Mock()
    mock.run = AsyncMock(return_value=AgentRunResult(output=return_value))
    return mock


@pytest.mark.asyncio
async def test_router_ask_question_route(
    mock_session: AsyncSessionMock,
    mock_whatsapp: AsyncMock,
    mock_embedding_client: AsyncMock,
    test_message: Message,
    mock_settings: Mock,
    monkeypatch: pytest.MonkeyPatch,
):
    # Mock the Agent class for routing
    mock_route_agent = MockAgent(Intent(intent=IntentEnum.ask_question))

    # Mock the Agent class for rephrasing
    mock_rephrasing_agent = MockAgent("rephrased question")

    # Mock the Agent class for generation
    mock_generation_agent = MockAgent("cool response")

    # Setup agent mocks - cycle through agents in order: route, rephrase, generate
    agents = {
        "route": mock_route_agent,
        "rephrase": mock_rephrasing_agent,
        "generate": mock_generation_agent,
    }
    agent_counter = 0

    def mock_agent_init(*args, **kwargs):
        nonlocal agent_counter
        return None

    def mock_agent_run(*args, **kwargs):
        nonlocal agent_counter
        agent = list(agents.values())[agent_counter]
        agent_counter = (agent_counter + 1) % len(agents)
        return agent.run(*args, **kwargs)

    monkeypatch.setattr(Agent, "__init__", mock_agent_init)
    monkeypatch.setattr(Agent, "run", mock_agent_run)

    # Mock session.exec() to handle all database queries
    def mock_exec_side_effect(*args, **kwargs):
        mock_result = AsyncMock()
        mock_result.all = Mock(return_value=[])  # Return empty by default
        mock_result.first = Mock(return_value=None)  # Return None for first()
        mock_result.__aiter__ = AsyncMock(return_value=iter([]))  # For iteration
        return mock_result

    mock_session.exec.side_effect = mock_exec_side_effect
    mock_session.get = AsyncMock(return_value=None)  # No existing records
    mock_session.add = AsyncMock()  # Mock add operation
    mock_session.flush = AsyncMock()  # Mock flush operation

    # Mock execute operation for upsert and search
    mock_execute_result = MagicMock()
    mock_execute_result.fetchall.return_value = []
    mock_session.execute = AsyncMock(return_value=mock_execute_result)

    # Mock begin_nested for nested transactions
    mock_nested = AsyncMock()
    mock_nested.__aenter__ = AsyncMock(return_value=mock_nested)
    mock_nested.__aexit__ = AsyncMock(return_value=None)
    mock_session.begin_nested = Mock(return_value=mock_nested)

    # Set up mock response for send_message
    mock_response = AsyncMock()
    mock_response.results.message_id = "response_id"
    mock_whatsapp.send_message.return_value = mock_response

    # Create router instance
    router = Router(mock_session, mock_whatsapp, mock_embedding_client, mock_settings)

    # Test the route
    await router(test_message)

    # Verify the message was sent
    mock_whatsapp.send_message.assert_called_once_with(
        SendMessageRequest(
            phone="user@s.whatsapp.net",
            message="cool response",
            reply_message_id="test_id",
        )
    )


@pytest.mark.asyncio
async def test_router_summarize_route(
    mock_session: AsyncSessionMock,
    mock_whatsapp: AsyncMock,
    mock_embedding_client: AsyncMock,
    test_message: Message,
    mock_settings: Mock,
    monkeypatch: pytest.MonkeyPatch,
):
    test_message.text = "What happened this week?"
    # Mock the Agent class for routing
    mock_route_agent = MockAgent(Intent(intent=IntentEnum.summarize))

    # Mock the Agent class for summarization
    mock_summarize_agent = MockAgent("Summary of messages")

    # Setup agent mocks
    agents = {"route": mock_route_agent, "summarize": mock_summarize_agent}
    agent_counter = 0

    def mock_agent_init(*args, **kwargs):
        nonlocal agent_counter
        return None

    def mock_agent_run(*args, **kwargs):
        nonlocal agent_counter
        agent = list(agents.values())[agent_counter]
        agent_counter = (agent_counter + 1) % len(agents)
        return agent.run(*args, **kwargs)

    monkeypatch.setattr(Agent, "__init__", mock_agent_init)
    monkeypatch.setattr(Agent, "run", mock_agent_run)

    # Mock session.exec() for message history
    mock_exec = AsyncMock()
    mock_exec.all.return_value = [test_message]
    mock_session.exec.return_value = mock_exec

    # Set up mock response for send_message
    mock_response = AsyncMock()
    mock_response.results.message_id = "response_id"
    mock_whatsapp.send_message.return_value = mock_response

    # Create router instance
    router = Router(mock_session, mock_whatsapp, mock_embedding_client, mock_settings)
    router.ask_knowledge_base = AsyncMock()

    # Test the route
    await router(test_message)

    router.ask_knowledge_base.assert_awaited_once_with(test_message)
    mock_whatsapp.send_message.assert_not_called()


@pytest.mark.asyncio
async def test_router_other_route(
    mock_session: AsyncSessionMock,
    mock_whatsapp: AsyncMock,
    mock_embedding_client: AsyncMock,
    test_message: Message,
    mock_settings: Mock,
    monkeypatch: pytest.MonkeyPatch,
):
    test_message.text = "Tell me about weather"
    # Mock the Agent class
    mock_agent = MockAgent(Intent(intent=IntentEnum.other))
    monkeypatch.setattr(Agent, "__init__", lambda *args, **kwargs: None)
    monkeypatch.setattr(Agent, "run", mock_agent.run)

    # Set up mock response for send_message
    mock_response = AsyncMock()
    mock_response.results.message_id = "response_id"
    mock_whatsapp.send_message.return_value = mock_response

    # Create router instance
    router = Router(mock_session, mock_whatsapp, mock_embedding_client, mock_settings)

    # Test the route
    await router(test_message)

    mock_whatsapp.send_message.assert_not_called()


@pytest.mark.asyncio
async def test_send_message(
    mock_session: AsyncSessionMock,
    mock_whatsapp: AsyncMock,
    mock_embedding_client: AsyncMock,
    mock_settings: Mock,
):
    # Set up mock response
    mock_response = AsyncMock()
    mock_response.results.message_id = "response_id"
    mock_whatsapp.send_message.return_value = mock_response
    mock_session.get.return_value = None  # Simulate sender doesn't exist

    # Create router instance
    router = Router(mock_session, mock_whatsapp, mock_embedding_client, mock_settings)

    # Test sending a message
    await router.send_message("user@s.whatsapp.net", "Test message")

    # Verify the message was sent and stored
    mock_whatsapp.send_message.assert_called_once()
    mock_session.flush.assert_called()


@pytest.mark.asyncio
async def test_router_summarize_with_opt_out(
    mock_session: AsyncSessionMock,
    mock_whatsapp: AsyncMock,
    mock_embedding_client: AsyncMock,
    test_message: Message,
    mock_settings: Mock,
    monkeypatch: pytest.MonkeyPatch,
):
    test_message.text = "Could you assist"
    # Mock the Agent class for routing
    mock_route_agent = MockAgent(Intent(intent=IntentEnum.summarize))

    # Mock the Agent class for summarization
    mock_summarize_agent = MockAgent("Summary of messages")

    # Setup agent mocks
    agents = {"route": mock_route_agent, "summarize": mock_summarize_agent}
    agent_counter = 0

    def mock_agent_init(*args, **kwargs):
        nonlocal agent_counter
        return None

    def mock_agent_run(*args, **kwargs):
        nonlocal agent_counter
        agent = list(agents.values())[agent_counter]
        agent_counter = (agent_counter + 1) % len(agents)
        return agent.run(*args, **kwargs)

    monkeypatch.setattr(Agent, "__init__", mock_agent_init)
    monkeypatch.setattr(Agent, "run", mock_agent_run)

    # Mock session.exec() for message history
    mock_exec = AsyncMock()
    mock_exec.all.return_value = [test_message]
    mock_session.exec.return_value = mock_exec

    # Set up mock response for send_message
    mock_response = AsyncMock()
    mock_response.results.message_id = "response_id"
    mock_whatsapp.send_message.return_value = mock_response

    # Create router instance
    router = Router(mock_session, mock_whatsapp, mock_embedding_client, mock_settings)

    # Mock get_opt_out_map
    with patch(
        "handler.router.get_opt_out_map", new_callable=AsyncMock
    ) as mock_get_opt_out_map:
        mock_get_opt_out_map.return_value = {"user": "John Doe"}

        # Test the route
        await router(test_message)

        # Summaries are intentionally silent unless a grounded content request
        # reaches the knowledge-base path.
        mock_get_opt_out_map.assert_not_called()

    mock_whatsapp.send_message.assert_not_called()


@pytest.mark.asyncio
async def test_router_nudges_clear_banter_without_llm(
    mock_session: AsyncSessionMock,
    mock_whatsapp: AsyncMock,
    mock_embedding_client: AsyncMock,
    mock_settings: Mock,
):
    router = Router(mock_session, mock_whatsapp, mock_embedding_client, mock_settings)
    router.send_message = AsyncMock()
    message = Message(
        message_id="joke-1",
        text="lol 😂",
        chat_jid="user@s.whatsapp.net",
        sender_jid="user@s.whatsapp.net",
    )

    await router(message)

    router.send_message.assert_awaited_once()
    assert (
        "here to help with the UniPods METI programme"
        in (router.send_message.await_args.args[1])
    )


@pytest.mark.asyncio
async def test_fixed_bot_reply_only_matches_assistant_questions(
    mock_session: AsyncSessionMock,
    mock_whatsapp: AsyncMock,
    mock_embedding_client: AsyncMock,
    mock_settings: Mock,
):
    router = Router(mock_session, mock_whatsapp, mock_embedding_client, mock_settings)
    router.send_message = AsyncMock()
    router.ask_knowledge_base = AsyncMock()
    router._route = AsyncMock(return_value=IntentEnum.ask_question)

    programme_questions = [
        "how many teams ( for bot hackathon) that still need someone",
        "what is this hackathon about?",
    ]
    for index, text in enumerate(programme_questions):
        message = Message(
            message_id=f"programme-{index}",
            text=text,
            chat_jid="user@s.whatsapp.net",
            sender_jid="user@s.whatsapp.net",
        )
        await router(message)

    assert router.send_message.await_count == 0
    assert router.ask_knowledge_base.await_count == 2

    router.ask_knowledge_base.reset_mock()
    await router(
        Message(
            message_id="assistant-question",
            text="who are you",
            chat_jid="user@s.whatsapp.net",
            sender_jid="user@s.whatsapp.net",
        )
    )

    router.send_message.assert_awaited_once()
    assert router.send_message.await_args.args[1] == (
        "I'm Nexus, the UniPods METI AI programme assistant. Ask me about sessions, deadlines, MIT, Wadhwani or links."
    )
    router.ask_knowledge_base.assert_not_awaited()


@pytest.mark.asyncio
async def test_router_routes_french_content_question_to_knowledge_base(
    mock_session: AsyncSessionMock,
    mock_whatsapp: AsyncMock,
    mock_embedding_client: AsyncMock,
    mock_settings: Mock,
):
    router = Router(mock_session, mock_whatsapp, mock_embedding_client, mock_settings)
    router.ask_knowledge_base = AsyncMock()
    router._route = AsyncMock(return_value=IntentEnum.other)
    message = Message(
        message_id="french-question",
        text="Pouvez-vous me donner le lien d'inscription au hackathon ?",
        chat_jid="user@s.whatsapp.net",
        sender_jid="user@s.whatsapp.net",
    )

    await router(message)

    router.ask_knowledge_base.assert_awaited_once_with(message)
    router._route.assert_not_awaited()


@pytest.mark.asyncio
async def test_router_routes_contextual_expansion_before_generic_silent_path(
    mock_session: AsyncSessionMock,
    mock_whatsapp: AsyncMock,
    mock_embedding_client: AsyncMock,
    mock_settings: Mock,
):
    router = Router(mock_session, mock_whatsapp, mock_embedding_client, mock_settings)
    router.ask_knowledge_base = AsyncMock()
    router.ask_knowledge_base.resolve_conversation_context = AsyncMock(
        return_value=ConversationResolution(
            current_message="In details",
            resolved_query="Expand the previous question.",
            kind=ConversationKind.expansion,
            retrieval_mode=RetrievalMode.conversational_followup,
            answer_depth=AnswerDepth.detailed,
            is_follow_up=True,
        )
    )
    router._route = AsyncMock(return_value=IntentEnum.other)
    message = Message(
        message_id="contextual-expansion",
        text="In details",
        chat_jid="user@s.whatsapp.net",
        sender_jid="user@s.whatsapp.net",
    )

    await router(message)

    router.ask_knowledge_base.assert_awaited_once_with(message)
    router._route.assert_not_awaited()
