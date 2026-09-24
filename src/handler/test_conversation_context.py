from datetime import datetime, timezone

import pytest

from handler.conversation_context import (
    AnswerDepth,
    ConversationKind,
    RetrievalMode,
    is_contextual_follow_up_candidate,
    resolve_conversation_context,
)
from models import Message


def _message(
    message_id: str,
    text: str,
    sender_jid: str,
    minutes: int,
) -> Message:
    return Message(
        message_id=message_id,
        text=text,
        chat_jid="user@s.whatsapp.net",
        sender_jid=sender_jid,
        timestamp=datetime(2026, 9, 22, 10, minutes, tzinfo=timezone.utc),
    )


def _history(previous_user: str, previous_answer: str) -> list[Message]:
    return [
        _message("user-1", previous_user, "user@s.whatsapp.net", 0),
        _message("bot-1", previous_answer, "bot@s.whatsapp.net", 1),
    ]


@pytest.mark.parametrize(
    "follow_up",
    (
        "in details",
        "tell me more",
        "is that all?",
        "what about the second one?",
        "what does each one handle?",
    ),
)
def test_contextual_follow_ups_resolve_against_previous_turn(follow_up: str):
    resolution = resolve_conversation_context(
        follow_up,
        _history(
            "Who are the admins?", "Diana and Munira coordinate programme support."
        ),
        current_sender="user@s.whatsapp.net",
        bot_jid="bot@s.whatsapp.net",
    )

    assert resolution.is_follow_up is True
    assert resolution.previous_user_message == "Who are the admins?"
    assert resolution.previous_assistant_message is not None
    assert "Who are the admins?" in resolution.resolved_query


def test_detail_follow_up_requests_a_detailed_answer():
    resolution = resolve_conversation_context(
        "In details",
        _history("How can I improve my chatbot?", "Start with clear answers."),
        current_sender="user@s.whatsapp.net",
        bot_jid="bot@s.whatsapp.net",
    )

    assert resolution.kind == ConversationKind.expansion
    assert resolution.answer_depth == AnswerDepth.detailed
    assert resolution.retrieval_mode == RetrievalMode.conversational_followup
    assert "How can I improve my chatbot?" in resolution.resolved_query


def test_summary_follow_up_requests_comprehensive_recent_context():
    resolution = resolve_conversation_context(
        "Give me everything important that happened recently.",
        _history("Tell me about the programme.", "Here is a short overview."),
        current_sender="user@s.whatsapp.net",
        bot_jid="bot@s.whatsapp.net",
    )

    assert resolution.answer_depth == AnswerDepth.summary
    assert resolution.retrieval_mode == RetrievalMode.summary


def test_standalone_and_social_messages_are_not_blindly_follow_ups():
    history = _history("Who are the admins?", "Diana coordinates programme matters.")
    standalone = resolve_conversation_context(
        "What is the deadline?",
        history,
        current_sender="user@s.whatsapp.net",
        bot_jid="bot@s.whatsapp.net",
    )
    thanks = resolve_conversation_context(
        "Thanks",
        history,
        current_sender="user@s.whatsapp.net",
        bot_jid="bot@s.whatsapp.net",
    )

    assert standalone.is_follow_up is False
    assert standalone.kind == ConversationKind.standalone
    assert standalone.retrieval_mode == RetrievalMode.direct_fact
    assert thanks.kind == ConversationKind.social
    assert is_contextual_follow_up_candidate("Thanks") is False


def test_ambiguous_follow_up_without_context_is_not_routed_as_contextual():
    resolution = resolve_conversation_context(
        "Tell me more",
        [],
        current_sender="user@s.whatsapp.net",
        bot_jid="bot@s.whatsapp.net",
    )

    assert resolution.is_follow_up is False
    assert resolution.resolved_query == "Tell me more"


def test_resolution_uses_the_latest_same_sender_turn_before_another_user():
    history = [
        _message("old-user", "Tell me about MIT.", "user@s.whatsapp.net", 0),
        _message("old-bot", "MIT is self-paced.", "bot@s.whatsapp.net", 1),
        _message("other-user", "I have another question.", "other@s.whatsapp.net", 2),
    ]

    resolution = resolve_conversation_context(
        "Explain that",
        history,
        current_sender="user@s.whatsapp.net",
        bot_jid="bot@s.whatsapp.net",
    )

    assert resolution.previous_user_message == "Tell me about MIT."
