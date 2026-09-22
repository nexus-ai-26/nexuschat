from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import pytest

from models import CatchupAttempt
from models import Message
from services.catchup import (
    CatchupService,
    GOWA_CHAT_HISTORY_MAX,
    find_unanswered_mentions,
    select_catchup_candidates,
    valid_admin_secret,
)
from utils.llm_provider import _provider_rate_limited_at, provider_rate_limit_recently
from whatsapp import WhatsAppClient


BOT_JID = "999@s.whatsapp.net"
GROUP_JID = "120363000000000000@g.us"
NOW = datetime.now(timezone.utc)


def make_message(
    message_id: str,
    text: str,
    *,
    timestamp: datetime = NOW,
    sender_jid: str = "111@s.whatsapp.net",
) -> Message:
    return Message(
        message_id=message_id,
        text=text,
        timestamp=timestamp,
        chat_jid=GROUP_JID,
        sender_jid=sender_jid,
    )


def test_already_answered_mention_is_skipped_and_second_scan_is_idempotent():
    message = make_message("m1", "@999 what time does it start?")

    candidates, already_answered = find_unanswered_mentions(
        [message],
        bot_jid=BOT_JID,
        answered_ids={message.message_id},
        window_start=NOW - timedelta(hours=6),
        now=NOW,
    )
    assert candidates == []
    assert already_answered == 1

    answered_ids: set[str] = set()
    first, _ = find_unanswered_mentions(
        [message],
        bot_jid=BOT_JID,
        answered_ids=answered_ids,
        window_start=NOW - timedelta(hours=6),
        now=NOW,
    )
    assert [item.message_id for item in first] == ["m1"]
    answered_ids.add(message.message_id)

    second, _ = find_unanswered_mentions(
        [message],
        bot_jid=BOT_JID,
        answered_ids=answered_ids,
        window_start=NOW - timedelta(hours=6),
        now=NOW,
    )
    assert second == []


def test_unanswered_mentions_are_oldest_first_and_cap_is_respected():
    messages = [
        make_message(
            f"m{index}",
            "@999 what is the answer?",
            timestamp=NOW - timedelta(minutes=3 * index),
        )
        for index in range(4)
    ]
    candidates, _ = find_unanswered_mentions(
        messages,
        bot_jid=BOT_JID,
        answered_ids=set(),
        window_start=NOW - timedelta(hours=6),
        now=NOW,
    )
    selected, skipped = select_catchup_candidates(candidates, 2)

    assert [item.message_id for item in selected] == ["m3", "m2"]
    assert skipped == 2


def test_older_than_window_is_skipped():
    old = make_message("old", "@999 answer this", timestamp=NOW - timedelta(hours=7))

    candidates, already_answered = find_unanswered_mentions(
        [old],
        bot_jid=BOT_JID,
        answered_ids=set(),
        window_start=NOW - timedelta(hours=6),
        now=NOW,
    )

    assert candidates == []
    assert already_answered == 0


def test_other_bots_and_bot_messages_are_skipped():
    messages = [
        make_message("bot", "@999 answer", sender_jid=BOT_JID),
        make_message("other", "@999 answer", sender_jid="888@s.whatsapp.net"),
    ]
    candidates, _ = find_unanswered_mentions(
        messages,
        bot_jid=BOT_JID,
        answered_ids=set(),
        window_start=NOW - timedelta(hours=6),
        now=NOW,
        other_bot_jids=["888@s.whatsapp.net"],
    )
    assert candidates == []


def test_admin_secret_requires_configured_value():
    assert valid_admin_secret("catch-up-secret", "catch-up-secret")
    assert not valid_admin_secret("wrong", "catch-up-secret")
    assert not valid_admin_secret("catch-up-secret", None)


def test_recent_provider_rate_limit_blocks_catchup():
    import time

    _provider_rate_limited_at["deepseek"] = time.monotonic()
    assert provider_rate_limit_recently(300)
    _provider_rate_limited_at.clear()


def test_catchup_defaults_are_stale_question_safe():
    from config import Settings

    assert Settings.model_fields["catchup_window_hours"].default == 1.0
    assert Settings.model_fields["catchup_max_replies"].default == 5


@pytest.mark.asyncio
async def test_catchup_records_failed_attempt_without_marking_answered():
    class Session:
        def __init__(self):
            self.rows: dict[str, CatchupAttempt] = {}

        async def get(self, model, message_id):
            return self.rows.get(message_id)

        def add(self, row):
            self.rows[row.message_id] = row

        async def commit(self):
            return None

    session = Session()
    service = CatchupService(SimpleNamespace(state=SimpleNamespace()))
    await service._begin_attempt(session, "failed-message")
    await service._finish_attempt(
        session, "failed-message", answered=False, error="ProviderFallbackError"
    )

    assert session.rows["failed-message"].status == "failed"
    assert session.rows["failed-message"].last_error == "ProviderFallbackError"


@pytest.mark.asyncio
async def test_bridge_history_uses_gowa_supported_limit():
    response = SimpleNamespace(results=[])
    whatsapp = SimpleNamespace(get_chat_messages=AsyncMock(return_value=response))
    service = CatchupService(SimpleNamespace(state=SimpleNamespace()))
    start = NOW - timedelta(hours=1)

    await service._bridge_history(cast(WhatsAppClient, whatsapp), GROUP_JID, start, NOW)

    params = whatsapp.get_chat_messages.await_args.kwargs["params"]
    assert params.limit == GOWA_CHAT_HISTORY_MAX == 100
