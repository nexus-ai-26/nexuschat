from datetime import datetime, timezone
from types import SimpleNamespace

from handler.auto_reply import (
    AUTO_REPLY_MAX_VECTOR_DISTANCE,
    AutoReplyLimiter,
    best_vector_distance,
    has_confident_match,
    is_no_answer,
    normalize_question,
    is_too_short_for_auto_reply,
    thread_key,
)
from models import Message


def _message(**kwargs) -> Message:
    defaults = dict(
        message_id="m1",
        chat_jid="g@g.us",
        sender_jid="user@s.whatsapp.net",
        group_jid="g@g.us",
        text="MIT platform blank pages",
        timestamp=datetime.now(timezone.utc),
    )
    defaults.update(kwargs)
    return Message(**defaults)


def test_has_confident_match_respects_cutoff():
    close = SimpleNamespace(vector_distance=0.31)
    far = SimpleNamespace(vector_distance=0.33)
    assert has_confident_match([close])
    assert not has_confident_match([far])
    assert not has_confident_match([])
    assert AUTO_REPLY_MAX_VECTOR_DISTANCE == 0.32


def test_best_vector_distance():
    results = [
        SimpleNamespace(vector_distance=0.4),
        SimpleNamespace(vector_distance=0.2),
    ]
    assert best_vector_distance(results) == 0.2
    assert best_vector_distance([]) is None


def test_thread_key_prefers_reply_to():
    root = _message(message_id="root")
    reply = _message(message_id="child", reply_to_id="root")
    assert thread_key(root) == "root"
    assert thread_key(reply) == "root"


def test_short_and_no_answer_helpers():
    assert is_too_short_for_auto_reply("ok")
    assert is_too_short_for_auto_reply("   hi   ")
    assert is_too_short_for_auto_reply(None)
    assert not is_too_short_for_auto_reply("Is MIT down?")
    assert is_no_answer("NO_ANSWER")
    assert is_no_answer("NO_ANSWER\nextra")
    assert not is_no_answer("NO_ANSWER is not enough evidence")


def test_limiter_cooldown_and_same_thread():
    limiter = AutoReplyLimiter(cooldown_seconds=90)
    first = _message(message_id="a")
    assert limiter.skip_reason("g@g.us", first) is None
    limiter.record("g@g.us", first)
    assert limiter.skip_reason("g@g.us", first) == "same-thread back-to-back"
    other = _message(message_id="b")
    reason = limiter.skip_reason("g@g.us", other)
    assert reason is not None and reason.startswith("cooldown")
    limiter.reset()
    assert limiter.skip_reason("g@g.us", other) is None


def test_limiter_enforces_group_cap():
    limiter = AutoReplyLimiter(group_max_replies=2, group_window_seconds=60)
    limiter.record("g@g.us", _message(message_id="a", sender_jid="a@s.whatsapp.net"))
    limiter.record("g@g.us", _message(message_id="b", sender_jid="b@s.whatsapp.net"))

    assert (
        limiter.skip_reason(
            "g@g.us", _message(message_id="c", sender_jid="c@s.whatsapp.net")
        )
        == "group cap (2 replies/60s)"
    )


def test_normalized_duplicate_question_is_skipped_for_ten_minutes():
    limiter = AutoReplyLimiter()
    first = _message(
        message_id="first",
        sender_jid="first@s.whatsapp.net",
        text="What is this hackathon about?",
    )
    second = _message(
        message_id="second",
        sender_jid="second@s.whatsapp.net",
        text=" what IS this hackathon about! ",
    )

    assert normalize_question(first.text) == normalize_question(second.text)
    assert limiter.skip_reason("g@g.us", first) is None
    limiter.record("g@g.us", first)
    assert limiter.skip_reason("g@g.us", second) == "duplicate question"
