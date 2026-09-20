from types import SimpleNamespace

import pytest

import services.webhook_processing as processing


@pytest.mark.asyncio
async def test_failure_reply_is_limited_per_chat(monkeypatch: pytest.MonkeyPatch):
    processing._failure_reply_sent_at.clear()
    clock = iter((100.0, 101.0, 701.0, 702.0))
    monkeypatch.setattr(processing, "monotonic", lambda: next(clock))

    assert await processing._claim_failure_reply("group@g.us", 600) is True
    assert await processing._claim_failure_reply("group@g.us", 600) is False
    assert await processing._claim_failure_reply("other@g.us", 600) is True
    assert await processing._claim_failure_reply("group@g.us", 600) is True

    processing._failure_reply_sent_at.clear()


def test_failure_reply_cooldown_setting_has_ten_minute_default():
    settings = SimpleNamespace()
    assert getattr(settings, "failure_reply_cooldown_seconds", 600.0) == 600.0
