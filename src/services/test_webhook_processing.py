from types import SimpleNamespace

import services.webhook_processing as processing


def test_failure_reply_cooldown_is_not_part_of_silent_processing():
    settings = SimpleNamespace()
    assert not hasattr(processing, "_failure_reply_sent_at")
    assert not hasattr(processing, "_send_failure_reply")
    assert not hasattr(settings, "failure_reply_cooldown_seconds")
