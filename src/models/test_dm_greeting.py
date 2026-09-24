from typing import Any, cast

from models import DMGreeting


def test_dm_greeting_timestamp_matches_timezone_aware_migration():
    table = cast(Any, DMGreeting).__table__
    assert table.c.sent_at.type.timezone is True
