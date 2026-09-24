from datetime import datetime, timezone

import pytest

from models import GroupSchedule
from services.group_scheduler import (
    SCHEDULE_CUSTOM,
    SCHEDULE_EVENING,
    SCHEDULE_EVERY_4_HOURS,
    next_schedule_time,
    validate_schedule,
)


def test_every_four_hours_schedule_is_bounded():
    now = datetime(2026, 9, 24, 10, tzinfo=timezone.utc)
    schedule = GroupSchedule(
        group_jid="g@g.us",
        schedule_kind=SCHEDULE_EVERY_4_HOURS,
        created_by="admin@s.whatsapp.net",
    )

    assert next_schedule_time(schedule, now=now) == datetime(
        2026, 9, 24, 14, tzinfo=timezone.utc
    )


def test_evening_schedule_rolls_to_next_local_evening():
    now = datetime(2026, 9, 24, 19, tzinfo=timezone.utc)
    schedule = GroupSchedule(
        group_jid="g@g.us",
        schedule_kind=SCHEDULE_EVENING,
        timezone_name="UTC",
        created_by="admin@s.whatsapp.net",
    )

    assert next_schedule_time(schedule, now=now).hour == 18
    assert next_schedule_time(schedule, now=now).day == 25


def test_custom_clock_schedule_is_supported_and_validated():
    now = datetime(2026, 9, 24, 10, tzinfo=timezone.utc)
    schedule = GroupSchedule(
        group_jid="g@g.us",
        schedule_kind=SCHEDULE_CUSTOM,
        cron_expression="11:30",
        created_by="admin@s.whatsapp.net",
    )

    validate_schedule(SCHEDULE_CUSTOM, cron_expression="11:30")
    assert next_schedule_time(schedule, now=now).hour == 11
    assert next_schedule_time(schedule, now=now).minute == 30


def test_custom_schedule_requires_a_value():
    with pytest.raises(ValueError, match="custom_schedule_requires_time"):
        validate_schedule(SCHEDULE_CUSTOM)


def test_unsupported_schedule_kind_is_rejected():
    with pytest.raises(ValueError, match="unsupported_schedule_kind"):
        validate_schedule("weekly")
