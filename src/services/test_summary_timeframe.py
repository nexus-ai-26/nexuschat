from datetime import datetime, timedelta

from services.summary_timeframe import MAX_SUMMARY_WINDOW, parse_summary_timeframe


def test_last_four_hours_is_parsed_exactly():
    now = datetime(2026, 9, 24, 10, 0)
    window = parse_summary_timeframe("@Nexus summarize the last 4 hours", now=now)

    assert window.explicit is True
    assert window.end == now.replace(tzinfo=window.end.tzinfo)
    assert window.end - window.start == timedelta(hours=4)


def test_last_two_weeks_is_clamped_to_safe_maximum():
    now = datetime(2026, 9, 24, 10, 0)
    window = parse_summary_timeframe("summarize the last 30 days", now=now)

    assert window.end - window.start == MAX_SUMMARY_WINDOW


def test_custom_yesterday_to_today_range_is_supported():
    now = datetime(2026, 9, 24, 10, 0)
    window = parse_summary_timeframe(
        "summarize from 2pm yesterday until 10am today", now=now
    )

    assert window.explicit is True
    assert window.start.hour == 14
    assert window.end.hour == 10
    assert window.start.date().day == 23
    assert window.end.date().day == 24


def test_missing_timeframe_uses_bounded_recent_default():
    now = datetime(2026, 9, 24, 10, 0)
    window = parse_summary_timeframe("summarize recent updates", now=now)

    assert window.explicit is False
    assert window.end - window.start == timedelta(days=2)
