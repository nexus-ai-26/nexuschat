"""Safe parsing of explicit group-summary time windows."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone


MAX_SUMMARY_WINDOW = timedelta(days=14)
DEFAULT_SUMMARY_WINDOW = timedelta(days=2)
_NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "eight": 8,
}
_DURATION_RE = re.compile(
    r"\blast\s+(?P<amount>\d+|one|two|three|four|five|six|eight)\s+"
    r"(?P<unit>hours?|days?|weeks?)\b",
    re.IGNORECASE,
)
_RANGE_RE = re.compile(
    r"\bfrom\s+(?P<start>\d{1,2}(?::\d{2})?\s*(?:am|pm)\s+"
    r"(?:yesterday|today))\s+(?:until|to|through)\s+"
    r"(?P<end>\d{1,2}(?::\d{2})?\s*(?:am|pm)\s+(?:yesterday|today))\b",
    re.IGNORECASE,
)
_TIME_RE = re.compile(
    r"^(?P<hour>\d{1,2})(?::(?P<minute>\d{2}))?\s*(?P<meridiem>am|pm)\s+"
    r"(?P<day>yesterday|today)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SummaryWindow:
    start: datetime
    end: datetime
    label: str
    explicit: bool


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _parse_clock(value: str, now: datetime) -> datetime | None:
    match = _TIME_RE.fullmatch(value.strip())
    if not match:
        return None
    hour = int(match.group("hour"))
    minute = int(match.group("minute") or 0)
    if hour > 12 or minute > 59:
        return None
    if match.group("meridiem").casefold() == "pm" and hour != 12:
        hour += 12
    if match.group("meridiem").casefold() == "am" and hour == 12:
        hour = 0
    day = now.date()
    if match.group("day").casefold() == "yesterday":
        day -= timedelta(days=1)
    return datetime.combine(day, time(hour, minute), tzinfo=now.tzinfo)


def parse_summary_timeframe(
    text: str | None, *, now: datetime | None = None
) -> SummaryWindow:
    """Parse a requested window, safely clamping it to fourteen days."""

    current = _aware(now or datetime.now().astimezone())
    query = " ".join((text or "").split())
    duration = _DURATION_RE.search(query)
    if duration:
        raw_amount = duration.group("amount").casefold()
        amount = int(raw_amount) if raw_amount.isdigit() else _NUMBER_WORDS[raw_amount]
        unit = duration.group("unit").casefold()
        if unit.startswith("hour"):
            delta = timedelta(hours=amount)
        elif unit.startswith("week"):
            delta = timedelta(days=amount * 7)
        else:
            delta = timedelta(days=amount)
        delta = min(delta, MAX_SUMMARY_WINDOW)
        return SummaryWindow(current - delta, current, f"last {amount} {unit}", True)

    time_range = _RANGE_RE.search(query)
    if time_range:
        start = _parse_clock(time_range.group("start"), current)
        end = _parse_clock(time_range.group("end"), current)
        if start is not None and end is not None and start < end:
            if end > current:
                end = current
            start = max(start, end - MAX_SUMMARY_WINDOW)
            return SummaryWindow(start, end, "custom range", True)

    if re.search(r"\btoday\b", query, re.IGNORECASE):
        start = datetime.combine(current.date(), time.min, tzinfo=current.tzinfo)
        return SummaryWindow(start, current, "today", True)

    if re.search(r"\bfortnight\b", query, re.IGNORECASE):
        return SummaryWindow(
            current - MAX_SUMMARY_WINDOW, current, "last fortnight", True
        )

    return SummaryWindow(
        current - DEFAULT_SUMMARY_WINDOW,
        current,
        "recent two days",
        False,
    )
