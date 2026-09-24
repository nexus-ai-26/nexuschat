"""Safe, database-backed group summary scheduling primitives."""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from sqlmodel import col, select

from models import Group, GroupSchedule
from services.group_control import group_is_selected


SCHEDULE_EVERY_4_HOURS = "every_4_hours"
SCHEDULE_EVENING = "evening"
SCHEDULE_CUSTOM = "custom"
SUPPORTED_SCHEDULE_KINDS = {
    SCHEDULE_EVERY_4_HOURS,
    SCHEDULE_EVENING,
    SCHEDULE_CUSTOM,
}
_CLOCK_RE = re.compile(r"^(?:[01]?\d|2[0-3]):[0-5]\d$")


def next_schedule_time(
    schedule: GroupSchedule, *, now: datetime | None = None
) -> datetime:
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    if schedule.schedule_kind == SCHEDULE_EVERY_4_HOURS:
        return current + timedelta(hours=4)
    if schedule.schedule_kind == SCHEDULE_EVENING:
        try:
            local_zone = ZoneInfo(schedule.timezone_name)
        except ZoneInfoNotFoundError:
            local_zone = timezone.utc
        local_now = current.astimezone(local_zone)
        target = local_now.replace(hour=18, minute=0, second=0, microsecond=0)
        if target <= local_now:
            target += timedelta(days=1)
        return target.astimezone(timezone.utc)
    custom = (schedule.cron_expression or "").strip()
    if _CLOCK_RE.fullmatch(custom):
        hour, minute = (int(part) for part in custom.split(":", 1))
        try:
            local_zone = ZoneInfo(schedule.timezone_name)
        except ZoneInfoNotFoundError:
            local_zone = timezone.utc
        local_now = current.astimezone(local_zone)
        target = local_now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if target <= local_now:
            target += timedelta(days=1)
        return target.astimezone(timezone.utc)
    # A retained cron expression is still safe to store and expose through the
    # control plane; unsupported expressions wait for a daily manual review.
    return current + timedelta(days=1)


def validate_schedule(
    schedule_kind: str,
    *,
    cron_expression: str | None = None,
    timezone_name: str = "UTC",
) -> None:
    if schedule_kind not in SUPPORTED_SCHEDULE_KINDS:
        raise ValueError("unsupported_schedule_kind")
    try:
        ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError as error:
        raise ValueError("invalid_timezone") from error
    if schedule_kind == SCHEDULE_CUSTOM and not (cron_expression or "").strip():
        raise ValueError("custom_schedule_requires_time")


async def claim_due_schedules(session, *, now: datetime | None = None) -> list[Group]:
    """Claim due rows before any LLM/network work, making execution idempotent."""

    current = now or datetime.now(timezone.utc)
    result = await session.exec(
        select(GroupSchedule)
        .where(GroupSchedule.enabled == True)  # noqa: E712
        .where(GroupSchedule.next_run_at != None)  # noqa: E711
        .where(col(GroupSchedule.next_run_at) <= current)
        .with_for_update()
    )
    due_groups: list[Group] = []
    for schedule in result.all():
        group = await session.get(Group, schedule.group_jid)
        schedule.last_run_at = current
        schedule.next_run_at = next_schedule_time(schedule, now=current)
        if group is not None and group_is_selected(group) and not group.paused:
            due_groups.append(group)
    await session.commit()
    return due_groups
