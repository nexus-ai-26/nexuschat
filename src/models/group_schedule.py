from datetime import datetime, timezone

from sqlalchemy import UniqueConstraint
from sqlmodel import Column, DateTime, Field, SQLModel


class GroupSchedule(SQLModel, table=True):
    """Durable group summary schedule definition."""

    id: int | None = Field(default=None, primary_key=True)
    group_jid: str = Field(max_length=255, foreign_key="group.group_jid")
    schedule_kind: str = Field(max_length=32)
    cron_expression: str | None = Field(default=None, max_length=128)
    timezone_name: str = Field(default="UTC", max_length=64)
    enabled: bool = Field(default=True, nullable=False)
    last_run_at: datetime | None = Field(
        default=None, sa_column=Column(DateTime(timezone=True), nullable=True)
    )
    next_run_at: datetime | None = Field(
        default=None, sa_column=Column(DateTime(timezone=True), nullable=True)
    )
    created_by: str = Field(max_length=255)
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )

    __table_args__ = (
        UniqueConstraint(
            "group_jid",
            "schedule_kind",
            name="uq_group_schedule_kind",
        ),
    )
