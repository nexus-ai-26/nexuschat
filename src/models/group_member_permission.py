from datetime import datetime, timezone

from sqlmodel import Column, DateTime, Field, SQLModel


class GroupMemberPermission(SQLModel, table=True):
    """Optional per-member override; default group policy remains permissive."""

    group_jid: str = Field(
        primary_key=True, max_length=255, foreign_key="group.group_jid"
    )
    member_jid: str = Field(primary_key=True, max_length=255)
    allowed: bool = Field(default=True, nullable=False)
    updated_by: str = Field(max_length=255)
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )
