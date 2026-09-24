from datetime import datetime, timezone

from sqlalchemy import Column, DateTime
from sqlmodel import Field, SQLModel


class CatchupAttempt(SQLModel, table=True):
    __tablename__: str = "catchup_attempt"

    message_id: str = Field(primary_key=True, max_length=255)
    attempted_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )
    status: str = Field(default="attempted", max_length=32)
    attempt_count: int = Field(default=1, nullable=False)
    last_error: str | None = Field(default=None)
