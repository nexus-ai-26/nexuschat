from datetime import datetime, timezone
from typing import ClassVar

from sqlalchemy import Column, DateTime
from sqlmodel import Field, SQLModel


class DMGreeting(SQLModel, table=True):
    """Durable claim that a private-chat opening was sent to a user."""

    __tablename__: ClassVar[str] = "dm_greeting"
    sender_jid: str = Field(primary_key=True, max_length=255)
    sent_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )
