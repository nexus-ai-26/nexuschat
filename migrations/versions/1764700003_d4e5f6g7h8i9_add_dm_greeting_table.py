"""add durable private-chat greeting claims

Revision ID: d4e5f6g7h8i9
Revises: c3d4e5f6g7h8
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "d4e5f6g7h8i9"
down_revision: Union[str, None] = "c3d4e5f6g7h8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "dm_greeting",
        sa.Column("sender_jid", sa.String(length=255), primary_key=True),
        sa.Column("sent_at", sa.DateTime(timezone=True), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("dm_greeting")
