"""add durable catch-up attempt records"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "c3d4e5f6g7h8"
down_revision: Union[str, None] = "b2c3d4e5f6g7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "catchup_attempt",
        sa.Column("message_id", sa.String(length=255), primary_key=True),
        sa.Column("attempted_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "status", sa.String(length=32), nullable=False, server_default="attempted"
        ),
        sa.Column("attempt_count", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("last_error", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_table("catchup_attempt")
