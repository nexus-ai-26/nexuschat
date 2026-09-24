"""add persistent group control plane

Revision ID: e5f6g7h8i9j0
Revises: d4e5f6g7h8i9
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "e5f6g7h8i9j0"
down_revision: Union[str, None] = "d4e5f6g7h8i9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "group",
        sa.Column("selected", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.add_column(
        "group",
        sa.Column("paused", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.add_column("group", sa.Column("paused_by", sa.String(length=255)))
    op.add_column(
        "group",
        sa.Column("paused_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "group",
        sa.Column("resumed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "group",
        sa.Column(
            "member_policy",
            sa.String(length=16),
            nullable=False,
            server_default="allow",
        ),
    )
    op.execute(sa.text('UPDATE "group" SET selected = managed'))
    op.alter_column("group", "selected", server_default=None)
    op.alter_column("group", "paused", server_default=None)
    op.alter_column("group", "member_policy", server_default=None)

    op.create_table(
        "groupmemberpermission",
        sa.Column("group_jid", sa.String(length=255), nullable=False),
        sa.Column("member_jid", sa.String(length=255), nullable=False),
        sa.Column("allowed", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("updated_by", sa.String(length=255), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["group_jid"], ["group.group_jid"]),
        sa.PrimaryKeyConstraint("group_jid", "member_jid"),
    )
    op.create_table(
        "groupschedule",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("group_jid", sa.String(length=255), nullable=False),
        sa.Column("schedule_kind", sa.String(length=32), nullable=False),
        sa.Column("cron_expression", sa.String(length=128)),
        sa.Column("timezone_name", sa.String(length=64), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("last_run_at", sa.DateTime(timezone=True)),
        sa.Column("next_run_at", sa.DateTime(timezone=True)),
        sa.Column("created_by", sa.String(length=255), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["group_jid"], ["group.group_jid"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("group_jid", "schedule_kind", name="uq_group_schedule_kind"),
    )
    op.alter_column("groupmemberpermission", "allowed", server_default=None)
    op.alter_column("groupschedule", "enabled", server_default=None)


def downgrade() -> None:
    op.drop_table("groupschedule")
    op.drop_table("groupmemberpermission")
    for name in ("member_policy", "resumed_at", "paused_at", "paused_by", "paused", "selected"):
        op.drop_column("group", name)
