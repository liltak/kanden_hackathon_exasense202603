"""Initial schema: tasks, simulations, mesh_files, users.

Revision ID: 001
Revises:
Create Date: 2026-03-01
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "tasks",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("task_type", sa.String(32), nullable=False, server_default="simulation"),
        sa.Column("status", sa.String(16), nullable=False, server_default="pending"),
        sa.Column("progress", sa.Float, nullable=False, server_default="0"),
        sa.Column("step", sa.String(64), nullable=True),
        sa.Column("message", sa.Text, nullable=True),
        sa.Column("params", sa.JSON, nullable=True),
        sa.Column("result", sa.JSON, nullable=True),
        sa.Column("elapsed_seconds", sa.Float, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_table(
        "simulations",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("task_id", sa.String(36), nullable=False, index=True),
        sa.Column("latitude", sa.Float, nullable=False),
        sa.Column("longitude", sa.Float, nullable=False),
        sa.Column("year", sa.Integer, nullable=False),
        sa.Column("config", sa.JSON, nullable=True),
        sa.Column("irradiance_data", sa.JSON, nullable=True),
        sa.Column("roi_data", sa.JSON, nullable=True),
        sa.Column("monthly_ghi", sa.JSON, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_table(
        "mesh_files",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("filename", sa.String(255), nullable=False),
        sa.Column("object_key", sa.String(512), nullable=False),
        sa.Column("num_vertices", sa.Integer, nullable=False),
        sa.Column("num_faces", sa.Integer, nullable=False),
        sa.Column("surface_area_m2", sa.Float, nullable=False),
        sa.Column("bounds_min", sa.JSON, nullable=False),
        sa.Column("bounds_max", sa.JSON, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_table(
        "users",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("username", sa.String(64), nullable=False, unique=True),
        sa.Column("email", sa.String(255), nullable=False, unique=True),
        sa.Column("hashed_password", sa.String(255), nullable=False),
        sa.Column("role", sa.String(16), nullable=False, server_default="viewer"),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default="true"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_users_username", "users", ["username"])
    op.create_index("ix_users_email", "users", ["email"])


def downgrade() -> None:
    op.drop_table("users")
    op.drop_table("mesh_files")
    op.drop_table("simulations")
    op.drop_table("tasks")
