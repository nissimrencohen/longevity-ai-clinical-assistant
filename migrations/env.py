"""Alembic environment.

The DSN comes from ``Settings`` rather than alembic.ini, so the migration runner
and the application can never disagree about which database they mean — and no
credentials live in a committed file.

Runs through the **async** engine on asyncpg. Alembic's default template is
synchronous and would need psycopg2 as a second driver purely for migrations;
using the async path keeps the dependency set to one Postgres driver.
"""

from __future__ import annotations

import asyncio
import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy.ext.asyncio import create_async_engine

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.core.config import settings  # noqa: E402
from backend.app.db.models import metadata  # noqa: E402

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = metadata


def run_migrations_offline() -> None:
    """Emit SQL to stdout without connecting — useful for review."""
    context.configure(
        url=settings.postgres_dsn,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def _run(connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    engine = create_async_engine(settings.postgres_dsn, pool_pre_ping=True)
    async with engine.connect() as connection:
        await connection.run_sync(_run)
    await engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
