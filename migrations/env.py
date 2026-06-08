"""Alembic environment (async-aware, portable; GBO-R8b / EVOL-R4).

The migration layer is the ONLY place an unavoidable dialect construct may live
(GBO-R8b); these revisions emit only the portable types the column factories produce
(enum-as-text+CHECK, portable ``JSON``, ``DateTime(timezone=True)``, ``Uuid``), so the
same revisions run unchanged on SQLite / PostgreSQL / MariaDB — only the DSN differs.

The target metadata is ``Base.metadata`` fully populated by importing the models
package. The database URL is never read from ``alembic.ini``: it comes from
``wattwise_core.config.get_settings().database_dsn`` (env / secret-manager only,
BOOT-R4), normalized onto the async drivers by the engine module.
"""

from __future__ import annotations

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy.engine import Connection

from wattwise_core.agent import memory as _agent_memory  # noqa: F401 (registers MemoryItem)
from wattwise_core.agent.state_store import AgentStateBase
from wattwise_core.config import get_settings
from wattwise_core.persistence.engine import create_engine_from_settings, normalize_dsn
from wattwise_core.persistence.models import Base

# Alembic Config object (reads alembic.ini).
config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# The canonical schema (Base) plus the SEPARATE agent-state store schema (AgentStateBase,
# ARCH-R13 — checkpoints/threads/writes never on the canonical store). Both are managed in
# one migration chain; target_metadata is the union so autogenerate/`alembic check` see the
# full schema rather than reporting the agent-state tables as extraneous.
target_metadata = [Base.metadata, AgentStateBase.metadata]


def _database_url() -> str:
    """Resolve the DSN from settings (fail-closed) and normalize to an async driver."""
    settings = get_settings()
    if settings.database_dsn is None:  # pragma: no cover - config layer enforces this
        raise RuntimeError("fail-closed: WATTWISE_DATABASE_DSN is required to run migrations")
    return normalize_dsn(settings.database_dsn.get_secret_value())


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode (emit SQL against a URL, no DBAPI)."""
    context.configure(
        url=_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True,
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def _do_run_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        # batch mode keeps ALTER portable on SQLite (no native ALTER) without code change.
        render_as_batch=True,
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    """Run migrations in 'online' mode against an async engine."""
    engine = create_engine_from_settings()
    try:
        async with engine.connect() as connection:
            await connection.run_sync(_do_run_migrations)
    finally:
        await engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
