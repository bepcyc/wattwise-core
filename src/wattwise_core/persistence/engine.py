"""Async engine + session factory, DSN-only across SQLite/PostgreSQL/MariaDB (GBO-R8b).

The backend is selected purely by the DSN (``WATTWISE_DATABASE_DSN``); no application
code branches on which backend is in use (the only dialect awareness lives in the
upsert seam). The engine targets ONLY the schema the migrations create — it never
connects to a pre-existing/host database (TASK data-safety rule).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from wattwise_core.config import Settings, get_settings


def _normalize_dsn(dsn: str) -> str:
    """Coerce common driverless DSNs onto the async drivers wattwise-core ships with."""
    prefixes = {
        "sqlite://": "sqlite+aiosqlite://",
        "postgresql://": "postgresql+asyncpg://",
        "postgres://": "postgresql+asyncpg://",
        "mysql://": "mysql+aiomysql://",
        "mariadb://": "mysql+aiomysql://",
    }
    scheme = dsn.split("://", 1)[0]
    if "+" in scheme:  # an explicit driver is already present; leave it alone
        return dsn
    for bare, async_form in prefixes.items():
        if dsn.startswith(bare):
            return async_form + dsn[len(bare) :]
    return dsn


def create_engine_from_settings(settings: Settings | None = None) -> AsyncEngine:
    """Create an :class:`AsyncEngine` from resolved settings.

    Raises if the DSN is absent (fail-closed; the config layer already enforces it).
    """
    settings = settings or get_settings()
    if settings.database_dsn is None:
        raise RuntimeError("fail-closed: WATTWISE_DATABASE_DSN is required to create an engine")
    dsn = _normalize_dsn(settings.database_dsn.get_secret_value())
    is_sqlite = dsn.startswith("sqlite")
    return create_async_engine(
        dsn,
        echo=False,
        future=True,
        # SQLite (esp. in-memory/file dev) wants a small, non-pooled footprint; the
        # server backends use a real pool with pre-ping for liveness.
        pool_pre_ping=not is_sqlite,
    )


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Create an :class:`AsyncSession` factory (expire_on_commit off for async use)."""
    return async_sessionmaker(bind=engine, expire_on_commit=False, class_=AsyncSession)


class Database:
    """Owns the engine + session factory for the process lifetime."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._engine = create_engine_from_settings(settings)
        self._session_factory = create_session_factory(self._engine)

    @property
    def engine(self) -> AsyncEngine:
        return self._engine

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        """Yield a session inside a transaction; commit on success, roll back on error."""
        async with self._session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    async def dispose(self) -> None:
        await self._engine.dispose()


__all__ = [
    "Database",
    "create_engine_from_settings",
    "create_session_factory",
]
