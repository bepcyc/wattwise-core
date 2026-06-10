"""Async engine + session factory, DSN-only across SQLite/PostgreSQL/MariaDB (GBO-R8b).

The backend is selected purely by the DSN (``WATTWISE_DATABASE_DSN``); no application
code branches on which backend is in use (the only dialect awareness lives in the
upsert seam). The engine targets ONLY the schema the migrations create — it never
connects to a pre-existing/host database (TASK data-safety rule).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy import event
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from wattwise_core.config import Settings, get_settings


def enable_sqlite_foreign_keys(engine: AsyncEngine) -> None:
    """Enforce foreign keys on SQLite (GBO-R8b, GBO-AC-7, TEN-R1).

    SQLite defaults FK enforcement OFF per connection, so an orphan personal row that
    PostgreSQL/MariaDB reject would be silently accepted — breaking the "runs unchanged
    DSN-only" parity. Issue ``PRAGMA foreign_keys=ON`` on every new DBAPI connection.
    """

    @event.listens_for(engine.sync_engine, "connect")
    def _set_pragma(dbapi_connection: object, _record: object) -> None:
        cursor = dbapi_connection.cursor()  # type: ignore[attr-defined]
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


def normalize_dsn(dsn: str) -> str:
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


def create_engine_from_settings(
    settings: Settings | None = None, *, dsn: str | None = None
) -> AsyncEngine:
    """Create an :class:`AsyncEngine` from resolved settings.

    Raises if the DSN is absent (fail-closed; the config layer already enforces it).
    ``dsn`` overrides the settings-resolved canonical DSN so a deployment can hand a layer
    its OWN per-write-domain role credential (DEPLOY-R4) while reusing the same engine
    construction (DSN-only backend selection, SQLite FK pragma, pooling).
    """
    if dsn is None:
        settings = settings or get_settings()
        if settings.database_dsn is None:
            raise RuntimeError(
                "fail-closed: WATTWISE_DATABASE_DSN is required to create an engine"
            )
        dsn = settings.database_dsn.get_secret_value()
    dsn = normalize_dsn(dsn)
    is_sqlite = dsn.startswith("sqlite")
    engine = create_async_engine(
        dsn,
        echo=False,
        future=True,
        # SQLite (esp. in-memory/file dev) wants a small, non-pooled footprint; the
        # server backends use a real pool with pre-ping for liveness.
        pool_pre_ping=not is_sqlite,
    )
    if is_sqlite:
        enable_sqlite_foreign_keys(engine)
    return engine


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Create an :class:`AsyncSession` factory (expire_on_commit off for async use)."""
    return async_sessionmaker(bind=engine, expire_on_commit=False, class_=AsyncSession)


class Database:
    """Owns the engine + session factory for the process lifetime.

    ``dsn`` (optional) binds this Database to a specific per-write-domain role credential
    (DEPLOY-R4) instead of the settings-resolved canonical DSN — e.g. the API's
    master-data-write role (ARCH-R3b). Construction is otherwise identical (DSN-only).
    """

    def __init__(self, settings: Settings | None = None, *, dsn: str | None = None) -> None:
        self._engine = create_engine_from_settings(settings, dsn=dsn)
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
    "enable_sqlite_foreign_keys",
    "normalize_dsn",
]
