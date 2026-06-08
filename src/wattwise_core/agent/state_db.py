"""Dedicated engine + connection pool for the agent-state store (ARCH-R13, DEPLOY-R4).

The durable-agent-state store (checkpoints, threads, pending writes, memory) MUST own
an engine/pool that is **NEVER shared with the canonical GBO** ``Database``
(``persistence.engine``). This is not only a separation-of-concerns nicety — it is a
deadlock-safety property (SPIKE-3):

A single durable graph run holds a canonical-store connection (the run's own
transaction) WHILE the checkpointer (``SqlAlchemyCheckpointSaver``) concurrently needs
its OWN connection to persist a checkpoint after every node transition (CKPT-R1). If
both connections are drawn from the SAME pool at a low ``pool_size``, the saver can
block forever waiting for a connection the run is still holding — a pool-exhaustion
DEADLOCK. Giving agent-state its own engine/pool removes that coupling entirely: the
two never contend for the same connections. (DEPLOY-R4 additionally lets agent-state
carry its own write credential, hence the optional distinct DSN below.)

This mirrors the canonical ``Database`` (same DSN normalization, sqlite ``PRAGMA
foreign_keys=ON``, server-backend ``pool_pre_ping``) and reuses its helpers, but is a
SEPARATE object owning a SEPARATE :class:`AsyncEngine`. It is intentionally NOT wired
into the production engine here (a later step does that); this module only provides the
class and a builder.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.engine import make_url
from sqlalchemy.exc import ArgumentError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import StaticPool

# Importing ``memory`` here (not only ``state_store``) registers ``agent_memory_item`` on
# ``AgentStateBase.metadata`` as an import side effect, so ``create_all`` below emits the
# FULL agent-state schema — mirroring the checkpoint tests, which import it for the same
# reason. ``state_store`` defines AgentThread/Checkpoint/Write; ``memory`` adds the memory
# table; both must be on the metadata before ``create_all``.
import wattwise_core.agent.memory  # noqa: F401  (registers agent_memory_item)
from wattwise_core.agent.state_store import AgentStateBase
from wattwise_core.config import Settings, get_settings
from wattwise_core.persistence.engine import (
    create_session_factory,
    enable_sqlite_foreign_keys,
    normalize_dsn,
)


def _is_memory_sqlite(dsn: str) -> bool:
    """True for an in-memory sqlite DSN (``:memory:`` or the empty-path form).

    Both ``sqlite+aiosqlite:///:memory:`` and the bare ``sqlite+aiosqlite://`` resolve
    to a per-connection ephemeral database, so each new pooled connection would open its
    OWN empty DB. The agent-state saver opens many short-lived sessions, so it MUST see a
    single shared in-memory DB — hence :class:`StaticPool` (one reused connection) below.

    Decided by PARSING the URL (``make_url(...).database``), not substring matching: the
    bare form has an empty/``None`` database while ``:memory:`` has database ``":memory:"``;
    a real file path that merely CONTAINS the literal ``:memory:`` (e.g.
    ``sqlite+aiosqlite:////data/db_:memory:_backup.sqlite``) parses to that file path and is
    correctly NOT in-memory. Falls closed to ``False`` if the DSN cannot be parsed.
    """
    if not dsn.startswith("sqlite"):
        return False
    try:
        database = make_url(dsn).database
    except ArgumentError:
        return False
    return database in (None, "", ":memory:")


def create_agent_state_engine(
    settings: Settings | None = None,
    *,
    dsn: str | None = None,
    pool_size: int | None = None,
    max_overflow: int | None = None,
) -> AsyncEngine:
    """Create the SEPARATE :class:`AsyncEngine` for the agent-state store (ARCH-R13).

    Resolves an agent-state DSN: an explicit ``dsn`` argument wins (DEPLOY-R4 lets the
    agent-state store carry its own write credential, distinct from the canonical one),
    otherwise it falls back to the canonical ``settings.database_dsn``. EITHER way this
    is an engine/pool DISTINCT from the canonical ``Database`` engine — the deadlock
    safety (SPIKE-3) comes from the separate pool, not from the DSN differing. Fails
    closed if no DSN is resolvable.

    Pooling:
      * sqlite ``:memory:`` -> :class:`StaticPool` + ``check_same_thread=False`` so a
        single shared in-memory DB backs the saver's many sessions (without it each
        aiosqlite connection gets its own empty ``:memory:`` and the saver breaks);
      * file-sqlite + server DSNs -> the default pool, with ``pool_pre_ping`` on the
        server backends for liveness (mirroring the canonical engine).

    ``pool_size`` / ``max_overflow`` are optional tunables for the non-memory branch
    (a deployment may cap the agent-state pool independently of the canonical one). They
    are also what the SPIKE-3 dedicated-pool test uses to build the agent side at
    ``pool_size=1, max_overflow=0`` so that the ONLY difference from the shared arm is
    pool SEPARATION (not spare default capacity). They do not apply to the StaticPool
    ``:memory:`` branch, which is always a single shared connection.
    """
    raw_dsn = dsn
    if raw_dsn is None:
        settings = settings or get_settings()
        dsn_secret = settings.database_dsn
        if dsn_secret is None:
            raise RuntimeError(
                "fail-closed: a DSN is required to create the agent-state engine (ARCH-R13)"
            )
        raw_dsn = dsn_secret.get_secret_value()
    dsn = normalize_dsn(raw_dsn)
    is_sqlite = dsn.startswith("sqlite")
    if _is_memory_sqlite(dsn):
        engine = create_async_engine(
            dsn,
            echo=False,
            future=True,
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
        )
    else:
        pool_kwargs: dict[str, int] = {}
        if pool_size is not None:
            pool_kwargs["pool_size"] = pool_size
        if max_overflow is not None:
            pool_kwargs["max_overflow"] = max_overflow
        engine = create_async_engine(
            dsn,
            echo=False,
            future=True,
            pool_pre_ping=not is_sqlite,
            **pool_kwargs,
        )
    if is_sqlite:
        enable_sqlite_foreign_keys(engine)
    return engine


class AgentStateDatabase:
    """Owns the SEPARATE engine + session factory for the agent-state store.

    Distinct from the canonical :class:`~wattwise_core.persistence.engine.Database`
    (ARCH-R13): the two never share an engine/pool, which is what makes a durable graph
    run + its checkpointer deadlock-free even at ``pool_size=1`` (SPIKE-3, DEPLOY-R4).
    The ``session_factory`` here is exactly what ``SqlAlchemyCheckpointSaver`` takes
    (CKPT-R1).
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        dsn: str | None = None,
        pool_size: int | None = None,
        max_overflow: int | None = None,
    ) -> None:
        self._engine = create_agent_state_engine(
            settings, dsn=dsn, pool_size=pool_size, max_overflow=max_overflow
        )
        self._session_factory = create_session_factory(self._engine)

    @property
    def engine(self) -> AsyncEngine:
        return self._engine

    @property
    def session_factory(self) -> async_sessionmaker[AsyncSession]:
        """The agent-state-write session factory the checkpointer is constructed with."""
        return self._session_factory

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

    async def create_all(self) -> None:
        """Create the agent-state schema on this engine (ARCH-R13 dedicated store).

        Targets ONLY the dedicated agent-state metadata (AgentThread/Checkpoint/Write +
        the memory table registered by the module-level ``memory`` import) — never the
        canonical ``Base`` (the store-separation guarantee, ARCH-R29).
        """
        async with self._engine.begin() as conn:
            await conn.run_sync(AgentStateBase.metadata.create_all)

    async def dispose(self) -> None:
        await self._engine.dispose()


def build_agent_state_database(
    settings: Settings | None = None,
    *,
    dsn: str | None = None,
    pool_size: int | None = None,
    max_overflow: int | None = None,
) -> AgentStateDatabase:
    """Build the dedicated agent-state database from settings (ARCH-R13, DEPLOY-R4).

    ``dsn`` optionally overrides the canonical DSN (DEPLOY-R4: a distinct agent-state
    write credential); when omitted the canonical ``database_dsn`` is reused, but ALWAYS
    on a separate engine/pool. ``pool_size`` / ``max_overflow`` optionally cap the
    non-memory pool (see :func:`create_agent_state_engine`).
    """
    return AgentStateDatabase(
        settings, dsn=dsn, pool_size=pool_size, max_overflow=max_overflow
    )


__all__ = [
    "AgentStateDatabase",
    "build_agent_state_database",
    "create_agent_state_engine",
]
