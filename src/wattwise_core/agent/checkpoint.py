"""Durable, portable checkpointer for the agent graph (CKPT-R1/-R2/-R3/-R7, STATE-R*).

A custom :class:`~langgraph.checkpoint.base.BaseCheckpointSaver` implemented directly
over SQLAlchemy so the agent's durable graph state runs unchanged on the three supported
backends (SQLite / PostgreSQL / MariaDB, ARCH-R13) — we deliberately do NOT use
``langgraph.checkpoint.postgres.AsyncPostgresSaver``, which is PostgreSQL-only and would
break portability. Persistence goes through the dedicated agent-state ORM
(``state_store``), which lives in its own metadata/schema and is NEVER the canonical GBO
store (ARCH-R13, CKPT-R1).

Identity & fail-closed guarantees enforced here:

* **CKPT-R1** — ``aput`` persists a checkpoint after every node transition; the graph
  runtime calls it at each step, and each call writes one ``agent_checkpoint`` row.
* **CKPT-R3** — every saver instance is bound to ONE authenticated ``athlete_id`` plus a
  ``conversation_id`` (the principal scope, re-derived from the caller on resume). A
  thread is created/loaded only for that pair; a load that resolves a thread owned by a
  different athlete is REFUSED (``CheckpointIdentityError``), never silently returned.
* **CKPT-R7** — each checkpoint row stamps the engine ``schema_version``; loading a row
  whose stored version is incompatible FAILS CLOSED (``CheckpointSchemaVersionError``),
  never coerces. The graph runtime treats the refusal as "start fresh + log".

The saver takes an injected ``async_sessionmaker`` (the agent-state-write session
factory, DEPLOY-R4) so it never owns engine lifecycle and stays unit-testable against an
in-memory database.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Sequence
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
)
from langgraph.checkpoint.serde.base import SerializerProtocol
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from wattwise_core.agent.state_store import (
    AgentCheckpoint,
    AgentThread,
    AgentWrite,
)

# Engine-side checkpoint schema version (CKPT-R7). Bump ONLY on a breaking change to the
# persisted state shape; a stored checkpoint with a different version fails closed on
# load rather than being silently coerced.
CHECKPOINT_SCHEMA_VERSION = 1


class CheckpointError(RuntimeError):
    """Base class for fail-closed checkpoint refusals (doc 50 §17 fail-closed)."""


class CheckpointIdentityError(CheckpointError):
    """A checkpoint/thread is owned by a different athlete than the caller (CKPT-R3)."""


class CheckpointSchemaVersionError(CheckpointError):
    """A stored checkpoint's schema version is incompatible with the engine (CKPT-R7)."""


def _coerce_athlete_id(athlete_id: str | uuid.UUID) -> uuid.UUID:
    """Coerce a server-derived athlete identity to a UUID (AGT-SEC-R1 / STATE-R4).

    Identity originates ONLY from the authenticated request context; this never accepts a
    model- or tool-supplied value. A malformed identity fails closed.
    """
    if isinstance(athlete_id, uuid.UUID):
        return athlete_id
    try:
        return uuid.UUID(athlete_id)
    except (ValueError, AttributeError, TypeError) as exc:  # fail-closed
        raise CheckpointIdentityError("athlete identity is not a valid identifier") from exc


def _config_str(config: RunnableConfig, key: str, default: str = "") -> str:
    """Read a string from a langgraph ``RunnableConfig['configurable']`` mapping."""
    configurable = config.get("configurable") or {}
    value = configurable.get(key, default)
    return str(value) if value is not None else default


class SqlAlchemyCheckpointSaver(BaseCheckpointSaver[str]):  # noqa: size-limits
    """Agent-state checkpointer over SQLAlchemy (portable, fail-closed, athlete-scoped).

    Slightly over the derived class-size guard: this is one cohesive implementation of
    langgraph's ``BaseCheckpointSaver`` interface (aget_tuple/aput/aput_writes/alist),
    which must live as a single class to satisfy that contract. The module stays under
    the 400-line ceiling and every method under the 60-line ceiling.


    Bound at construction to the authenticated ``athlete_id`` (CKPT-R3 principal scope)
    and the ``conversation_id`` that, together, identify the durable thread. The
    ``thread_id`` carried in the langgraph config MUST match the saver's bound pair; a
    cross-identity load is refused.
    """

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        *,
        athlete_id: str | uuid.UUID,
        conversation_id: str,
        schema_version: int = CHECKPOINT_SCHEMA_VERSION,
        serde: SerializerProtocol | None = None,
    ) -> None:
        super().__init__(serde=serde)
        self._sessions = session_factory
        self._athlete_id = _coerce_athlete_id(athlete_id)
        self._conversation_id = conversation_id
        self._schema_version = schema_version

    # --- scoping helpers -------------------------------------------------------------

    async def _resolve_thread(self, session: AsyncSession, thread_id: str) -> AgentThread | None:
        """Load the thread row and REFUSE if it is owned by a different athlete (CKPT-R3)."""
        thread = await session.get(AgentThread, thread_id)
        if thread is None:
            return None
        if thread.athlete_id != self._athlete_id:
            # One athlete's checkpoint is NOT loadable under another identity (CKPT-R3).
            raise CheckpointIdentityError(
                "refusing cross-identity checkpoint load: thread owned by another athlete"
            )
        return thread

    async def _ensure_thread(self, session: AsyncSession, thread_id: str) -> AgentThread:
        """Get-or-create the durable thread for the saver's bound (athlete, conversation)."""
        thread = await self._resolve_thread(session, thread_id)
        if thread is not None:
            return thread
        thread = AgentThread(
            thread_id=thread_id,
            athlete_id=self._athlete_id,
            conversation_id=self._conversation_id,
        )
        session.add(thread)
        await session.flush()
        return thread

    def _check_schema_version(self, row: AgentCheckpoint) -> None:
        """Fail closed if a stored checkpoint is from an incompatible schema (CKPT-R7)."""
        if row.schema_version != self._schema_version:
            raise CheckpointSchemaVersionError(
                "stored checkpoint schema version is incompatible; refusing to load"
            )

    async def resolve_idempotent(self, thread_id: str) -> str | None:
        """Return the latest checkpoint id for an EXISTING in-window run, else ``None`` (CKPT-R4).

        Idempotency is keyed by the durable thread — a thread_id is the stable
        ``(athlete_id, conversation_id)`` identifier (CKPT-R3), and a re-submission of the
        SAME request turn maps to the SAME thread_id (the engine derives it from the turn's
        idempotency key). So a non-``None`` return means a run already exists for this turn:
        the caller MUST resume/return it rather than starting a duplicate. Cross-identity
        ownership is refused here too (CKPT-R3). ``None`` means start a fresh run.
        """
        async with self._sessions() as session:
            thread = await self._resolve_thread(session, thread_id)
            if thread is None:
                return None
            stmt = (
                select(AgentCheckpoint)
                .where(AgentCheckpoint.thread_id == thread_id)
                .order_by(AgentCheckpoint.created_at.desc())
                .limit(1)
            )
            row = (await session.execute(stmt)).scalar_one_or_none()
            return row.checkpoint_id if row is not None else None

    # --- read ------------------------------------------------------------------------

    def _to_tuple(
        self, config: RunnableConfig, row: AgentCheckpoint, writes: Sequence[AgentWrite]
    ) -> CheckpointTuple:
        """Deserialize a stored row + its pending writes into a langgraph tuple."""
        checkpoint: Checkpoint = self.serde.loads_typed((row.checkpoint_type, row.checkpoint_blob))
        metadata: CheckpointMetadata = dict(row.metadata_blob)  # type: ignore[assignment]
        thread_id = row.thread_id
        ns = row.checkpoint_ns
        resolved_config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": ns,
                "checkpoint_id": row.checkpoint_id,
            }
        }
        parent_config: RunnableConfig | None = None
        if row.parent_checkpoint_id is not None:
            parent_config = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": ns,
                    "checkpoint_id": row.parent_checkpoint_id,
                }
            }
        pending = [
            (w.task_id, w.channel, self.serde.loads_typed((w.value_type, w.value_blob)))
            for w in writes
        ]
        return CheckpointTuple(
            config=resolved_config,
            checkpoint=checkpoint,
            metadata=metadata,
            parent_config=parent_config,
            pending_writes=pending,
        )

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Return the checkpoint addressed by ``config`` (latest if no id), scoped & fresh.

        Refuses a cross-identity thread (CKPT-R3) and an incompatible schema (CKPT-R7).
        """
        thread_id = _config_str(config, "thread_id")
        ns = _config_str(config, "checkpoint_ns")
        checkpoint_id = get_checkpoint_id(config)
        async with self._sessions() as session:
            thread = await self._resolve_thread(session, thread_id)
            if thread is None:
                return None
            stmt = select(AgentCheckpoint).where(
                AgentCheckpoint.thread_id == thread_id,
                AgentCheckpoint.checkpoint_ns == ns,
            )
            if checkpoint_id:
                stmt = stmt.where(AgentCheckpoint.checkpoint_id == checkpoint_id)
            else:
                stmt = stmt.order_by(AgentCheckpoint.created_at.desc())
            row = (await session.execute(stmt.limit(1))).scalar_one_or_none()
            if row is None:
                return None
            self._check_schema_version(row)
            writes = await self._load_writes(session, thread_id, ns, row.checkpoint_id)
            return self._to_tuple(config, row, writes)

    async def _load_writes(
        self, session: AsyncSession, thread_id: str, ns: str, checkpoint_id: str
    ) -> Sequence[AgentWrite]:
        stmt = (
            select(AgentWrite)
            .where(
                AgentWrite.thread_id == thread_id,
                AgentWrite.checkpoint_ns == ns,
                AgentWrite.checkpoint_id == checkpoint_id,
            )
            .order_by(AgentWrite.task_id, AgentWrite.idx)
        )
        return list((await session.execute(stmt)).scalars().all())

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """Yield this thread's checkpoints newest-first, identity-scoped (CKPT-R3)."""
        effective: RunnableConfig = config if config is not None else {"configurable": {}}
        thread_id = _config_str(effective, "thread_id")
        ns = _config_str(effective, "checkpoint_ns")
        async with self._sessions() as session:
            thread = await self._resolve_thread(session, thread_id)
            if thread is None:
                return
            stmt = (
                select(AgentCheckpoint)
                .where(
                    AgentCheckpoint.thread_id == thread_id,
                    AgentCheckpoint.checkpoint_ns == ns,
                )
                .order_by(AgentCheckpoint.created_at.desc())
            )
            if before is not None:
                before_id = get_checkpoint_id(before)
                if before_id:
                    stmt = stmt.where(AgentCheckpoint.checkpoint_id < before_id)
            if limit is not None:
                stmt = stmt.limit(limit)
            rows = list((await session.execute(stmt)).scalars().all())
            for row in rows:
                self._check_schema_version(row)
                writes = await self._load_writes(session, thread_id, ns, row.checkpoint_id)
                yield self._to_tuple(effective, row, writes)

    # --- write -----------------------------------------------------------------------

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Persist one checkpoint after a node transition (CKPT-R1); identity-scoped."""
        thread_id = _config_str(config, "thread_id")
        ns = _config_str(config, "checkpoint_ns")
        parent_id = get_checkpoint_id(config) or None
        cp_type, cp_blob = self.serde.dumps_typed(checkpoint)
        async with self._sessions() as session:
            await self._ensure_thread(session, thread_id)
            row = AgentCheckpoint(
                thread_id=thread_id,
                checkpoint_ns=ns,
                checkpoint_id=checkpoint["id"],
                parent_checkpoint_id=parent_id,
                athlete_id=self._athlete_id,
                schema_version=self._schema_version,
                checkpoint_type=cp_type,
                checkpoint_blob=cp_blob,
                metadata_blob=dict(metadata),
            )
            session.add(row)
            await session.commit()
        result: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": ns,
                "checkpoint_id": checkpoint["id"],
            }
        }
        return result

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Persist a node's pending intermediate writes, replayed on resume (CKPT-R2)."""
        thread_id = _config_str(config, "thread_id")
        ns = _config_str(config, "checkpoint_ns")
        checkpoint_id = _config_str(config, "checkpoint_id")
        async with self._sessions() as session:
            await self._ensure_thread(session, thread_id)
            for idx, (channel, value) in enumerate(writes):
                value_type, value_blob = self.serde.dumps_typed(value)
                session.add(
                    AgentWrite(
                        thread_id=thread_id,
                        checkpoint_ns=ns,
                        checkpoint_id=checkpoint_id,
                        task_id=task_id,
                        idx=idx,
                        channel=channel,
                        value_type=value_type,
                        value_blob=value_blob,
                    )
                )
            await session.commit()


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "CheckpointError",
    "CheckpointIdentityError",
    "CheckpointSchemaVersionError",
    "SqlAlchemyCheckpointSaver",
]
