"""Durable, portable checkpointer for the agent graph (CKPT-R1/-R2/-R3/-R7/-R9, STATE-R*).

A custom :class:`~langgraph.checkpoint.base.BaseCheckpointSaver` implemented directly over
SQLAlchemy so the agent's durable graph state runs unchanged on the three supported backends
(SQLite / PostgreSQL / MariaDB, ARCH-R13) — deliberately NOT the PostgreSQL-only
``AsyncPostgresSaver``. Persistence goes through the dedicated agent-state ORM (``state_store``),
which lives in its own metadata/schema and is NEVER the canonical GBO store (ARCH-R13, CKPT-R1).

Identity & fail-closed guarantees (each detailed on the relevant method): ``aput`` persists after
every node transition (CKPT-R1); every saver is bound to ONE authenticated
``(athlete_id, conversation_id)`` and a cross-identity load is REFUSED (CKPT-R3); an incompatible
``schema_version`` fails closed (CKPT-R7); the HITL ledger (``record_interrupt`` /
``consume_interrupt`` / ``interrupt_status``) gates resume on an atomic guarded UPDATE (CKPT-R9).
The injected ``async_sessionmaker`` (DEPLOY-R4) means the saver never owns engine lifecycle.
"""

from __future__ import annotations

import contextlib
import uuid
from collections.abc import AsyncIterator, Sequence
from typing import Any, Literal, cast

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
)
from langgraph.checkpoint.serde.base import SerializerProtocol
from sqlalchemy import Table, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from wattwise_core.agent import checkpoint_interrupts as ledger
from wattwise_core.agent.redaction import (
    IDENTITY_CHANNELS,
    redact_checkpoint,
    redact_state_payload,
)
from wattwise_core.agent.state_store import (
    AgentCheckpoint,
    AgentThread,
    AgentWrite,
)
from wattwise_core.observability.logging import get_logger
from wattwise_core.persistence.types import uuid7
from wattwise_core.persistence.upsert import ensure_row, upsert

_logger = get_logger(__name__)

# Engine-side checkpoint schema version (CKPT-R7). Bump ONLY on a breaking change to the
# persisted state shape; a stored checkpoint with a different version fails closed on
# load rather than being silently coerced.
#
# v2 (D-P2): the turn-boundary protocol reshaped AgentState (turn_id/run_epoch channels,
# _turn_monotonic decrease-to-floor counters, turn-keyed retrieved/coverage_gaps). A v1
# checkpoint lacks those channels, so it MUST fail closed and start fresh, never be coerced
# into the new shape (CheckpointSchemaVersionError; F-SCHEMA-BUMP).
CHECKPOINT_SCHEMA_VERSION = 2


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

    Over the derived class-size guard: one cohesive implementation of langgraph's
    ``BaseCheckpointSaver`` interface (aget_tuple/aput/aput_writes/alist) plus the HITL
    interrupt ledger (record/consume), which must live as a single class to satisfy that
    contract; every method stays under the 60-line ceiling.

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
        """Get-or-create the durable thread for the saver's bound (athlete, conversation).

        Concurrency-safe (CKPT-R1): a single graph run makes the langgraph runtime call
        ``aput``/``aput_writes`` on SEPARATE sessions/connections that race to create the
        thread. The create therefore goes through the sanctioned atomic upsert seam's
        :func:`~wattwise_core.persistence.upsert.ensure_row` (UPS-R2) keyed on
        ``thread_id``: ONE atomic insert-or-ignore in its own short transaction, so both
        racers succeed — never a plain ``INSERT`` whose loser raises (PostgreSQL/SQLite
        unique violation; MariaDB under ``innodb_snapshot_isolation`` surfaces the race
        as error 1020, which is NOT an ``IntegrityError``, so a catch-and-retry seam
        cannot be the mechanism — the seam removes the failure at the root instead).
        After the ensure the row is resolved via ``_resolve_thread`` so a
        cross-identity row is REFUSED, never adopted (CKPT-R3); the caller only needs
        the row to EXIST (it discards the return value).

        Ordering matters: the ensure runs FIRST, before this method touches ``session``
        (every caller invokes it as the session's first statement). Resolving first and
        ensuring on a miss would hold the caller's pooled connection while waiting for
        the ensure's second one — under N concurrent first-touches that exhausts a
        bounded pool (every holder waits on an empty pool: deadlock-by-timeout). With
        ensure-first each task holds at most one connection at a time, and the caller's
        snapshot is created after the ensure commit, so the re-resolve sees the row.
        """
        # Atomic insert-or-ignore on the thread natural key (UPS-R2): concurrent
        # ensures are resolved by the DATABASE in one statement, not by exceptions.
        # Mitigation: under rare PostgreSQL timing races the existing row may carry a
        # different thread_id (e.g. an empty string when config["thread_id"] is None),
        # so the ON CONFLICT (thread_id) clause does not catch the
        # (athlete_id, conversation_id) unique-constraint violation. Swallowing that
        # specific IntegrityError and proceeding to the post-insert verification is safe
        # because _resolve_thread below fail-closed on a missing row (CKPT-R3).
        with contextlib.suppress(IntegrityError):
            await ensure_row(
                self._sessions,
                cast(Table, AgentThread.__table__),
                {
                    "thread_id": thread_id,
                    "athlete_id": self._athlete_id,
                    "conversation_id": self._conversation_id,
                },
                conflict_keys=["thread_id"],
            )
        existing = await self._resolve_thread(session, thread_id)
        if existing is None:  # fail closed: the row must exist after an atomic upsert
            raise CheckpointError(
                "agent_thread row absent after atomic ensure-thread upsert; refusing to proceed"
            )
        return existing

    def _check_schema_version(self, row: AgentCheckpoint) -> None:
        """Fail closed if a stored checkpoint is from an incompatible schema (CKPT-R7)."""
        if row.schema_version != self._schema_version:
            raise CheckpointSchemaVersionError(
                "stored checkpoint schema version is incompatible; refusing to load"
            )

    def _guard_schema_version(self, row: AgentCheckpoint) -> bool:
        """Check schema version; log and return ``False`` on mismatch instead of raising."""
        try:
            self._check_schema_version(row)
        except CheckpointSchemaVersionError as exc:
            _logger.warning(
                "checkpoint_schema_bump",
                thread_id=row.thread_id,
                checkpoint_id=row.checkpoint_id,
                stored_version=row.schema_version,
                expected_version=self._schema_version,
                exc_info=str(exc),
            )
            return False
        return True

    async def resolve_idempotent(self, thread_id: str) -> str | None:
        """Return the latest checkpoint id for an EXISTING in-window run, else ``None`` (CKPT-R4).

        Idempotency is keyed by the durable thread (the stable ``(athlete_id, conversation_id)``
        id, CKPT-R3); a re-submission of the SAME turn maps to the SAME thread_id (engine-derived
        from the turn's idempotency key). A non-``None`` return means a run already exists for this
        turn — the caller MUST resume it, not start a duplicate; cross-identity ownership is refused
        (CKPT-R3); ``None`` means start fresh.
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
            if row is None or not self._guard_schema_version(row):
                return None  # CKPT-R7: start fresh rather than coerce an incompatible schema
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
                if not self._guard_schema_version(row):
                    continue  # CKPT-R7: skip stale checkpoints, start fresh from the next
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
        """Persist one checkpoint after a node transition (CKPT-R1); identity-scoped.

        PII in the checkpointed state (the athlete's ``messages``/``request_text`` and the
        composed ``draft``/``grounded_text`` — special-category content, MEM-R3) is masked
        through the central redactor BEFORE the blob is serialized, so the persisted bytes
        carry no unmasked PII (AGT-SEC-R4 "redacted ... before persistence", CKPT-R8 §10).
        Redaction is unconditional on the persistence path (fail-closed) — durable state is
        never written raw — and only masks high-confidence PII/secret spans, so the blob
        still deserializes and resume stays identical (CKPT-R2).
        """
        thread_id = _config_str(config, "thread_id")
        ns = _config_str(config, "checkpoint_ns")
        parent_id = get_checkpoint_id(config) or None
        cp_type, cp_blob = self.serde.dumps_typed(redact_checkpoint(checkpoint))
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
        conf = {"thread_id": thread_id, "checkpoint_ns": ns, "checkpoint_id": checkpoint["id"]}
        return {"configurable": conf}

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Persist a node's pending intermediate writes, replayed on resume (CKPT-R2).

        Special langgraph channels (``__resume__``/``__interrupt__``/``__error__``/
        ``__scheduled__``) are keyed by their reserved negative ``idx`` from
        ``WRITES_IDX_MAP`` (as langgraph's reference saver does), so a ``__resume__`` write
        (idx -4, the human HITL decision) can NEVER collide at positional idx 0 with a branch
        write and be silently dropped. The natural key ``(thread, ns, checkpoint, task, idx)``
        is upserted last-write-wins through the sanctioned dialect seam, so a re-delivered
        write of the SAME channel overwrites rather than being ignored (portable, CKPT-R2).
        """
        thread_id = _config_str(config, "thread_id")
        ns = _config_str(config, "checkpoint_ns")
        checkpoint_id = _config_str(config, "checkpoint_id")
        async with self._sessions() as session:
            await self._ensure_thread(session, thread_id)
            for idx, (channel, value) in enumerate(writes):
                write_idx = WRITES_IDX_MAP.get(channel, idx)
                # Mask PII in the pending intermediate write before it is serialized, so a
                # node's not-yet-checkpointed output (which may carry the athlete's words or
                # composed prose) is never persisted raw (AGT-SEC-R4 / CKPT-R8). An IDENTITY
                # channel (athlete_id/thread_id/turn_id/...) is left verbatim — it is an opaque
                # internal identifier, not PII, and masking it would corrupt durable scoping
                # (CKPT-R3). Redaction only masks high-confidence PII spans, so the replayed
                # write (CKPT-R2) keeps its shape and type.
                masked = value if channel in IDENTITY_CHANNELS else redact_state_payload(value)
                value_type, value_blob = self.serde.dumps_typed(masked)
                await upsert(
                    session,
                    cast(Table, AgentWrite.__table__),  # ORM table is a Table at runtime
                    {
                        "id": uuid7(),
                        "thread_id": thread_id,
                        "checkpoint_ns": ns,
                        "checkpoint_id": checkpoint_id,
                        "task_id": task_id,
                        "idx": write_idx,
                        "channel": channel,
                        "value_type": value_type,
                        "value_blob": value_blob,
                    },
                    conflict_keys=["thread_id", "checkpoint_ns", "checkpoint_id", "task_id", "idx"],
                    update_columns=["channel", "value_type", "value_blob"],
                )
            await session.commit()

    # --- HITL approval-gate ledger (CKPT-R9 / D-P2) ----------------------------------
    # The three ledger operations live in :mod:`checkpoint_interrupts` (QUAL-R9 size split); the
    # saver binds its identity/session-factory + thread get-or-create into them so each guard stays
    # athlete-scoped (CKPT-R3). The seam names are unchanged so every caller path stays stable.

    async def record_interrupt(self, thread_id: str, interrupt_id: str) -> None:
        """Record a ``live`` approval-gate interrupt row, idempotently (CKPT-R9)."""
        await ledger.record_interrupt(
            self._sessions, self._ensure_thread, self._athlete_id, thread_id, interrupt_id
        )

    async def consume_interrupt(self, thread_id: str, interrupt_id: str) -> bool:
        """Atomically consume a ``live`` interrupt; True ⇒ resume, False ⇒ 404/409 (CKPT-R9)."""
        return await ledger.consume_interrupt(
            self._sessions, self._athlete_id, thread_id, interrupt_id
        )

    async def interrupt_status(
        self, thread_id: str, interrupt_id: str
    ) -> Literal["unknown", "live", "consumed"]:
        """Athlete-scoped status of an interrupt for the 404-vs-409 split (CKPT-R9, API-R12a)."""
        return await ledger.interrupt_status(
            self._sessions, self._athlete_id, thread_id, interrupt_id
        )


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "CheckpointError",
    "CheckpointIdentityError",
    "CheckpointSchemaVersionError",
    "SqlAlchemyCheckpointSaver",
]
