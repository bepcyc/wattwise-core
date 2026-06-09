"""ORM models for the dedicated agent-state store (ARCH-R13, CKPT-R*, STATE-R*).

The agent orchestrator's durable graph state — checkpoints, threads, and pending
writes — lives in a store that is **NEVER the canonical GBO store** (doc 10 ARCH-R13,
doc 50 §4 CKPT-R1). To make that separation *structural* rather than a convention, the
models here declare their own :class:`~sqlalchemy.MetaData` on a private
:class:`AgentStateBase` — they are deliberately NOT registered on the canonical
``wattwise_core.persistence.base.Base``. The store-separation test (ARCH-R29) therefore
holds by construction: no agent-state table can ever appear in the canonical metadata,
and the two metadatas can be granted distinct write roles (DEPLOY-R4).

Portability (ARCH-R13: SQLite / PostgreSQL / MariaDB, DSN-only) is preserved by reusing
the same portable column factories the canonical layer uses — ``sa.Uuid``,
``DateTime(timezone=True)``, portable ``JSON`` — and a stable naming convention so the
single agent-state migration (``0002_agent_state``) renders identically on all three
backends.

Tables (logical ``agent_state`` category, ARCH-R13):

* ``agent_thread`` — one durable conversation thread, scoped write-once to
  ``(athlete_id, conversation_id)`` (CKPT-R3). The owning ``athlete_id`` is the only
  identity a checkpoint may be loaded under; a cross-identity load is refused.
* ``agent_checkpoint`` — one persisted graph checkpoint per ``(thread, checkpoint_id)``
  written after **every** node transition (CKPT-R1), carrying the serialized state blob,
  its ``schema_version`` (CKPT-R7 fail-closed on mismatch), the parent checkpoint id
  (resume lineage, CKPT-R2), and the owning ``athlete_id`` (defence-in-depth scoping).
* ``agent_write`` — one pending intermediate write produced by a node, replayed on
  resume so a mid-flight-killed run reconstructs identically (CKPT-R2).
* ``agent_interrupt`` — one human-in-the-loop approval-gate ledger row (CKPT-R9). When
  the graph raises a langgraph interrupt at an approval-gated plan, the interrupt-gate
  records a ``live`` row; a ``POST …/decision`` consumes it via an ATOMIC guarded UPDATE
  (``SET status='consumed' WHERE thread_id=? AND interrupt_id=? AND athlete_id=? AND
  status='live'``) whose ``rowcount`` decides resume-vs-409 (fail-closed, CKPT-R9). The
  owning ``athlete_id`` is duplicated (as on ``agent_checkpoint``) as defence-in-depth so
  a row is independently identity-scoped and joins the per-athlete erasure target set
  (CKPT-R8 / PRIV-R8).
"""

from __future__ import annotations

import datetime as _dt
import uuid

from sqlalchemy import (
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    MetaData,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import JSON

from wattwise_core.persistence.base import NAMING_CONVENTION
from wattwise_core.persistence.types import UtcDateTime, utcnow, uuid7

# Logical agent-state category name (ARCH-R13). Used as the table-name prefix so the
# agent-state tables are unmistakable and never collide with canonical tables, while
# staying portable to SQLite (which has no real schemas).
AGENT_STATE_PREFIX = "agent_"


class AgentStateBase(DeclarativeBase):
    """Declarative base for the agent-state store ONLY (ARCH-R13).

    Deliberately distinct from the canonical ``Base``: agent state and canonical master
    data MUST NOT share a metadata/schema/write credential (doc 10 ARCH-R13, DEPLOY-R4).
    """

    metadata = MetaData(naming_convention=NAMING_CONVENTION)


class AgentThread(AgentStateBase):
    """One durable conversation thread scoped to ``(athlete_id, conversation_id)``.

    CKPT-R3: thread identity is write-once and athlete-scoped. ``athlete_id`` is the
    authenticated owner re-derived from the caller on every resume; a checkpoint stored
    under one athlete is NEVER loadable under another (enforced by the checkpointer,
    which refuses a load whose caller identity mismatches this row). The
    ``(athlete_id, conversation_id)`` pair is UNIQUE so a thread maps to exactly one
    owner.
    """

    __tablename__ = AGENT_STATE_PREFIX + "thread"
    __table_args__ = (
        UniqueConstraint(
            "athlete_id",
            "conversation_id",
            name="uq_agent_thread_athlete_conversation",
        ),
    )

    thread_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    athlete_id: Mapped[uuid.UUID] = mapped_column(nullable=False, index=True)
    conversation_id: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[_dt.datetime] = mapped_column(
        UtcDateTime(), default=utcnow, nullable=False
    )


class AgentCheckpoint(AgentStateBase):
    """One persisted graph checkpoint (CKPT-R1: after EVERY node transition).

    A checkpoint is addressed by ``(thread_id, checkpoint_ns, checkpoint_id)`` and links
    to its ``parent_checkpoint_id`` for resume lineage (CKPT-R2). ``schema_version`` is
    stamped at write time; a load under an incompatible engine schema version fails
    closed (CKPT-R7) rather than silently coercing the blob. ``athlete_id`` is duplicated
    here as defence-in-depth so a row is independently identity-bound even if read
    outside the thread join.
    """

    __tablename__ = AGENT_STATE_PREFIX + "checkpoint"
    __table_args__ = (
        UniqueConstraint(
            "thread_id",
            "checkpoint_ns",
            "checkpoint_id",
            name="uq_agent_checkpoint_thread_ns_id",
        ),
        Index(
            "ix_agent_checkpoint_thread_ns_created",
            "thread_id",
            "checkpoint_ns",
            "created_at",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid7)
    thread_id: Mapped[str] = mapped_column(
        ForeignKey(AGENT_STATE_PREFIX + "thread.thread_id"),
        nullable=False,
        index=True,
    )
    checkpoint_ns: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    checkpoint_id: Mapped[str] = mapped_column(String(128), nullable=False)
    parent_checkpoint_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    athlete_id: Mapped[uuid.UUID] = mapped_column(nullable=False, index=True)
    schema_version: Mapped[int] = mapped_column(Integer, nullable=False)
    # Serialized checkpoint + metadata. ``*_type`` carry the serializer's content-type
    # tag so deserialization is lossless and never guesses an encoding (fail-closed).
    checkpoint_type: Mapped[str] = mapped_column(String(64), nullable=False)
    checkpoint_blob: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    metadata_blob: Mapped[dict[str, object]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[_dt.datetime] = mapped_column(
        UtcDateTime(), default=utcnow, nullable=False
    )


class AgentWrite(AgentStateBase):
    """One pending intermediate write emitted by a node, replayed on resume (CKPT-R2).

    Keyed by ``(thread_id, checkpoint_ns, checkpoint_id, task_id, idx)`` so a node's
    writes are ordered and idempotent under re-delivery. Each carries a serialized value
    with its content-type tag, mirroring ``agent_checkpoint``.
    """

    __tablename__ = AGENT_STATE_PREFIX + "write"
    __table_args__ = (
        UniqueConstraint(
            "thread_id",
            "checkpoint_ns",
            "checkpoint_id",
            "task_id",
            "idx",
            name="uq_agent_write_identity",
        ),
        Index(
            "ix_agent_write_checkpoint",
            "thread_id",
            "checkpoint_ns",
            "checkpoint_id",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid7)
    thread_id: Mapped[str] = mapped_column(
        ForeignKey(AGENT_STATE_PREFIX + "thread.thread_id"),
        nullable=False,
    )
    checkpoint_ns: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    checkpoint_id: Mapped[str] = mapped_column(String(128), nullable=False)
    task_id: Mapped[str] = mapped_column(String(128), nullable=False)
    idx: Mapped[int] = mapped_column(Integer, nullable=False)
    channel: Mapped[str] = mapped_column(String(255), nullable=False)
    value_type: Mapped[str] = mapped_column(String(64), nullable=False)
    value_blob: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    created_at: Mapped[_dt.datetime] = mapped_column(
        UtcDateTime(), default=utcnow, nullable=False
    )


class AgentInterrupt(AgentStateBase):
    """One HITL approval-gate ledger row, live until consumed (CKPT-R9 / D-P2).

    The interrupt-gate inserts a ``live`` row when the graph raises a langgraph interrupt
    at an approval-gated plan; ``POST …/decision`` then consumes it via the ATOMIC guarded
    UPDATE ``SET status='consumed' WHERE thread_id=? AND interrupt_id=? AND athlete_id=?
    AND status='live'`` whose ``rowcount`` decides resume (1) vs 409/404 (0) — fail-closed
    so a double-decision or cross-identity attempt can never resume twice. The
    ``(thread_id, interrupt_id)`` pair is UNIQUE so a gate raises exactly one live row per
    interrupt. ``athlete_id`` is duplicated here (as on ``agent_checkpoint``) as
    defence-in-depth: the consume guard is independently identity-scoped (CKPT-R9) and the
    row joins the per-athlete erasure target set (CKPT-R8 / PRIV-R8) even outside the
    thread join.
    """

    __tablename__ = AGENT_STATE_PREFIX + "interrupt"
    __table_args__ = (
        UniqueConstraint(
            "thread_id",
            "interrupt_id",
            name="uq_agent_interrupt_thread_interrupt",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid7)
    thread_id: Mapped[str] = mapped_column(
        String(128),
        ForeignKey(AGENT_STATE_PREFIX + "thread.thread_id"),
        nullable=False,
        index=True,
    )
    athlete_id: Mapped[uuid.UUID] = mapped_column(nullable=False, index=True)
    interrupt_id: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="live")
    created_at: Mapped[_dt.datetime] = mapped_column(
        UtcDateTime(), default=utcnow, nullable=False
    )


__all__ = [
    "AGENT_STATE_PREFIX",
    "AgentCheckpoint",
    "AgentInterrupt",
    "AgentStateBase",
    "AgentThread",
    "AgentWrite",
]
