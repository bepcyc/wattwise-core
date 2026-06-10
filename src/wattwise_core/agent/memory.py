"""Durable athlete memory: the MemoryStore seam + its OSS relational implementation.

Cited requirements (doc 50): MEM-R1 (scope — personalization only, NEVER a canonical
number), MEM-R2 (ground-truth-preserving raw episodes; inferred items marked),
MEM-R3 (retention/erasure + untrusted content may NOT write memory), MEM-R4 (the
``MemoryStore`` recall seam; OSS impl = athlete-scoped recency/keyword query over a
relational table; no mandatory vector DB), MEM-R5 (the closed ``memory_item_kind``
enum, owned here). Also AGT-OBS-R5a (a blocked untrusted write is a typed anomaly).

The store holds preferences/goals/constraints/episodes in the athlete's own words; it
is structurally unable to hold an analytic number (no numeric metric field exists),
so a canonical value (CTL/TSS/W'/HRV/...) is always read LIVE from the analytics
service (doc 40), never substituted from memory (MEM-R1, EVAL-R2a). Memory rows live
in the dedicated agent-state store, scoped to the owner ``athlete_id`` (MEM-R3); they
are never the canonical GBO store and never a source of analytic ground truth.
"""

from __future__ import annotations

import datetime as _dt
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

from sqlalchemy import String, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from wattwise_core.agent.state_store import AGENT_STATE_PREFIX, AgentStateBase
from wattwise_core.persistence.types import (
    created_at_column,
    enum_column,
    pk_column,
    updated_at_column,
)

# --- closed canonical enum (MEM-R5; owned by doc 50) ---


class MemoryItemKind(StrEnum):
    """Closed ``memory_item_kind`` enum (MEM-R5), 1:1 to the MEM-R1 scope.

    Extending this set is a spec revision, never a runtime concern. No kind exists for
    an analytic number — memory holds personalization, never canonical ground truth
    (MEM-R1).
    """

    GOAL = "goal"
    CONSTRAINT = "constraint"
    LOAD_RESPONSE = "load_response"
    PREFERENCE = "preference"
    LANGUAGE = "language"
    PLAN_HISTORY = "plan_history"


# --- the persisted verbosity preference (VOICE-R8 §382 / MEM-R1; agent-state, not master-data) ---

#: The marker prefix of the single ``PREFERENCE``-kind memory item holding the athlete's persisted
#: response-length default (VOICE-R8 §382). The run path (engine) and the ``/v1/user-settings/
#: response-length`` surface both read/write THIS one item, so the value the athlete sets is exactly
#: the run-path default — a single source of truth in the AGENT-STATE store, never canonical master-
#: data (doc 60 §8.10). Content shape: ``response_length=<short|standard|detailed>``.
RESPONSE_LENGTH_PREF_PREFIX = "response_length="

#: The closed VOICE-R8 verbosity set; run/GET fall back to ``standard`` for an unset/unknown value.
RESPONSE_LENGTHS: tuple[str, ...] = ("short", "standard", "detailed")


def response_length_from_items(items: Sequence[RecalledItem]) -> str:
    """The persisted verbosity default carried by ``items``, else ``standard`` (VOICE-R8 §382).

    Scans recalled memory for the single ``PREFERENCE``-kind item whose content starts with
    :data:`RESPONSE_LENGTH_PREF_PREFIX` and returns its closed-set value, falling back closed to
    ``standard`` when absent or unrecognized. The ONE place the marker is parsed, so the run-path
    default and the GET endpoint resolve verbosity identically (the store-split single source).
    """
    for item in items:
        if item.kind is MemoryItemKind.PREFERENCE and item.content.startswith(
            RESPONSE_LENGTH_PREF_PREFIX
        ):
            stored = item.content[len(RESPONSE_LENGTH_PREF_PREFIX) :].strip()
            return stored if stored in RESPONSE_LENGTHS else "standard"
    return "standard"


class UntrustedMemoryWriteError(RuntimeError):
    """Raised when untrusted/scraped content attempts to write memory (MEM-R3/INJECT-R3).

    The write is refused; the caller records the attempt as an AGT-OBS-R5a injection /
    anomaly event. Untrusted content MUST NOT alter identity/scope/tooling/grounding
    via memory, and a memory item MUST NOT grant capabilities or raise a model tier.
    """


# --- ORM table (DEDICATED agent-state store; never the canonical GBO store) ---


class MemoryItem(AgentStateBase):
    """One durable, ground-truth-preserving memory episode (MEM-R1/R2/R3, ARCH-R13).

    Registered on :class:`AgentStateBase` (the dedicated agent-state metadata), NOT the
    canonical ``Base`` — durable agent memory MUST live in the agent-state store, never
    the canonical GBO store (MEM-R3/MEM-R4/ARCH-R13), with its own write credential and
    erased alongside checkpoints (CKPT-R8). ``athlete_id`` is an agent-state-side scope
    column (defence-in-depth, like ``AgentCheckpoint``), NOT a foreign key into the
    canonical ``athlete`` table. ``content`` preserves the athlete's own words (MEM-R2);
    ``inferred`` marks an LLM-derived item (MEM-R2). There is deliberately NO numeric
    column: the store cannot hold a canonical analytic value (MEM-R1).
    """

    __tablename__ = AGENT_STATE_PREFIX + "memory_item"

    memory_item_id: Mapped[uuid.UUID] = pk_column()
    athlete_id: Mapped[uuid.UUID] = mapped_column(nullable=False, index=True)
    kind: Mapped[MemoryItemKind] = enum_column(MemoryItemKind, nullable=False)
    content: Mapped[str] = mapped_column(String(2048), nullable=False)
    inferred: Mapped[bool] = mapped_column(default=False, nullable=False)
    created_at: Mapped[_dt.datetime] = created_at_column()
    updated_at: Mapped[_dt.datetime] = updated_at_column()


# --- recall result (returned to the engine; never raw numbers) ---


@dataclass(frozen=True, slots=True)
class RecalledItem:
    """A ranked memory item returned by ``fetch_relevant`` (MEM-R4).

    Carries only personalization context; never an analytic number (MEM-R1). The
    ``inferred`` flag lets the engine treat an LLM-derived item as not-asserted
    (MEM-R2).
    """

    memory_item_id: str
    kind: MemoryItemKind
    content: str
    inferred: bool
    recorded_at: _dt.datetime


# --- the recall seam (MEM-R4) ---


@runtime_checkable
class MemoryStore(Protocol):
    """The single athlete-scoped MemoryStore/recall seam (MEM-R4).

    ONE interface: ``write_episode`` (preserve a raw episode) and ``fetch_relevant``
    (athlete-scoped ranked recall). The OSS impl is relational recency/keyword; an
    embedding/ANN backend (pgvector, then dedicated at commercial scale) plugs in
    behind this SAME interface, never a re-architecture (MEM-R4). Scope is always the
    authenticated owner ``athlete_id`` — never widened by a model/tool argument.
    """

    async def write_episode(
        self,
        *,
        athlete_id: str,
        kind: MemoryItemKind,
        content: str,
        trusted: bool,
        inferred: bool = False,
    ) -> RecalledItem: ...

    async def upsert_preference(
        self, *, athlete_id: str, marker: str, content: str
    ) -> RecalledItem: ...

    async def fetch_relevant(
        self, *, athlete_id: str, query: str, limit: int = 8
    ) -> Sequence[RecalledItem]: ...

    async def erase(self, *, athlete_id: str) -> int: ...


# --- OSS implementation: relational recency/keyword recall ---


def _coerce_uuid(value: str | uuid.UUID) -> uuid.UUID:
    """Coerce a string id to a UUID at the query boundary (portable UUID binds UUIDs)."""
    return value if isinstance(value, uuid.UUID) else uuid.UUID(value)


def _to_recalled(row: MemoryItem) -> RecalledItem:
    """Project an ORM row onto the engine-facing recall record."""
    return RecalledItem(
        memory_item_id=str(row.memory_item_id),
        kind=row.kind,
        content=row.content,
        inferred=row.inferred,
        recorded_at=row.created_at,
    )


def _keyword_score(content: str, terms: frozenset[str]) -> int:
    """Count how many query terms appear in an item (case-insensitive keyword recall)."""
    haystack = content.casefold()
    return sum(1 for term in terms if term in haystack)


class OssMemoryStore:
    """OSS ``MemoryStore`` over the relational agent-state store (MEM-R4).

    Recall = athlete-scoped recency + keyword overlap; no vector DB dependency. All
    queries are scoped by the authenticated ``athlete_id`` only (MEM-R3) — one
    athlete's memory is never loadable under another identity.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def write_episode(
        self,
        *,
        athlete_id: str,
        kind: MemoryItemKind,
        content: str,
        trusted: bool,
        inferred: bool = False,
    ) -> RecalledItem:
        """Persist a raw episode (MEM-R2), refusing untrusted-sourced writes (MEM-R3).

        ``trusted`` MUST be set by the engine for content originating from the owner or
        from an engine-trusted decision — never from source-synced free text, scraped
        pages, or any tool result body. A non-trusted write raises
        :class:`UntrustedMemoryWriteError` so the caller emits an AGT-OBS-R5a event;
        nothing is persisted (fail-closed). ``content`` is free text only: the store
        has no numeric field, so a canonical analytic value cannot be stored (MEM-R1).
        """
        if not trusted:
            raise UntrustedMemoryWriteError(
                "untrusted content may not write memory (MEM-R3/INJECT-R3)"
            )
        row = MemoryItem(
            athlete_id=_coerce_uuid(athlete_id),
            kind=kind,
            content=content,
            inferred=inferred,
        )
        self._session.add(row)
        await self._session.flush()
        return _to_recalled(row)

    async def upsert_preference(
        self, *, athlete_id: str, marker: str, content: str
    ) -> RecalledItem:
        """Upsert the ONE owner-scoped PREFERENCE row carrying ``marker`` (MEM-R1, idempotent).

        Backs a single-valued agent-interaction preference (e.g. the VOICE-R8 §382 verbosity
        default, ``marker="response_length="``) held in the agent-state store, NOT a canonical
        master-data entity. UPDATES the existing ``PREFERENCE``-kind row whose ``content`` starts
        with ``marker`` (so a re-write replaces the value, never duplicating a second row), else
        INSERTS one. Scoped STRICTLY to the server-derived owner (MEM-R3). ``content`` is the FULL
        marker-prefixed value (``response_length=detailed``); this is trusted owner-originated
        preference state, so no untrusted-write guard applies (the value is a closed enum the
        caller validated, never source-synced/scraped text). Returns the persisted row.
        """
        stmt = (
            select(MemoryItem)
            .where(
                MemoryItem.athlete_id == _coerce_uuid(athlete_id),
                MemoryItem.kind == MemoryItemKind.PREFERENCE,
                MemoryItem.content.startswith(marker),
            )
            .order_by(MemoryItem.created_at.desc())
        )
        existing = list((await self._session.execute(stmt)).scalars().all())
        if existing:
            row = existing[0]
            row.content = content
            row.inferred = False
            # Defensively collapse any historical duplicates to the ONE preference row (MEM-R1).
            for stale in existing[1:]:
                await self._session.delete(stale)
        else:
            row = MemoryItem(
                athlete_id=_coerce_uuid(athlete_id),
                kind=MemoryItemKind.PREFERENCE,
                content=content,
                inferred=False,
            )
            self._session.add(row)
        await self._session.flush()
        return _to_recalled(row)

    async def fetch_relevant(
        self, *, athlete_id: str, query: str, limit: int = 8
    ) -> Sequence[RecalledItem]:
        """Athlete-scoped ranked recall by keyword overlap then recency (MEM-R4).

        Scoped strictly to the authenticated ``athlete_id`` (MEM-R3). Ranking is
        deterministic: more query-term hits first, then most-recent first, then a
        stable id tiebreak. Returns personalization context only — never a canonical
        number (MEM-R1).
        """
        stmt = (
            select(MemoryItem)
            .where(MemoryItem.athlete_id == _coerce_uuid(athlete_id))
            .order_by(MemoryItem.created_at.desc())
        )
        rows = list((await self._session.execute(stmt)).scalars().all())
        terms = frozenset(t for t in query.casefold().split() if t)
        ranked = sorted(
            rows,
            key=lambda r: (
                -_keyword_score(r.content, terms),
                _recency_key(r.created_at),
                str(r.memory_item_id),
            ),
        )
        return [_to_recalled(r) for r in ranked[:limit]]

    async def erase(self, *, athlete_id: str) -> int:
        """Erase all memory rows for an athlete (MEM-R3 per-athlete erasure)."""
        stmt = select(MemoryItem).where(MemoryItem.athlete_id == _coerce_uuid(athlete_id))
        rows = list((await self._session.execute(stmt)).scalars().all())
        for row in rows:
            await self._session.delete(row)
        await self._session.flush()
        return len(rows)


def _recency_key(recorded_at: _dt.datetime) -> float:
    """Most-recent-first sort key (negated epoch seconds; tz-aware safe)."""
    moment = recorded_at if recorded_at.tzinfo is not None else recorded_at.replace(tzinfo=_dt.UTC)
    return -moment.timestamp()


__all__ = [
    "RESPONSE_LENGTHS",
    "RESPONSE_LENGTH_PREF_PREFIX",
    "MemoryItem",
    "MemoryItemKind",
    "MemoryStore",
    "OssMemoryStore",
    "RecalledItem",
    "UntrustedMemoryWriteError",
    "response_length_from_items",
]
