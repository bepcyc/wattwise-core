"""Engine-owned extension seams (SEAM-R11, ARCH-R31, CONF-R7, DEDUP-R6).

This module declares the engine's typed ``Protocol`` seams that govern canonical-store
access and cross-source conflict resolution. They are real, working extension points in
the open-source engine; only the commercial business logic mounted *through* them
(tenant-scoping, the advanced multi-source resolver) is out of OSS scope (doc 90).

The two seams owned here are:

* ``SessionProvider`` (SEAM-R11 / ARCH-R31) — the ONE engine-owned session/repository
  provider through which ALL canonical-store access (reads AND writes) flows. It takes the
  server-derived ``subject`` context (ARCH-R16 — athlete in OSS; (tenant, athlete) in the
  commercial layer). No layer may open its own canonical session or reach the store around
  this provider. The OSS default :class:`EngineSessionProvider` performs NO tenant scoping
  (single-athlete) but IS the single attach point the commercial tenant-scoped overlay
  (doc 90 COMM-R22) mounts on, never by patching OSS code.

* ``ConflictResolver`` (CONF-R7 / DEDUP-R6) — the pluggable identity + field-level
  conflict-resolution strategy ``IngestService`` is injected with (never a direct import).
  OSS ships exactly ONE deterministic, conservative default
  (:class:`DefaultConflictResolver` -> DEDUP-R7); the advanced high-accuracy multi-source
  resolver (DEDUP-R8) is a COMMERCIAL feature supplied through this same seam. The
  single-count invariant (DEDUP-R1/R4) holds regardless of which resolver is active.

This module is rankless (cross-cutting): any layer may depend on these Protocols, and the
seam wraps the inner persistence/ingestion machinery without inverting the import direction.
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Protocol, runtime_checkable

from sqlalchemy.ext.asyncio import AsyncSession

from wattwise_core.agent.memory import MemoryStore
from wattwise_core.agent.tiering import ModelRoutingPolicy
from wattwise_core.domain.candidate import FieldCandidate
from wattwise_core.entitlement import EntitlementResolver
from wattwise_core.ingestion.base import FileImportAdapter, SourceAdapter
from wattwise_core.ingestion.dedup import (
    DEFAULT_DURATION_TOL_FRAC,
    DEFAULT_DURATION_TOL_S,
    DEFAULT_START_WINDOW_S,
    ResolvedField,
    resolve_activity_identity,
    resolve_field,
)
from wattwise_core.persistence import Database

#: The subject a NON-scoped system/operator canonical open carries (SEAM-R11 / ARCH-R31).
#:
#: ``SessionProvider.session`` is the SINGLE canonical choke point, so even a request-less
#: system path (the readiness DB-reachability probe, OBS-R6.2; the operator-driven whole-athlete
#: erasure executor before it has narrowed to one subject) MUST open through it rather than reach
#: the store around it. Those paths are not bound to a per-request server-derived athlete, so they
#: carry this explicit non-scoped marker. In the OSS provider it is inert (ARCH-R31 — the OSS
#: provider does NO tenant scoping); the commercial tenant-scoped provider treats it as the
#: operator/system context, never as a real tenant subject (it is deliberately not a valid athlete
#: id). It is a sanctioned, named subject — NOT a bypass of the provider.
SYSTEM_SUBJECT: str = "_system"


@runtime_checkable
class SessionProvider(Protocol):
    """The single engine-owned canonical-store session provider (SEAM-R11 / ARCH-R31).

    ALL canonical-store access — reads AND writes — MUST flow through this one provider;
    no layer may open its own canonical session or reach the store around it. ``session``
    takes the server-derived ``subject`` (ARCH-R16; non-ambient, never client-asserted)
    and yields ONE transactional :class:`AsyncSession` (commit on success, roll back on
    error). The OSS default does NO tenant scoping; the commercial tenant-scoped overlay
    (doc 90 COMM-R22) is a conforming implementation supplied through this seam — it is the
    data-access slot of the COMM-R14 ``PluginBundle``.
    """

    def session(self, *, subject: str) -> AbstractAsyncContextManager[AsyncSession]:
        """Yield one transactional canonical session for ``subject``."""
        ...


class EngineSessionProvider:
    """OSS default :class:`SessionProvider`: the single un-scoped canonical choke point.

    Wraps the engine-owned :class:`~wattwise_core.persistence.engine.Database` and yields
    its transactional session. It accepts and carries the server-derived ``subject`` but
    applies NO tenant scoping (single-athlete OSS, ARCH-R31) — it is exactly the one
    provider/choke point ARCH-R31's positive clause names, and the attach point the
    commercial tenant-scoped provider replaces (SEAM-R11), never by patching OSS code.
    """

    __slots__ = ("_database",)

    def __init__(self, database: Database) -> None:
        self._database = database

    @asynccontextmanager
    async def session(self, *, subject: str) -> AsyncIterator[AsyncSession]:
        """Yield one transactional canonical session; ``subject`` is carried, not scoped."""
        _ = subject  # OSS default applies no tenant scoping (ARCH-R31); subject is inert here
        async with self._database.session() as session:
            yield session


@runtime_checkable
class ConflictResolver(Protocol):
    """The pluggable cross-source identity + field-conflict resolver seam (CONF-R7/DEDUP-R6).

    A conforming resolver decides cross-source activity identity (``resolve_activity_identity``,
    MAP-R9..R12) and the canonical value of one field across contributing candidates
    (``resolve_field``, CONF-R2/R3/R5). It MUST satisfy the single-count invariant
    (DEDUP-R1/R4) regardless of strategy. OSS ships exactly one deterministic, conservative
    default (DEDUP-R7); the advanced multi-source resolver (DEDUP-R8) is commercial and rides
    this same seam — it is the dedup-resolver slot of the COMM-R14 ``PluginBundle``.
    """

    def resolve_field(
        self, candidates: list[FieldCandidate], *, dispute_tolerance: float | None = None
    ) -> ResolvedField | None:
        """Resolve one canonical field from contributing candidates (CONF-R2/R3/R5)."""
        ...

    def resolve_activity_identity(
        self,
        a_start: _dt.datetime,
        a_duration_s: float,
        a_sport: str,
        a_fingerprint: str | None,
        b_start: _dt.datetime,
        b_duration_s: float,
        b_sport: str,
        b_fingerprint: str | None,
        *,
        start_window_s: float = DEFAULT_START_WINDOW_S,
        duration_tol_frac: float = DEFAULT_DURATION_TOL_FRAC,
        duration_tol_s: float = DEFAULT_DURATION_TOL_S,
    ) -> bool:
        """Decide whether two activity candidates are the same real-world session (MAP-R10)."""
        ...


class DefaultConflictResolver:
    """OSS default :class:`ConflictResolver` (DEDUP-R7) — deterministic and conservative.

    Delegates to the shipped pure ``resolve_field`` / ``resolve_activity_identity`` (the
    DEDUP-R7 conservative resolver: collapses only high-confidence identity matches, keeps
    ambiguous candidates separate). It satisfies DEDUP-R1/R4; the advanced commercial
    resolver (DEDUP-R8) replaces it through the seam without editing the consumer.
    """

    __slots__ = ()

    def resolve_field(
        self, candidates: list[FieldCandidate], *, dispute_tolerance: float | None = None
    ) -> ResolvedField | None:
        return resolve_field(candidates, dispute_tolerance=dispute_tolerance)

    def resolve_activity_identity(
        self,
        a_start: _dt.datetime,
        a_duration_s: float,
        a_sport: str,
        a_fingerprint: str | None,
        b_start: _dt.datetime,
        b_duration_s: float,
        b_sport: str,
        b_fingerprint: str | None,
        *,
        start_window_s: float = DEFAULT_START_WINDOW_S,
        duration_tol_frac: float = DEFAULT_DURATION_TOL_FRAC,
        duration_tol_s: float = DEFAULT_DURATION_TOL_S,
    ) -> bool:
        return resolve_activity_identity(
            a_start,
            a_duration_s,
            a_sport,
            a_fingerprint,
            b_start,
            b_duration_s,
            b_sport,
            b_fingerprint,
            start_window_s=start_window_s,
            duration_tol_frac=duration_tol_frac,
            duration_tol_s=duration_tol_s,
        )


# Re-exported integration seams (doc 90 §5.1: every stable extension seam is importable
# from `wattwise_core.seams`, so a consumer never reaches into private modules). The
# Protocols stay DEFINED beside their implementations; this module is the one public
# import surface. (The coach-config selector and HITL resume handler are config/
# checkpoint surfaces without a standalone Protocol yet; they join here when typed.)
__all__ = [
    "SYSTEM_SUBJECT",
    "ConflictResolver",
    "DefaultConflictResolver",
    "EngineSessionProvider",
    "EntitlementResolver",
    "FileImportAdapter",
    "MemoryStore",
    "ModelRoutingPolicy",
    "SessionProvider",
    "SourceAdapter",
]
