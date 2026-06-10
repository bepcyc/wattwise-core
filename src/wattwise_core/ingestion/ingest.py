"""Canonical ingest write path (UPS-R*, CONF-R*, MAP-R9..R12, DEDUP-R1/R7).

Takes the pure adapters' :class:`GboCandidate` list and lands it into the canonical
store: each candidate is persisted (tier 2), its identity resolved across sources to
ONE canonical activity (MAP-R9..R12), and the resolved canonical record written by
running the field-level conflict resolver over every retained candidate (CONF-R2/R3).
The single-count invariant holds — a real session from N sources becomes one activity
(DEDUP-R1). Everything for one candidate batch happens inside the caller's session
transaction (UPS-R6); re-ingesting unchanged content is a value-level no-op (UPS-R3).

Ingestion (L3) is the ONLY writer to the canonical store (L4, ARCH-R3); it imports
persistence inward toward the canonical core.

Identity resolution runs in two legs (MAP-R10): a TYPED ``strong_fingerprint`` (a real
shared device/file UUID carried on the candidate — never the per-source
``source_native_id`` dedup key, which can falsely collide across unrelated sessions)
matches retained candidates REGARDLESS of the time window, still gated on sport
compatibility; otherwise the conservative WINDOWED fuzzy path (DEDUP-R7) matches
existing activities whose ``start_time`` is within ``_IDENTITY_WINDOW`` (±2h) on
start/duration/sport. Every identity decision is recorded on the candidate row
(rule fired, match score, matched ids — MAP-R12) so a merge is explainable and can be
split later by the explicit ``reresolve.split_activity`` operation.

The cohesive sub-steps (batch landing, identity resolution, canonical writes, local-day
projection, file capture) live in the sibling :mod:`._ingest_steps`; this module owns
the facade and its orchestration only.
"""

from __future__ import annotations

import datetime as _dt
import uuid
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from wattwise_core.domain.candidate import GboCandidate
from wattwise_core.domain.enums import GboType
from wattwise_core.ingestion._canonical import OriginalFile
from wattwise_core.ingestion._ingest_steps import (
    _land_batch,
    _resolve_activity_id,
    _write_activity_canonical,
    _write_wellness,
)
from wattwise_core.ingestion.capability import (
    require_declared_types,
)
from wattwise_core.ingestion.watermark import SyncedRange, advance_and_heal
from wattwise_core.persistence.types import uuid7
from wattwise_core.seams import ConflictResolver, DefaultConflictResolver
from wattwise_core.storage import ObjectStore


@dataclass(slots=True)
class IngestResult:
    """Summary of an ingest batch."""

    activities_written: set[str] = field(default_factory=set)
    wellness_written: int = 0
    candidates_persisted: int = 0
    candidates_failed: int = 0
    # MAP-R6: candidates persisted to quarantine (failed validation, retained with the
    # failing rule id, excluded from resolution) — distinct from hard failures.
    candidates_quarantined: int = 0
    watermarks_advanced: int = 0
    gaps_closed: int = 0


class IngestService:
    """Persists adapter candidates and resolves them into canonical records."""

    def __init__(
        self,
        session: AsyncSession,
        *,
        object_store: ObjectStore | None = None,
        batch_size: int | None = None,
        resolver: ConflictResolver | None = None,
    ) -> None:
        self._session = session
        self._object_store = object_store
        # PERF-R1 / ING-UPS-R1/R3: candidates are landed in bounded batches. The size is
        # configuration (CFG-R1a), supplied by the caller; ``None`` lands the whole list as
        # one batch (the fault-isolation savepoint per row is what bounds blast radius).
        self._batch_size = batch_size
        # The dedup/conflict resolver is INJECTED behind the seam (CONF-R7/DEDUP-R6),
        # never directly imported: the OSS conservative default (DEDUP-R7) unless a
        # commercial DEDUP-R8 resolver is supplied through the seam.
        self._resolver: ConflictResolver = resolver or DefaultConflictResolver()

    async def ingest(
        self,
        athlete_id: str | uuid.UUID,
        source_descriptor_id: str | uuid.UUID,
        candidates: list[GboCandidate],
        *,
        connection_id: str | uuid.UUID | None = None,
        ingest_run_id: uuid.UUID | None = None,
        original_files: list[OriginalFile] | None = None,
        synced_range: SyncedRange | None = None,
        declared_gbo_types: frozenset[GboType] | None = None,
        source_key: str = "",
    ) -> IngestResult:
        """Land candidates into the canonical store in DURABLE, fault-isolated batches.

        Candidates are processed in batches of ``batch_size`` (PERF-R1 / ING-UPS-R1); each
        batch's candidate rows land in ONE multi-row ``VALUES`` upsert round-trip and each
        record is resolved in its OWN ``SAVEPOINT`` so one bad record rolls back only itself
        (a whole-run rollback is prohibited, ING-UPS-R3). Every successful batch is
        **committed before the next begins**, so a later batch's failure leaves all earlier
        batches durably persisted (ING-UPS-R3 / ACC-4) — SAVEPOINTs alone would be lost on an
        outer rollback. ``original_files`` are stored verbatim and linked via ``activity_file``
        (ING-R8/FIL-R1). Wellness candidates resolve across ALL same-day candidates (CONF-R2),
        never last-write-wins.

        When ``synced_range`` is given, the per-``gbo_type`` watermark is advanced and any
        OPEN transient gap fully inside that range is closed — AFTER all batch data has been
        committed above (SYN-R3 / ING-UPS-R2 / ING-GAP-R4), so store, cursor, and gap state
        stay mutually consistent and a crash mid-run never advances past un-committed data (ING-R6).

        ADP-R3 (fail-closed): BEFORE any write, every candidate's ``gbo_type`` must be in
        the adapter's ``declared_gbo_types`` (when given) AND in the engine-writable set —
        an undeclared/unknown type raises :class:`UndeclaredGboTypeError`; the batch is
        REFUSED, never partially/silently dropped from the canonical store.
        """
        require_declared_types(candidates, declared_gbo_types, source_key=source_key)
        athlete = _uid(athlete_id)
        descriptor = _uid(source_descriptor_id)
        run_id = ingest_run_id or uuid7()
        files_by_native = {f.source_native_id: f for f in (original_files or [])}
        result = IngestResult()
        wellness_dates: set[_dt.date] = set()
        for batch in _batched(candidates, self._batch_size):
            await _land_batch(
                self,
                athlete,
                descriptor,
                batch,
                connection_id,
                run_id,
                files_by_native,
                wellness_dates,
                result,
            )
            # ING-UPS-R3 / ACC-4: commit each batch as its own durable unit so a later
            # batch's failure cannot lose an already-completed batch.
            await self._session.commit()
        for local_date in wellness_dates:
            await self._write_wellness(athlete, local_date)
            result.wellness_written += 1
        if synced_range is not None:
            # Advance the watermark + self-heal covered transient gaps (SYN-R3 / ING-UPS-R2 /
            # ING-GAP-R4) AFTER all batch data is committed above, so cursor/gap state never
            # diverge from durable data and a crash never advances past un-committed data (ING-R6).
            advanced = await advance_and_heal(
                self._session,
                athlete,
                descriptor,
                candidates,
                synced_range,
                ingest_run_id=run_id,
            )
            result.watermarks_advanced = advanced.watermarks_advanced
            result.gaps_closed = advanced.gaps_closed
        await self._session.commit()
        return result

    async def _resolve_activity_id(
        self, athlete: uuid.UUID, cand: GboCandidate
    ) -> tuple[uuid.UUID, dict[str, Any]]:
        """Resolve a NEW candidate to a canonical activity id (MAP-R9..R12, DEDUP-R7).

        Delegates to :func:`._ingest_steps._resolve_activity_id` (the two-leg
        strong-fingerprint / windowed-fuzzy resolution); kept as a method seam.
        """
        return await _resolve_activity_id(self, athlete, cand)

    async def _write_activity_canonical(self, athlete: uuid.UUID, activity_id: uuid.UUID) -> None:
        """Resolve every field across candidates and write the canonical activity (UPS-R2).

        Delegates to :func:`._ingest_steps._write_activity_canonical`; kept as a
        method seam (the re-resolution path calls it on the service).
        """
        await _write_activity_canonical(self, athlete, activity_id)

    async def _write_wellness(self, athlete: uuid.UUID, local_date: _dt.date) -> None:
        """Resolve daily wellness across ALL candidates for the date (CONF-R2/ING-UPS-R5).

        Delegates to :func:`._ingest_steps._write_wellness`; kept as a method seam
        (the re-resolution path calls it on the service).
        """
        await _write_wellness(self, athlete, local_date)


def _batched(candidates: list[GboCandidate], size: int | None) -> list[list[GboCandidate]]:
    """Split candidates into bounded batches (PERF-R1 / ING-UPS-R1); ``None`` = one batch."""
    if size is None or size >= len(candidates):
        return [candidates] if candidates else []
    return [candidates[i : i + size] for i in range(0, len(candidates), size)]


def _uid(value: str | uuid.UUID) -> uuid.UUID:
    return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))


__all__ = ["IngestResult", "IngestService", "OriginalFile", "SyncedRange"]
