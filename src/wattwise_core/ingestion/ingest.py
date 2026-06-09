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

Identity resolution here is WINDOWED-ONLY (conservative, DEDUP-R7): a NEW candidate is
matched against existing activities whose ``start_time`` is within ``_IDENTITY_WINDOW``
(±2h) via the fuzzy start/duration/sport path. Genuine cross-source strong-fingerprint
matching REGARDLESS of the window (MAP-R10) requires a TYPED ``strong_fingerprint``
distinct from ``source_native_id`` — a real shared device/file UUID, not the per-source
dedup key (two unrelated sessions, or two stripped FITs yielding a degenerate file_id,
can collide on ``source_native_id`` and must NOT merge). That typed cross-window
fingerprint match is DEFERRED and is NOT implemented via the per-source native id.
"""

from __future__ import annotations

import datetime as _dt
import uuid
from dataclasses import dataclass, field
from typing import Any, cast

from sqlalchemy import Table, select
from sqlalchemy.ext.asyncio import AsyncSession

from wattwise_core.domain.candidate import GboCandidate
from wattwise_core.domain.enums import GboType
from wattwise_core.ingestion import _canonical as _cw
from wattwise_core.ingestion._candidate_store import (
    persist_candidates_bulk,
    prepare_batch,
)
from wattwise_core.ingestion._canonical import OriginalFile
from wattwise_core.ingestion._mapping import (
    _ACTIVITY_SCALARS,
    _LAP_SCALARS,
    _activity_values,
    _highest_trust,
    _parse_date,
    _parse_start_time,
    _resolve_scalars,
    _validate_payload,
    _whole_source_tier_of,
)
from wattwise_core.ingestion.dedup import resolve_activity_identity
from wattwise_core.ingestion.trust import load_trust_policy
from wattwise_core.ingestion.watermark import SyncedRange, advance_and_heal
from wattwise_core.persistence.models import Activity, SourceCandidate
from wattwise_core.persistence.types import uuid7
from wattwise_core.persistence.upsert import upsert
from wattwise_core.storage import ObjectStore, create_object_store

_IDENTITY_WINDOW = _dt.timedelta(hours=2)


@dataclass(slots=True)
class IngestResult:
    """Summary of an ingest batch."""

    activities_written: set[str] = field(default_factory=set)
    wellness_written: int = 0
    candidates_persisted: int = 0
    candidates_failed: int = 0
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
    ) -> None:
        self._session = session
        self._object_store = object_store
        # PERF-R1 / ING-UPS-R1/R3: candidates are landed in bounded batches. The size is
        # configuration (CFG-R1a), supplied by the caller; ``None`` lands the whole list as
        # one batch (the fault-isolation savepoint per row is what bounds blast radius).
        self._batch_size = batch_size

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
        """
        athlete = _uid(athlete_id)
        descriptor = _uid(source_descriptor_id)
        run_id = ingest_run_id or uuid7()
        files_by_native = {f.source_native_id: f for f in (original_files or [])}
        result = IngestResult()
        wellness_dates: set[_dt.date] = set()
        for batch in _batched(candidates, self._batch_size):
            await _land_batch(
                self, athlete, descriptor, batch, connection_id, run_id,
                files_by_native, wellness_dates, result,
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
                self._session, athlete, descriptor, candidates, synced_range,
                ingest_run_id=run_id,
            )
            result.watermarks_advanced = advanced.watermarks_advanced
            result.gaps_closed = advanced.gaps_closed
        await self._session.commit()
        return result

    async def _resolve_and_write_activity(
        self, athlete: uuid.UUID, row: SourceCandidate, cand: GboCandidate
    ) -> uuid.UUID:
        """Resolve identity (reusing the row's prior id) and write the canonical activity."""
        if row.resolved_activity_id is not None:
            activity_id = row.resolved_activity_id  # ING-R6: reuse the resolved identity
        else:
            activity_id = await self._resolve_activity_id(athlete, cand)
            row.resolved_activity_id = activity_id
            await self._session.flush()
        await self._write_activity_canonical(athlete, activity_id)
        return activity_id

    async def _resolve_activity_id(self, athlete: uuid.UUID, cand: GboCandidate) -> uuid.UUID:
        """Resolve a NEW candidate to a canonical activity id (MAP-R9..R12, DEDUP-R7).

        WINDOWED-ONLY (conservative): considers only existing activities whose
        ``start_time`` is within ``_IDENTITY_WINDOW`` (±2h) of the candidate, in a stable
        order (start_time, then activity_id), and runs the fuzzy start/duration/sport
        matcher per windowed candidate; reuses the first match, else mints a new id.

        A cross-source strong-fingerprint match REGARDLESS of the window (MAP-R10) is
        DEFERRED: it needs a TYPED ``strong_fingerprint`` (a real shared device/file UUID),
        NOT the per-source ``source_native_id`` dedup key (which can falsely collide across
        unrelated sessions). Same-source re-ingest is already handled upstream by
        candidate-key id reuse (``resolved_activity_id``, ING-R6) BEFORE this runs.
        """
        start = _parse_start_time(cand.payload["start_time"])
        duration = float(cast("float", cand.payload.get("elapsed_time_s") or 0))
        sport = str(cand.payload.get("sport") or "other")
        for act in await self._windowed_activities(athlete, start):
            # SQLite returns tz-naive datetimes; coerce to UTC for the matcher (GBO-R32).
            act_start = _parse_start_time(act.start_time)
            if resolve_activity_identity(
                start, duration, sport, None,
                act_start, float(act.elapsed_time_s or 0), act.sport, None,
            ):
                return act.activity_id
        return uuid7()

    async def _windowed_activities(
        self, athlete: uuid.UUID, start: _dt.datetime
    ) -> list[Activity]:
        """Existing activities whose ``start_time`` falls within ±2h of ``start``.

        Returns them in a stable order (start_time, then activity_id) so identity
        resolution is deterministic (CONF-R4). The fuzzy start/duration/sport matcher
        is run per candidate; nothing outside the window is considered (DEDUP-R7).
        """
        lo, hi = start - _IDENTITY_WINDOW, start + _IDENTITY_WINDOW
        stmt = (
            select(Activity)
            .where(
                Activity.athlete_id == athlete,
                Activity.start_time >= lo,
                Activity.start_time <= hi,
            )
            .order_by(Activity.start_time, Activity.activity_id)
        )
        return list((await self._session.execute(stmt)).scalars().all())

    async def _write_activity_canonical(
        self, athlete: uuid.UUID, activity_id: uuid.UUID
    ) -> None:
        """Resolve every field across candidates and write the canonical activity (UPS-R2).

        The canonical row is persisted through the atomic upsert seam keyed on the
        resolved canonical key ``activity_id`` — a single insert-or-update, never a
        ``session.get`` then add/setattr check-then-write, so two sync runs landing the
        same resolved activity cannot race (UPS-R2). Only the resolved columns are
        supplied, so unresolved fields keep their prior canonical value when the key exists.
        """
        candidates = await _activity_candidates(self._session, athlete, activity_id)
        if not candidates:
            return
        policy = await load_trust_policy(self._session, athlete, candidates)
        scalars, coverage = _resolve_scalars(candidates, _ACTIVITY_SCALARS, policy)
        values, update_columns = _activity_values(activity_id, athlete, scalars, coverage)
        await upsert(
            self._session,
            cast("Table", Activity.__table__),
            values,
            conflict_keys=["activity_id"],
            update_columns=update_columns,
        )
        await self._session.flush()
        # Streams resolve PER CHANNEL under each channel's effective tier (CONF-R3/SF-3);
        # an empty policy makes this the candidate's adapter tier (the prior behaviour).
        streams = _cw.resolve_streams(candidates, policy)
        best = _highest_trust(candidates)
        laps = cast("list[dict[str, Any]]", best.payload.get("laps") or [])
        if streams:
            await _cw.upsert_stream_set(self._session, activity_id, streams)
        await _cw.upsert_laps(self._session, activity_id, laps, _LAP_SCALARS)

    async def _write_wellness(self, athlete: uuid.UUID, local_date: _dt.date) -> None:
        """Resolve daily wellness across ALL candidates for the date (CONF-R2/ING-UPS-R5)."""
        candidates = await _wellness_candidates(self._session, athlete, local_date)
        policy = await load_trust_policy(self._session, athlete, candidates)
        # Wellness fields resolve under the whole-source effective tier; an empty policy
        # makes this the candidate's adapter tier (byte-identical to the prior behaviour).
        await _cw.write_wellness_canonical(
            self._session, athlete, local_date, candidates, _whole_source_tier_of(policy)
        )

    async def _capture_original(
        self,
        athlete: uuid.UUID,
        descriptor: uuid.UUID,
        activity_id: uuid.UUID,
        original: OriginalFile | None,
        fetched_at: _dt.datetime | None,
    ) -> None:
        """Store the verbatim original file + its activity_file reference (ING-R8/FIL-R1)."""
        if original is None:
            return  # a direct-API source has no original recording file -> no ActivityFile
        store = self._object_store or create_object_store()
        await _cw.create_activity_file(
            self._session, store, athlete=athlete, activity_id=activity_id,
            source_descriptor_id=descriptor, original=original, fetched_at=fetched_at,
        )


def _batched(candidates: list[GboCandidate], size: int | None) -> list[list[GboCandidate]]:
    """Split candidates into bounded batches (PERF-R1 / ING-UPS-R1); ``None`` = one batch."""
    if size is None or size >= len(candidates):
        return [candidates] if candidates else []
    return [candidates[i : i + size] for i in range(0, len(candidates), size)]


async def _land_batch(
    svc: IngestService,
    athlete: uuid.UUID,
    descriptor: uuid.UUID,
    batch: list[GboCandidate],
    connection_id: str | uuid.UUID | None,
    run_id: uuid.UUID,
    files_by_native: dict[str, OriginalFile],
    wellness_dates: set[_dt.date],
    result: IngestResult,
) -> None:
    """Land ONE batch: validate+prepare per record, bulk-insert, then resolve per record.

    Each candidate is validated+prepared in its own ``SAVEPOINT`` so a malformed record
    rolls back only itself (ING-UPS-R3 record isolation); the surviving rows land in a
    SINGLE multi-row upsert round-trip (ING-UPS-R1 / PERF-R1), then each is resolved +
    canonical-written in its own ``SAVEPOINT`` so a resolution failure likewise isolates.
    """
    prepared, failed = await prepare_batch(
        svc._session, athlete, descriptor, batch, connection_id, run_id,
        validate=_validate_payload,
    )
    result.candidates_failed += failed
    if not prepared:
        return
    rows = await persist_candidates_bulk(svc._session, athlete, descriptor, prepared)
    for prep in prepared:
        await _resolve_candidate(
            svc, athlete, descriptor, prep.cand, rows[prep.cand.source_native_id],
            files_by_native, wellness_dates, result,
        )


async def _resolve_candidate(
    svc: IngestService,
    athlete: uuid.UUID,
    descriptor: uuid.UUID,
    cand: GboCandidate,
    row: SourceCandidate,
    files_by_native: dict[str, OriginalFile],
    wellness_dates: set[_dt.date],
    result: IngestResult,
) -> None:
    """Resolve + canonical-write ONE already-persisted candidate inside its own SAVEPOINT.

    The candidate row is already durable from the batch's bulk insert; a resolution failure
    rolls back only this savepoint and is counted, never aborting the batch (ING-UPS-R3
    record isolation). ING-UPS-R3's range-precise gap (ING-GAP-R5) for the failed record is
    DEFERRED to the watermark/gap model (ING-UPS-R2).
    """
    try:
        async with svc._session.begin_nested():
            if cand.gbo_type == GboType.ACTIVITY.value:
                activity_id = await svc._resolve_and_write_activity(athlete, row, cand)
                await svc._capture_original(
                    athlete, descriptor, activity_id,
                    files_by_native.get(cand.source_native_id), cand.fetched_at,
                )
                result.activities_written.add(str(activity_id))
            elif cand.gbo_type == GboType.DAILY_WELLNESS.value:
                wellness_dates.add(_parse_date(cand.payload["local_date"]))
    except Exception:
        result.candidates_failed += 1  # ING-UPS-R3 record isolation; keep the run
        return
    result.candidates_persisted += 1


async def _activity_candidates(
    session: AsyncSession, athlete: uuid.UUID, activity_id: uuid.UUID
) -> list[SourceCandidate]:
    """All non-superseded activity candidates resolved to ``activity_id`` (the resolution set)."""
    stmt = select(SourceCandidate).where(
        SourceCandidate.athlete_id == athlete,
        SourceCandidate.gbo_type == GboType.ACTIVITY,
        SourceCandidate.resolved_activity_id == activity_id,
        SourceCandidate.is_superseded.is_(False),
    )
    return list((await session.execute(stmt)).scalars().all())


async def _wellness_candidates(
    session: AsyncSession, athlete: uuid.UUID, local_date: _dt.date
) -> list[SourceCandidate]:
    """All non-superseded daily-wellness candidates for ``local_date`` (the resolution set)."""
    stmt = select(SourceCandidate).where(
        SourceCandidate.athlete_id == athlete,
        SourceCandidate.gbo_type == GboType.DAILY_WELLNESS,
        SourceCandidate.is_superseded.is_(False),
    )
    rows = (await session.execute(stmt)).scalars().all()
    return [c for c in rows if _parse_date(c.payload.get("local_date")) == local_date]


def _uid(value: str | uuid.UUID) -> uuid.UUID:
    return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))


__all__ = ["IngestResult", "IngestService", "OriginalFile", "SyncedRange"]
