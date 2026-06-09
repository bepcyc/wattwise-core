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
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, cast

from sqlalchemy import Table, select
from sqlalchemy.ext.asyncio import AsyncSession

from wattwise_core.domain.candidate import GboCandidate
from wattwise_core.domain.enums import Fidelity, GboType, trust_rank
from wattwise_core.ingestion import _canonical as _cw
from wattwise_core.ingestion._candidate_store import (
    persist_candidates_bulk,
    prepare_batch,
)
from wattwise_core.ingestion._canonical import OriginalFile
from wattwise_core.ingestion.dedup import resolve_activity_identity, resolve_field
from wattwise_core.ingestion.trust import TrustPolicy, load_trust_policy
from wattwise_core.persistence.models import Activity, SourceCandidate
from wattwise_core.persistence.models.athlete_preference import WHOLE_SOURCE_CHANNEL
from wattwise_core.persistence.types import utcnow, uuid7
from wattwise_core.persistence.upsert import upsert
from wattwise_core.storage import ObjectStore, create_object_store

# Canonical scalar fields carried on an activity candidate's payload (resolved per
# field across candidates; streams/laps are handled separately).
_ACTIVITY_SCALARS = (
    "start_time", "sport", "sub_sport", "elapsed_time_s", "moving_time_s", "distance_m",
    "total_work_j", "energy_kj", "avg_power_w", "max_power_w", "avg_hr_bpm", "max_hr_bpm",
    "avg_cadence_rpm", "avg_speed_mps", "elevation_gain_m", "avg_temp_c", "device_class",
)
_LAP_SCALARS = (
    "start_offset_s", "duration_s", "distance_m", "avg_power_w", "max_power_w",
    "avg_hr_bpm", "max_hr_bpm", "avg_cadence_rpm", "avg_speed_mps", "elevation_gain_m",
)
_IDENTITY_WINDOW = _dt.timedelta(hours=2)


@dataclass(slots=True)
class IngestResult:
    """Summary of an ingest batch."""

    activities_written: set[str] = field(default_factory=set)
    wellness_written: int = 0
    candidates_persisted: int = 0
    candidates_failed: int = 0


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
        written and committed with the final batch.
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


def _validate_payload(cand: GboCandidate) -> None:
    """Parse the resolution-critical payload fields, raising on a malformed candidate."""
    if cand.gbo_type == GboType.ACTIVITY.value:
        _parse_start_time(cand.payload["start_time"])
    elif cand.gbo_type == GboType.DAILY_WELLNESS.value:
        _parse_date(cand.payload["local_date"])


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


def _resolve_scalars(
    candidates: list[SourceCandidate], fields: tuple[str, ...], policy: TrustPolicy
) -> tuple[dict[str, Any], dict[str, object]]:
    """Resolve each scalar field across candidates + build its coverage (CONF-R2/R5).

    Returns ``(resolved_values, coverage)``. Each field is resolved with its EFFECTIVE
    per-channel trust tier (``policy.tier(candidate, fname)`` — the configurable PRV-R7
    re-rank, defaulting to the adapter tier when unconfigured). A field whose >=2
    contributors materially disagree beyond the per-field dispute tolerance gets
    ``coverage.disputed=True`` — the best value is still selected, the disagreement is
    surfaced not hidden (CONF-R5).
    """
    resolved: dict[str, Any] = {}
    coverage: dict[str, object] = {}
    for fname in fields:
        tier_of = _channel_tier_of(policy, fname)  # effective per-channel tier (PRV-R7)
        contributors = _cw.field_candidates(candidates, fname, tier_of)
        winner = resolve_field(contributors, dispute_tolerance=_cw.dispute_tolerance(fname))
        if winner is None:
            continue
        resolved[fname] = winner.value
        # Badge the RESOLVED WINNER's tier, NOT an arbitrary scanned contributor (PRV-R6).
        coverage[fname] = _cw.coverage_for(
            True, winner.winning_trust_tier, disputed=winner.disputed
        ).to_jsonable()
    return resolved, coverage


_ACTIVITY_COLUMNS = frozenset(Activity.__table__.columns.keys())


def _activity_values(
    activity_id: uuid.UUID, athlete: uuid.UUID, scalars: dict[str, Any], coverage: dict[str, object]
) -> tuple[dict[str, Any], list[str]]:
    """The activity row value-dict + the update-on-collision set for the atomic upsert (UPS-R2).

    Carries the resolved scalars (``start_time`` parsed to tz-aware UTC), the derived
    ``has_power``/``has_hr``/``coverage`` flags, and a fresh ``updated_at``. ``sport`` is
    NOT NULL, so a new row defaults to ``"other"`` when unresolved. The returned update set
    is exactly the resolved/derived columns — ``sport`` is included ONLY when resolved, so a
    conflicting (existing) row never has a previously-resolved value regressed to a default,
    matching the prior setattr-only behaviour (no zero-filling, PRV-R6).
    """
    values: dict[str, Any] = {"activity_id": activity_id, "athlete_id": athlete}
    update_columns: list[str] = []
    for key, value in scalars.items():
        col = "start_time" if key == "start_time" else key
        if col not in _ACTIVITY_COLUMNS:
            continue
        values[col] = _parse_start_time(value) if key == "start_time" else value
        update_columns.append(col)
    values.setdefault("sport", "other")  # NOT NULL on a fresh insert; refreshed only if resolved
    values["has_power"] = scalars.get("avg_power_w") is not None
    values["has_hr"] = scalars.get("avg_hr_bpm") is not None
    values["coverage"] = coverage
    values["updated_at"] = utcnow()
    update_columns += ["has_power", "has_hr", "coverage", "updated_at"]
    return values, update_columns


def _channel_tier_of(
    policy: TrustPolicy, channel: str
) -> Callable[[SourceCandidate], Fidelity]:
    """A channel-bound effective-tier seam ``(candidate) -> Fidelity`` for ``_canonical``.

    Binds the channel so the single-arg ``tier_of`` the ``_canonical`` helpers call
    resolves the EFFECTIVE per-channel tier (PRV-R7), keeping ``dedup.resolve_field`` and
    the ``_canonical`` helpers free of any DB read — the policy is already in memory.
    """
    return lambda candidate: policy.tier(candidate, channel)


def _whole_source_tier_of(policy: TrustPolicy) -> Callable[[SourceCandidate], Fidelity]:
    """The effective-tier seam bound to the whole-source channel (``"*"``).

    Used for record-level surfaces (streams, wellness) that resolve under the
    whole-source effective tier: per-athlete ``"*"`` override → descriptor ``"*"`` /
    ``default_fidelity`` → the candidate's adapter tier (the prior behaviour when
    unconfigured).
    """
    return lambda candidate: policy.tier(candidate, WHOLE_SOURCE_CHANNEL)


def _tier_of(candidate: SourceCandidate) -> Fidelity:
    """The candidate's ACTUAL adapter-assigned tier (NOT re-ranked by config).

    Used only for config-independent candidate selection (e.g. which candidate's ``laps``
    payload to take, ``_highest_trust``) — never for field-level conflict resolution,
    which goes through the configurable :class:`TrustPolicy`.
    """
    raw = candidate.trust_profile.get("tier", Fidelity.PLATFORM_COMPUTED.value)
    return Fidelity(str(raw))


def _highest_trust(candidates: list[SourceCandidate]) -> SourceCandidate:
    return min(candidates, key=lambda c: (trust_rank(_tier_of(c)), str(c.source_descriptor_id)))


def _parse_start_time(value: Any) -> _dt.datetime:
    """Parse a stored ISO start_time back to a tz-aware UTC datetime."""
    dt = value if isinstance(value, _dt.datetime) else _dt.datetime.fromisoformat(str(value))
    return dt if dt.tzinfo else dt.replace(tzinfo=_dt.UTC)


def _parse_date(value: Any) -> _dt.date:
    return value if isinstance(value, _dt.date) else _dt.date.fromisoformat(str(value))


def _uid(value: str | uuid.UUID) -> uuid.UUID:
    return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))


__all__ = ["IngestResult", "IngestService", "OriginalFile"]
