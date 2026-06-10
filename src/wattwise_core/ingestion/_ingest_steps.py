"""Ingest write-path sub-steps (UPS-R*, CONF-R*, MAP-R9..R12, DEDUP-R1/R7).

The cohesive helper functions the canonical ingest facade
(:class:`wattwise_core.ingestion.ingest.IngestService`) composes, factored to a sibling
module so the facade stays within the QUAL-R9 size ceilings WITHOUT changing its public
API: batch landing + per-record fault isolation (ING-UPS-R1/R3), the two-leg identity
resolution (MAP-R9..R12 strong-fingerprint, DEDUP-R7 windowed fuzzy), the canonical
activity/wellness writes through the atomic upsert seam (UPS-R2, CONF-R2/R3), the
local-day projection (GBO-R33/R35), and original-file capture (ING-R8/FIL-R1). Each
function takes the service (its session / injected resolver / object store) explicitly;
all behavior is unchanged from the pre-split module.
"""

from __future__ import annotations

import datetime as _dt
import uuid
from typing import TYPE_CHECKING, Any, cast

from sqlalchemy import Table, select
from sqlalchemy.ext.asyncio import AsyncSession

from wattwise_core.domain.candidate import GboCandidate
from wattwise_core.domain.enums import GboType
from wattwise_core.ingestion import _canonical as _cw
from wattwise_core.ingestion._candidate_store import (
    persist_candidates_bulk,
    persist_quarantined,
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
from wattwise_core.ingestion.capability import UndeclaredGboTypeError
from wattwise_core.ingestion.trust import load_trust_policy
from wattwise_core.ingestion.validation import validate_candidate
from wattwise_core.persistence.localdate import (
    MissingReferenceTimezone,
    project_local_date,
    project_local_wall_clock,
)
from wattwise_core.persistence.models import (
    Activity,
    Athlete,
    SourceCandidate,
    SourceDescriptor,
)
from wattwise_core.persistence.types import uuid7
from wattwise_core.persistence.upsert import upsert
from wattwise_core.storage import create_object_store

if TYPE_CHECKING:
    from wattwise_core.ingestion.ingest import IngestResult, IngestService

_IDENTITY_WINDOW = _dt.timedelta(hours=2)


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
    # MAP-R2/MAP-R6 validation gate BEFORE persistence: a candidate carrying a
    # non-canonical key or violating a canonical invariant is QUARANTINED — persisted
    # with its lineage + the failing rule id, excluded from resolution, never partially
    # written into the canonical store.
    passing: list[GboCandidate] = []
    for cand in batch:
        rule_id = validate_candidate(cand)
        if rule_id is None:
            passing.append(cand)
            continue
        try:
            async with svc._session.begin_nested():
                await persist_quarantined(
                    svc._session, athlete, descriptor, cand, connection_id, run_id,
                    rule_id,
                )
        except Exception:
            result.candidates_failed += 1  # ING-UPS-R3 record isolation; keep the run
        else:
            result.candidates_quarantined += 1
    prepared, failed = await prepare_batch(
        svc._session, athlete, descriptor, passing, connection_id, run_id,
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
                activity_id = await _resolve_and_write_activity(svc, athlete, row, cand)
                await _capture_original(
                    svc, athlete, descriptor, activity_id,
                    files_by_native.get(cand.source_native_id), cand.fetched_at,
                )
                result.activities_written.add(str(activity_id))
            elif cand.gbo_type == GboType.DAILY_WELLNESS.value:
                wellness_dates.add(_parse_date(cand.payload["local_date"]))
            else:  # unreachable after require_declared_types; refuse, never drop (ADP-R3)
                raise UndeclaredGboTypeError(str(cand.gbo_type), "")
    except UndeclaredGboTypeError:
        raise  # ADP-R3: a typed REFUSAL, never silently counted as a failed record
    except Exception:
        result.candidates_failed += 1  # ING-UPS-R3 record isolation; keep the run
        return
    result.candidates_persisted += 1


async def _resolve_and_write_activity(
    svc: IngestService, athlete: uuid.UUID, row: SourceCandidate, cand: GboCandidate
) -> uuid.UUID:
    """Resolve identity (reusing the row's prior id) and write the canonical activity."""
    if row.resolved_activity_id is not None:
        activity_id = row.resolved_activity_id  # ING-R6: reuse the resolved identity
    else:
        activity_id, decision = await _resolve_activity_id(svc, athlete, cand)
        row.resolved_activity_id = activity_id
        # MAP-R12: persist the identity decision (rule fired, score, matched ids)
        # so the merge is explainable and reversible by an explicit split.
        row.identity_resolution = decision
        await svc._session.flush()
    await _write_activity_canonical(svc, athlete, activity_id)
    return activity_id


async def _resolve_activity_id(
    svc: IngestService, athlete: uuid.UUID, cand: GboCandidate
) -> tuple[uuid.UUID, dict[str, Any]]:
    """Resolve a NEW candidate to a canonical activity id (MAP-R9..R12, DEDUP-R7).

    Two legs, in order:

    1. STRONG-FINGERPRINT, regardless of the time window (MAP-R10): a candidate
       carrying a TYPED ``strong_fingerprint`` (a real shared device/file UUID —
       never the per-source ``source_native_id`` dedup key) is matched against
       retained candidates with the SAME fingerprint; the resolver still gates on
       sport compatibility before merging.
    2. WINDOWED fuzzy match (conservative, DEDUP-R7): existing activities whose
       ``start_time`` is within ``_IDENTITY_WINDOW`` (±2h), in a stable order
       (start_time, then activity_id), through the fuzzy start/duration/sport
       matcher; first match wins, else a new id is minted.

    Returns ``(activity_id, decision)`` where ``decision`` is the MAP-R12 record
    (rule that fired, match score, matched ids) persisted on the candidate row.
    """
    start = _parse_start_time(cand.payload["start_time"])
    duration = float(cast("float", cand.payload.get("elapsed_time_s") or 0))
    sport = str(cand.payload.get("sport") or "other")
    matched = await _fingerprint_match(svc, athlete, cand, start, duration, sport)
    if matched is not None:
        return matched
    for act in await _windowed_activities(svc._session, athlete, start):
        # SQLite returns tz-naive datetimes; coerce to UTC for the matcher (GBO-R32).
        act_start = _parse_start_time(act.start_time)
        if svc._resolver.resolve_activity_identity(
            start, duration, sport, None,
            act_start, float(act.elapsed_time_s or 0), act.sport, None,
        ):
            decision = {
                "rule": "windowed_fuzzy",
                "match_score": _window_score(start, act_start),
                "matched_activity_id": str(act.activity_id),
            }
            return act.activity_id, decision
    return uuid7(), {"rule": "no_match_new_record", "match_score": 0.0}


async def _fingerprint_match(
    svc: IngestService,
    athlete: uuid.UUID,
    cand: GboCandidate,
    start: _dt.datetime,
    duration: float,
    sport: str,
) -> tuple[uuid.UUID, dict[str, Any]] | None:
    """The MAP-R10 strong-fingerprint leg: match retained candidates cross-window.

    Considers only CONTRIBUTING candidates (not superseded/tombstoned/quarantined,
    active descriptor) that carry the SAME typed fingerprint and already resolved to
    a canonical activity, in a stable order. The resolver's sport gate still applies
    (a shared fingerprint must never merge incompatible sports).
    """
    if cand.strong_fingerprint is None:
        return None
    stmt = _contributing(
        select(SourceCandidate).where(
            SourceCandidate.athlete_id == athlete,
            SourceCandidate.gbo_type == GboType.ACTIVITY,
            SourceCandidate.strong_fingerprint == cand.strong_fingerprint,
            SourceCandidate.resolved_activity_id.is_not(None),
        )
    ).order_by(SourceCandidate.source_candidate_id)
    for row in (await svc._session.execute(stmt)).scalars().all():
        row_start = _parse_start_time(row.payload.get("start_time"))
        row_duration = float(cast("float", row.payload.get("elapsed_time_s") or 0))
        row_sport = str(row.payload.get("sport") or "other")
        if svc._resolver.resolve_activity_identity(
            start, duration, sport, cand.strong_fingerprint,
            row_start, row_duration, row_sport, row.strong_fingerprint,
        ):
            decision = {
                "rule": "strong_fingerprint",
                "match_score": 1.0,
                "matched_activity_id": str(row.resolved_activity_id),
                "matched_candidate_ids": [str(row.source_candidate_id)],
            }
            return cast("uuid.UUID", row.resolved_activity_id), decision
    return None


async def _windowed_activities(
    session: AsyncSession, athlete: uuid.UUID, start: _dt.datetime
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
    return list((await session.execute(stmt)).scalars().all())


async def _write_activity_canonical(
    svc: IngestService, athlete: uuid.UUID, activity_id: uuid.UUID
) -> None:
    """Resolve every field across candidates and write the canonical activity (UPS-R2).

    The canonical row is persisted through the atomic upsert seam keyed on the
    resolved canonical key ``activity_id`` — a single insert-or-update, never a
    ``session.get`` then add/setattr check-then-write, so two sync runs landing the
    same resolved activity cannot race (UPS-R2). Only the resolved columns are
    supplied, so unresolved fields keep their prior canonical value when the key exists.
    """
    candidates = await _activity_candidates(svc._session, athlete, activity_id)
    if not candidates:
        return
    policy = await load_trust_policy(svc._session, athlete, candidates)
    scalars, coverage, field_resolution = _resolve_scalars(
        candidates, _ACTIVITY_SCALARS, policy, svc._resolver
    )
    local_projection = await _project_local(svc._session, athlete, activity_id, scalars)
    values, update_columns = _activity_values(
        activity_id, athlete, scalars, coverage, local_projection,
        policy_version=policy.policy_version,  # CONF-R6: recorded with the values
        field_resolution=field_resolution,  # LIN-R3: per-field resolution record
    )
    await upsert(
        svc._session,
        cast("Table", Activity.__table__),
        values,
        conflict_keys=["activity_id"],
        update_columns=update_columns,
    )
    await svc._session.flush()
    # Streams resolve PER CHANNEL under each channel's effective tier (CONF-R3/SF-3);
    # an empty policy makes this the candidate's adapter tier (the prior behaviour).
    streams = _cw.resolve_streams(candidates, policy)
    best = _highest_trust(candidates)
    laps = cast("list[dict[str, Any]]", best.payload.get("laps") or [])
    if streams:
        await _cw.upsert_stream_set(svc._session, activity_id, streams)
    await _cw.upsert_laps(svc._session, activity_id, laps, _LAP_SCALARS)


async def _project_local(
    session: AsyncSession, athlete: uuid.UUID, activity_id: uuid.UUID, scalars: dict[str, Any]
) -> tuple[_dt.datetime, _dt.date] | None:
    """Project the resolved UTC ``start_time`` into the athlete's local day (GBO-R33/R35).

    Returns ``(start_time_local, local_date)`` — the display wall-clock and the reproducible
    day-attribution bucket — or ``None`` when ``start_time`` did not resolve (nothing to
    project). The athlete's reference timezone is effective-dated (GBO-R34): a re-ingest of
    an instant predating a later relocation keeps the ``local_date`` it already carries
    (passed as ``prior_local_date``), so a relocation never retroactively re-buckets prior
    days. A missing/unresolvable reference timezone raises
    :class:`~wattwise_core.persistence.localdate.MissingReferenceTimezone`, isolating the
    record (fail-closed, CFG-R1a/R6) — never a silent UTC default.
    """
    raw_start = scalars.get("start_time")
    if raw_start is None:
        return None  # no instant resolved → nothing to bucket
    start = _parse_start_time(raw_start)
    owner = await session.get(Athlete, athlete)
    if owner is None:
        raise MissingReferenceTimezone("athlete row missing for local-date projection")
    existing = await session.get(Activity, activity_id)
    prior = existing.local_date if existing is not None else None
    local_date = project_local_date(start, owner, prior_local_date=prior)
    return project_local_wall_clock(start, owner), local_date


async def _write_wellness(
    svc: IngestService, athlete: uuid.UUID, local_date: _dt.date
) -> None:
    """Resolve daily wellness across ALL candidates for the date (CONF-R2/ING-UPS-R5)."""
    candidates = await _wellness_candidates(svc._session, athlete, local_date)
    policy = await load_trust_policy(svc._session, athlete, candidates)
    # Wellness fields resolve under the whole-source effective tier; an empty policy
    # makes this the candidate's adapter tier (byte-identical to the prior behaviour).
    await _cw.write_wellness_canonical(
        svc._session, athlete, local_date, candidates,
        _whole_source_tier_of(policy),
        policy_version=policy.policy_version,  # CONF-R6
    )


async def _capture_original(
    svc: IngestService,
    athlete: uuid.UUID,
    descriptor: uuid.UUID,
    activity_id: uuid.UUID,
    original: OriginalFile | None,
    fetched_at: _dt.datetime | None,
) -> None:
    """Store the verbatim original file + its activity_file reference (ING-R8/FIL-R1)."""
    if original is None:
        return  # a direct-API source has no original recording file -> no ActivityFile
    store = svc._object_store or create_object_store()
    await _cw.create_activity_file(
        svc._session, store, athlete=athlete, activity_id=activity_id,
        source_descriptor_id=descriptor, original=original, fetched_at=fetched_at,
    )


def _contributing(stmt: Any) -> Any:
    """Restrict a candidate select to rows allowed to CONTRIBUTE to resolution.

    Excluded (each one a distinct lifecycle state, never silently re-included):
    superseded versions (UPS-R5), tombstones (UPS-R5 source-side deletion), quarantined
    candidates (MAP-R6 failed validation), and candidates of a DEACTIVATED source
    descriptor (EVOL-R2: disabling a source is configuration; its retained rows stop
    contributing but stay durably stored for reversibility, DM-SUB-R5).
    """
    return (
        stmt.join(
            SourceDescriptor,
            SourceDescriptor.source_descriptor_id == SourceCandidate.source_descriptor_id,
        )
        .where(
            SourceCandidate.is_superseded.is_(False),
            SourceCandidate.is_tombstone.is_(False),
            SourceCandidate.quarantine_rule_id.is_(None),
            SourceDescriptor.is_active.is_(True),
        )
    )


async def _activity_candidates(
    session: AsyncSession, athlete: uuid.UUID, activity_id: uuid.UUID
) -> list[SourceCandidate]:
    """All CONTRIBUTING activity candidates resolved to ``activity_id`` (the resolution set)."""
    stmt = _contributing(
        select(SourceCandidate).where(
            SourceCandidate.athlete_id == athlete,
            SourceCandidate.gbo_type == GboType.ACTIVITY,
            SourceCandidate.resolved_activity_id == activity_id,
        )
    )
    return list((await session.execute(stmt)).scalars().all())


async def _wellness_candidates(
    session: AsyncSession, athlete: uuid.UUID, local_date: _dt.date
) -> list[SourceCandidate]:
    """All CONTRIBUTING daily-wellness candidates for ``local_date`` (the resolution set)."""
    stmt = _contributing(
        select(SourceCandidate).where(
            SourceCandidate.athlete_id == athlete,
            SourceCandidate.gbo_type == GboType.DAILY_WELLNESS,
        )
    )
    rows = (await session.execute(stmt)).scalars().all()
    return [c for c in rows if _parse_date(c.payload.get("local_date")) == local_date]


def _window_score(a: _dt.datetime, b: _dt.datetime) -> float:
    """A [0,1] closeness score for a windowed match (MAP-R12 decision record).

    1.0 = identical start instants, linearly decaying to 0.0 at the edge of the
    ±2h identity window. Descriptive audit data only — never a matching input.
    """
    delta = abs((a - b).total_seconds())
    window = _IDENTITY_WINDOW.total_seconds()
    return max(0.0, 1.0 - delta / window)


__all__ = [
    "_activity_candidates",
    "_land_batch",
    "_resolve_activity_id",
    "_wellness_candidates",
    "_write_activity_canonical",
    "_write_wellness",
]
