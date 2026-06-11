"""Ingest identity resolution + contributing-candidate selection (MAP-R9..R12, DEDUP-R7).

The focused sibling of :mod:`wattwise_core.ingestion._ingest_steps` (QUAL-R9 size split) that
owns the two-leg activity identity resolution — the cross-window MAP-R10 strong-fingerprint leg
and the conservative ±2h windowed fuzzy leg (DEDUP-R7) with its MAP-R12 decision record — plus
the CONTRIBUTING-candidate selection shared by the canonical writes and the re-resolution path
(superseded/tombstoned/quarantined/deactivated rows never contribute, UPS-R5 / MAP-R6 / EVOL-R2).
Each function takes the service (its session / injected resolver) explicitly; all behavior is
unchanged from the pre-split module, and ``_ingest_steps`` re-exports the shared names so every
historical ``from wattwise_core.ingestion._ingest_steps import ...`` path stays stable.
"""

from __future__ import annotations

import datetime as _dt
import uuid
from typing import TYPE_CHECKING, Any, cast

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from wattwise_core.domain.candidate import GboCandidate
from wattwise_core.domain.enums import GboType
from wattwise_core.ingestion._mapping import _parse_date, _parse_start_time
from wattwise_core.persistence.models import (
    Activity,
    SourceCandidate,
    SourceDescriptor,
)
from wattwise_core.persistence.types import uuid7

if TYPE_CHECKING:
    from wattwise_core.ingestion.ingest import IngestService

_IDENTITY_WINDOW = _dt.timedelta(hours=2)


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
            start,
            duration,
            sport,
            None,
            act_start,
            float(act.elapsed_time_s or 0),
            act.sport,
            None,
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
            start,
            duration,
            sport,
            cand.strong_fingerprint,
            row_start,
            row_duration,
            row_sport,
            row.strong_fingerprint,
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


def _contributing(stmt: Any) -> Any:
    """Restrict a candidate select to rows allowed to CONTRIBUTE to resolution.

    Excluded (each one a distinct lifecycle state, never silently re-included):
    superseded versions (UPS-R5), tombstones (UPS-R5 source-side deletion), quarantined
    candidates (MAP-R6 failed validation), and candidates of a DEACTIVATED source
    descriptor (EVOL-R2: disabling a source is configuration; its retained rows stop
    contributing but stay durably stored for reversibility, DM-SUB-R5).
    """
    return stmt.join(
        SourceDescriptor,
        SourceDescriptor.source_descriptor_id == SourceCandidate.source_descriptor_id,
    ).where(
        SourceCandidate.is_superseded.is_(False),
        SourceCandidate.is_tombstone.is_(False),
        SourceCandidate.quarantine_rule_id.is_(None),
        SourceDescriptor.is_active.is_(True),
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
    "_resolve_activity_id",
    "_wellness_candidates",
]
