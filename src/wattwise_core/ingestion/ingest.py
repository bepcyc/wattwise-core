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
"""

from __future__ import annotations

import datetime as _dt
import uuid
from dataclasses import dataclass, field
from typing import Any, cast

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from wattwise_core.domain.candidate import GboCandidate
from wattwise_core.domain.enums import Fidelity, GboType, trust_rank
from wattwise_core.ingestion import _canonical as _cw
from wattwise_core.ingestion._canonical import OriginalFile
from wattwise_core.ingestion.dedup import resolve_activity_identity, resolve_field
from wattwise_core.persistence.models import Activity, SourceCandidate
from wattwise_core.persistence.types import utcnow, uuid7
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


class IngestService:
    """Persists adapter candidates and resolves them into canonical records."""

    def __init__(self, session: AsyncSession, *, object_store: ObjectStore | None = None) -> None:
        self._session = session
        self._object_store = object_store

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
        """Land a batch of candidates into the canonical store (one transaction).

        ``original_files`` carries the verbatim tier-1 recording artifacts (a
        ``file_import`` upload supplies them; a direct-API source supplies none); each
        is stored verbatim and linked to its resolved activity via ``activity_file``
        (ING-R8/FIL-R1). Wellness candidates are batched per ``local_date`` so the row
        is resolved across ALL same-day candidates (CONF-R2), never last-write-wins.
        """
        athlete = _uid(athlete_id)
        descriptor = _uid(source_descriptor_id)
        run_id = ingest_run_id or uuid7()
        files_by_native = {f.source_native_id: f for f in (original_files or [])}
        result = IngestResult()
        wellness_dates: set[_dt.date] = set()
        for cand in candidates:
            row = await _persist_candidate(
                self._session, athlete, descriptor, cand, connection_id, run_id
            )
            result.candidates_persisted += 1
            if cand.gbo_type == GboType.ACTIVITY.value:
                activity_id = await self._resolve_and_write_activity(athlete, row, cand)
                await self._capture_original(
                    athlete, descriptor, activity_id, files_by_native.get(cand.source_native_id),
                    cand.fetched_at,
                )
                result.activities_written.add(str(activity_id))
            elif cand.gbo_type == GboType.DAILY_WELLNESS.value:
                wellness_dates.add(_parse_date(cand.payload["local_date"]))
        for local_date in wellness_dates:
            await self._write_wellness(athlete, local_date)
            result.wellness_written += 1
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

        Reuses an existing canonical activity when the conservative matcher (start-time
        window + duration tolerance + compatible sport, OR a shared fingerprint derived
        from a stored candidate, MAP-R10) is satisfied; otherwise mints a new id.
        """
        start = _parse_start_time(cand.payload["start_time"])
        duration = float(cast("float", cand.payload.get("elapsed_time_s") or 0))
        sport = str(cand.payload.get("sport") or "other")
        fingerprint = cand.source_native_id
        lo, hi = start - _IDENTITY_WINDOW, start + _IDENTITY_WINDOW
        stmt = select(Activity).where(
            Activity.athlete_id == athlete,
            Activity.start_time >= lo,
            Activity.start_time <= hi,
        )
        for act in (await self._session.execute(stmt)).scalars().all():
            # SQLite returns tz-naive datetimes; coerce to UTC for the matcher (GBO-R32).
            act_start = _parse_start_time(act.start_time)
            act_fp = await self._activity_fingerprint(athlete, act.activity_id)
            if resolve_activity_identity(
                start, duration, sport, fingerprint,
                act_start, float(act.elapsed_time_s or 0), act.sport, act_fp,
            ):
                return act.activity_id
        return uuid7()

    async def _activity_fingerprint(
        self, athlete: uuid.UUID, activity_id: uuid.UUID
    ) -> str | None:
        """The stored activity's identity fingerprint from a resolved candidate (MAP-R10)."""
        stmt = select(SourceCandidate.source_native_id).where(
            SourceCandidate.athlete_id == athlete,
            SourceCandidate.resolved_activity_id == activity_id,
            SourceCandidate.is_superseded.is_(False),
        )
        return (await self._session.execute(stmt)).scalars().first()

    async def _write_activity_canonical(
        self, athlete: uuid.UUID, activity_id: uuid.UUID
    ) -> None:
        """Resolve every field across candidates and write the canonical activity."""
        candidates = await self._activity_candidates(athlete, activity_id)
        if not candidates:
            return
        scalars, coverage = _resolve_scalars(candidates, _ACTIVITY_SCALARS)
        act = await self._session.get(Activity, activity_id)
        if act is None:
            act = Activity(activity_id=activity_id, athlete_id=athlete, sport="other")
            self._session.add(act)
        _apply_activity_scalars(act, scalars)
        act.coverage = coverage
        await self._session.flush()
        best = _highest_trust(candidates)
        streams = cast("dict[str, Any]", best.payload.get("streams") or {})
        laps = cast("list[dict[str, Any]]", best.payload.get("laps") or [])
        if streams:
            await _cw.upsert_stream_set(self._session, activity_id, streams)
        await _cw.upsert_laps(self._session, activity_id, laps, _LAP_SCALARS)

    async def _write_wellness(self, athlete: uuid.UUID, local_date: _dt.date) -> None:
        """Resolve daily wellness across ALL candidates for the date (CONF-R2/ING-UPS-R5)."""
        candidates = await self._wellness_candidates(athlete, local_date)
        await _cw.write_wellness_canonical(
            self._session, athlete, local_date, candidates, _tier_of
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

    async def _activity_candidates(
        self, athlete: uuid.UUID, activity_id: uuid.UUID
    ) -> list[SourceCandidate]:
        stmt = select(SourceCandidate).where(
            SourceCandidate.athlete_id == athlete,
            SourceCandidate.gbo_type == GboType.ACTIVITY,
            SourceCandidate.resolved_activity_id == activity_id,
            SourceCandidate.is_superseded.is_(False),
        )
        return list((await self._session.execute(stmt)).scalars().all())

    async def _wellness_candidates(
        self, athlete: uuid.UUID, local_date: _dt.date
    ) -> list[SourceCandidate]:
        stmt = select(SourceCandidate).where(
            SourceCandidate.athlete_id == athlete,
            SourceCandidate.gbo_type == GboType.DAILY_WELLNESS,
            SourceCandidate.is_superseded.is_(False),
        )
        rows = (await self._session.execute(stmt)).scalars().all()
        return [c for c in rows if _parse_date(c.payload.get("local_date")) == local_date]


# --- module-level write/resolution helpers (stateless; take the session) ---


async def _persist_candidate(
    session: AsyncSession,
    athlete: uuid.UUID,
    descriptor: uuid.UUID,
    cand: GboCandidate,
    connection_id: str | uuid.UUID | None,
    run_id: uuid.UUID,
) -> SourceCandidate:
    """Upsert the source candidate; supersede-and-version on a CHANGED re-ingest.

    Unchanged content is a value-level no-op (UPS-R3). A CHANGED re-ingest (same
    candidate key, new ``content_hash``) marks the prior row ``is_superseded=True`` and
    INSERTS a NEW candidate version, carrying its ``resolved_activity_id`` forward
    (ING-R6), preserving the prior for audit (PRV-R2 / UPS-R5) rather than mutating it.
    """
    stmt = select(SourceCandidate).where(
        SourceCandidate.athlete_id == athlete,
        SourceCandidate.source_descriptor_id == descriptor,
        SourceCandidate.source_native_id == cand.source_native_id,
        SourceCandidate.gbo_type == GboType(cand.gbo_type),
        SourceCandidate.is_superseded.is_(False),
    )
    existing = (await session.execute(stmt)).scalar_one_or_none()
    if existing is not None and existing.content_hash == cand.content_hash:
        existing.fetched_at = cand.fetched_at  # UPS-R3 no-op: refresh fetch metadata only
        existing.ingest_run_id = run_id
        return existing
    prior_activity_id = None
    if existing is not None:
        # PRV-R2: preserve the prior version for audit. The candidate-key unique
        # constraint admits only ONE row per key, so the superseded row's
        # source_native_id is version-tagged (its observed identity is retained in the
        # untouched payload/content_hash) and the NEW row reclaims the canonical key.
        existing.is_superseded = True
        existing.source_native_id = _superseded_native_id(existing)
        prior_activity_id = existing.resolved_activity_id
        await session.flush()
    row = _new_candidate(athlete, descriptor, cand, connection_id, run_id)
    row.resolved_activity_id = prior_activity_id  # carry the resolved identity forward (ING-R6)
    session.add(row)
    await session.flush()
    return row


def _superseded_native_id(prior: SourceCandidate) -> str:
    """A version-tagged native id freeing the canonical key for the new version (PRV-R2).

    The prior row stays fully readable for audit (its payload/content_hash are
    untouched); only its candidate-key slot is vacated so the new version can hold the
    canonical key under the single-row unique constraint.
    """
    tag = f"#superseded:{prior.content_hash[:16]}"
    base = prior.source_native_id.split("#superseded:", 1)[0]
    return f"{base}{tag}"


def _new_candidate(
    athlete: uuid.UUID,
    descriptor: uuid.UUID,
    cand: GboCandidate,
    connection_id: str | uuid.UUID | None,
    run_id: uuid.UUID,
) -> SourceCandidate:
    return SourceCandidate(
        athlete_id=athlete,
        source_descriptor_id=descriptor,
        connection_id=_uid(connection_id) if connection_id else None,
        source_native_id=cand.source_native_id,
        gbo_type=GboType(cand.gbo_type),
        observed_at=cand.observed_at,
        fetched_at=cand.fetched_at,
        content_hash=cand.content_hash,
        adapter_version=cand.adapter_version,
        mapping_version=cand.mapping_version,
        trust_profile={"tier": cand.trust_tier.value},
        payload=_jsonsafe(cand.payload),
        confidence=cand.confidence,
        ingest_run_id=run_id,
        untrusted_content=cand.untrusted_content,
    )


def _resolve_scalars(
    candidates: list[SourceCandidate], fields: tuple[str, ...]
) -> tuple[dict[str, Any], dict[str, object]]:
    """Resolve each scalar field across candidates + build its coverage (CONF-R2/R5).

    Returns ``(resolved_values, coverage)``. A field whose >=2 contributors materially
    disagree beyond the per-field dispute tolerance gets ``coverage.disputed=True`` —
    the best value is still selected, the disagreement is surfaced not hidden (CONF-R5).
    """
    resolved: dict[str, Any] = {}
    coverage: dict[str, object] = {}
    for fname in fields:
        contributors = _cw.field_candidates(candidates, fname, _tier_of)
        winner = resolve_field(contributors, dispute_tolerance=_cw.dispute_tolerance(fname))
        if winner is None:
            continue
        resolved[fname] = winner.value
        coverage[fname] = _cw.coverage_for(
            True, contributors[0].trust_tier, disputed=winner.disputed
        ).to_jsonable()
    return resolved, coverage


def _apply_activity_scalars(act: Activity, scalars: dict[str, Any]) -> None:
    """Write resolved scalars onto the activity, parsing start_time, never zero-filling."""
    for key, value in scalars.items():
        if key == "start_time":
            act.start_time = _parse_start_time(value)
        elif hasattr(act, key):
            setattr(act, key, value)
    act.has_power = scalars.get("avg_power_w") is not None
    act.has_hr = scalars.get("avg_hr_bpm") is not None
    act.updated_at = utcnow()


def _tier_of(candidate: SourceCandidate) -> Fidelity:
    raw = candidate.trust_profile.get("tier", Fidelity.PLATFORM_COMPUTED.value)
    return Fidelity(str(raw))


def _highest_trust(candidates: list[SourceCandidate]) -> SourceCandidate:
    return min(candidates, key=lambda c: (trust_rank(_tier_of(c)), str(c.source_descriptor_id)))


def _jsonsafe(value: Any) -> Any:
    """Coerce a mapped payload to JSON-storable form (datetimes/dates -> ISO strings)."""
    if isinstance(value, _dt.datetime | _dt.date):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: _jsonsafe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonsafe(v) for v in value]
    return value


def _parse_start_time(value: Any) -> _dt.datetime:
    """Parse a stored ISO start_time back to a tz-aware UTC datetime."""
    dt = value if isinstance(value, _dt.datetime) else _dt.datetime.fromisoformat(str(value))
    return dt if dt.tzinfo else dt.replace(tzinfo=_dt.UTC)


def _parse_date(value: Any) -> _dt.date:
    return value if isinstance(value, _dt.date) else _dt.date.fromisoformat(str(value))


def _uid(value: str | uuid.UUID) -> uuid.UUID:
    return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))


__all__ = ["IngestResult", "IngestService", "OriginalFile"]
