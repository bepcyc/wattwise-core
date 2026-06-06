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

from wattwise_core.domain.candidate import FieldCandidate, GboCandidate
from wattwise_core.domain.enums import (
    Fidelity,
    GboType,
    SampleBasis,
    StreamChannelName,
    StreamSetKind,
    trust_rank,
)
from wattwise_core.ingestion.dedup import resolve_activity_identity, resolve_field
from wattwise_core.persistence.models import (
    Activity,
    ActivityLap,
    ActivityStreamSet,
    DailyWellness,
    SourceCandidate,
    StreamChannel,
)
from wattwise_core.persistence.types import utcnow, uuid7

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

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def ingest(
        self,
        athlete_id: str | uuid.UUID,
        source_descriptor_id: str | uuid.UUID,
        candidates: list[GboCandidate],
        *,
        connection_id: str | uuid.UUID | None = None,
        ingest_run_id: uuid.UUID | None = None,
    ) -> IngestResult:
        """Land a batch of candidates into the canonical store (one transaction)."""
        athlete = _uid(athlete_id)
        descriptor = _uid(source_descriptor_id)
        run_id = ingest_run_id or uuid7()
        result = IngestResult()
        for cand in candidates:
            row = await self._persist_candidate(athlete, descriptor, cand, connection_id, run_id)
            result.candidates_persisted += 1
            if cand.gbo_type == GboType.ACTIVITY.value:
                activity_id = await self._resolve_activity_id(athlete, cand)
                row.resolved_activity_id = activity_id
                await self._session.flush()
                await self._write_activity_canonical(athlete, activity_id)
                result.activities_written.add(str(activity_id))
            elif cand.gbo_type == GboType.DAILY_WELLNESS.value:
                await _write_wellness_canonical(self._session, athlete, cand)
                result.wellness_written += 1
        return result

    async def _persist_candidate(
        self,
        athlete: uuid.UUID,
        descriptor: uuid.UUID,
        cand: GboCandidate,
        connection_id: str | uuid.UUID | None,
        run_id: uuid.UUID,
    ) -> SourceCandidate:
        """Upsert the source candidate on its natural key; retain the mapped payload."""
        stmt = select(SourceCandidate).where(
            SourceCandidate.athlete_id == athlete,
            SourceCandidate.source_descriptor_id == descriptor,
            SourceCandidate.source_native_id == cand.source_native_id,
            SourceCandidate.gbo_type == GboType(cand.gbo_type),
        )
        existing = (await self._session.execute(stmt)).scalar_one_or_none()
        if existing is not None:
            # Unchanged content -> value-level no-op (UPS-R3): only refresh fetch metadata.
            if existing.content_hash != cand.content_hash:
                existing.content_hash = cand.content_hash
                existing.payload = _jsonsafe(cand.payload)
                existing.trust_profile = {"tier": cand.trust_tier.value}
                existing.confidence = cand.confidence
            existing.fetched_at = cand.fetched_at
            existing.ingest_run_id = run_id
            return existing
        row = SourceCandidate(
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
        self._session.add(row)
        await self._session.flush()
        return row

    async def _resolve_activity_id(self, athlete: uuid.UUID, cand: GboCandidate) -> uuid.UUID:
        """Resolve the candidate to a canonical activity id (MAP-R9..R12, DEDUP-R7).

        Reuses an existing canonical activity when the conservative matcher (start-time
        window + duration tolerance + compatible sport, or a shared fingerprint) is
        satisfied; otherwise mints a new id. Conservative: ambiguity stays separate.
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
            if resolve_activity_identity(
                start, duration, sport, fingerprint,
                act_start, float(act.elapsed_time_s or 0), act.sport, None,
            ):
                return act.activity_id
        return uuid7()

    async def _write_activity_canonical(
        self, athlete: uuid.UUID, activity_id: uuid.UUID
    ) -> None:
        """Resolve every field across candidates and write the canonical activity."""
        candidates = await self._activity_candidates(athlete, activity_id)
        if not candidates:
            return
        scalars = _resolve_scalars(candidates, _ACTIVITY_SCALARS)
        act = await self._session.get(Activity, activity_id)
        if act is None:
            act = Activity(activity_id=activity_id, athlete_id=athlete, sport="other")
            self._session.add(act)
        _apply_activity_scalars(act, scalars)
        await self._session.flush()
        best = _highest_trust(candidates)
        streams = cast("dict[str, Any]", best.payload.get("streams") or {})
        laps = cast("list[dict[str, Any]]", best.payload.get("laps") or [])
        if streams:
            await _upsert_stream_set(self._session, activity_id, streams)
        await _upsert_laps(self._session, activity_id, laps)

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


# --- module-level write helpers (stateless; take the session explicitly) ---


async def _upsert_stream_set(
    session: AsyncSession, activity_id: uuid.UUID, streams: dict[str, Any]
) -> None:
    """Get-or-create the activity stream set and upsert each channel."""
    stmt = select(ActivityStreamSet).where(ActivityStreamSet.activity_id == activity_id)
    stream_set = (await session.execute(stmt)).scalar_one_or_none()
    if stream_set is None:
        first = next(iter(streams.values()))
        stream_set = ActivityStreamSet(
            activity_id=activity_id,
            sample_basis=SampleBasis(first.get("sample_basis", "time")),
            sample_rate_hz=first.get("sample_rate_hz", 1.0),
            sample_count=len(first.get("values", [])),
            t0=utcnow(),
        )
        session.add(stream_set)
        await session.flush()
    for name, chan in streams.items():
        await _upsert_channel(session, stream_set.stream_set_id, name, chan)


async def _upsert_channel(
    session: AsyncSession, stream_set_id: uuid.UUID, name: str, chan: dict[str, Any]
) -> None:
    stmt = select(StreamChannel).where(
        StreamChannel.stream_set_id == stream_set_id,
        StreamChannel.channel == StreamChannelName(name),
    )
    existing = (await session.execute(stmt)).scalar_one_or_none()
    values = chan.get("values", [])
    if existing is None:
        session.add(
            StreamChannel(
                stream_set_id=stream_set_id,
                set_kind=StreamSetKind.ACTIVITY,
                channel=StreamChannelName(name),
                sample_basis=SampleBasis(chan.get("sample_basis", "time")),
                values=values,
                coverage={},
            )
        )
    else:
        existing.values = values


async def _upsert_laps(
    session: AsyncSession, activity_id: uuid.UUID, laps: list[dict[str, Any]]
) -> None:
    for lap in laps:
        idx = int(lap["lap_index"])
        stmt = select(ActivityLap).where(
            ActivityLap.activity_id == activity_id, ActivityLap.lap_index == idx
        )
        existing = (await session.execute(stmt)).scalar_one_or_none()
        fields = {k: lap.get(k) for k in _LAP_SCALARS}
        if existing is None:
            session.add(ActivityLap(activity_id=activity_id, lap_index=idx, **fields))
        else:
            for k, v in fields.items():
                setattr(existing, k, v)


async def _write_wellness_canonical(
    session: AsyncSession, athlete: uuid.UUID, cand: GboCandidate
) -> None:
    """Upsert the daily wellness row on (athlete_id, local_date) (GBO-R24)."""
    payload = cand.payload
    local_date = _parse_date(payload["local_date"])
    stmt = select(DailyWellness).where(
        DailyWellness.athlete_id == athlete, DailyWellness.local_date == local_date
    )
    row = (await session.execute(stmt)).scalar_one_or_none()
    if row is None:
        row = DailyWellness(athlete_id=athlete, local_date=local_date, coverage={})
        session.add(row)
    for key in ("resting_hr_bpm", "hrv_rmssd_ms", "hrv_sdnn_ms", "sleep_score", "steps"):
        if payload.get(key) is not None:
            setattr(row, key, payload[key])
    await session.flush()


# --- module-level pure helpers (resolution + coercion) ---


def _resolve_scalars(
    candidates: list[SourceCandidate], fields: tuple[str, ...]
) -> dict[str, Any]:
    """Resolve each scalar field across candidates via the conflict resolver (CONF-R2)."""
    resolved: dict[str, Any] = {}
    for fname in fields:
        contributors = [
            FieldCandidate(
                value=c.payload[fname],
                trust_tier=_tier_of(c),
                source_descriptor_id=str(c.source_descriptor_id),
                confidence=float(c.confidence) if c.confidence is not None else 1.0,
                observed_at=c.observed_at,
                fetched_at=c.fetched_at,
            )
            for c in candidates
            if c.payload.get(fname) is not None
        ]
        winner = resolve_field(contributors)
        if winner is not None:
            resolved[fname] = winner.value
    return resolved


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


__all__ = ["IngestResult", "IngestService"]
