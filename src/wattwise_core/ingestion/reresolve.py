"""Re-resolution lifecycle: source withdrawal/return, tombstones, splits (E3).

Owning requirements: CONF-R6 (re-resolution from retained candidates without
re-fetch, under a recorded policy version), EVOL-R2 (disabling a source is a
configuration action; affected records re-resolve from the remaining candidates and
coverage degrades honestly), DM-SUB-R5 / doc 30 §9A ING-SUB-R3..R7 (withdrawal
re-resolves each affected channel to the next-best available equivalence-class
member, recording the fidelity downgrade as a coverage signal; re-connection
re-resolves UPWARD automatically), UPS-R5 (a source-side deletion is a typed
tombstone candidate — it removes that source's contribution; the canonical record
persists while other sources still contribute, and is removed only when NO
contributor remains; never a cascade delete of a multi-source record), and MAP-R12
(an explicit, recorded SPLIT operation undoes a mistaken identity merge).

Everything here re-reads the durably retained ``source_candidate`` rows — there is
NO source re-fetch on any path (UPS-R4/CONF-R6) and NO value is ever fabricated: a
channel whose class is fully empty becomes a typed coverage gap (ING-SUB-R6).
"""

from __future__ import annotations

import datetime as _dt
import uuid

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from wattwise_core.domain.enums import GboType
from wattwise_core.ingestion._candidate_store import _superseded_native_id
from wattwise_core.ingestion._mapping import _parse_date
from wattwise_core.ingestion.ingest import (
    IngestService,
    _activity_candidates,
    _wellness_candidates,
)
from wattwise_core.persistence.models import (
    Activity,
    ActivityFile,
    ActivityLap,
    ActivityStreamSet,
    DailyWellness,
    SourceCandidate,
    SourceDescriptor,
    StreamChannel,
)
from wattwise_core.persistence.types import utcnow, uuid7
from wattwise_core.seams import ConflictResolver


async def re_resolve_activity(
    session: AsyncSession,
    athlete: uuid.UUID,
    activity_id: uuid.UUID,
    *,
    resolver: ConflictResolver | None = None,
) -> None:
    """Re-run conflict resolution for ONE activity from retained candidates (CONF-R6).

    Reads ONLY the durable ``source_candidate`` store (no re-fetch); the canonical row
    is rewritten with the values the CURRENT trust policy resolves, and the policy
    version that produced them is recorded on the row. With zero remaining contributors
    the canonical record is left intact (ONB-R5: previously-ingested canonical data
    stays usable when a source is merely withdrawn — removal happens only through the
    tombstone deletion path).
    """
    svc = IngestService(session, resolver=resolver)
    await svc._write_activity_canonical(athlete, activity_id)
    await session.flush()
    # The canonical write lands through the Core upsert seam, so any identity-mapped
    # ORM instance of the row would be stale: expire cached state so a same-session
    # read observes the re-resolved values, never the pre-lifecycle ones.
    session.expire_all()


async def re_resolve_wellness(
    session: AsyncSession,
    athlete: uuid.UUID,
    local_date: _dt.date,
    *,
    resolver: ConflictResolver | None = None,
) -> None:
    """Re-run conflict resolution for ONE wellness day from retained candidates (CONF-R6)."""
    svc = IngestService(session, resolver=resolver)
    await svc._write_wellness(athlete, local_date)
    await session.flush()
    session.expire_all()  # same-session reads must observe the re-resolved row


async def re_resolve_source_records(
    session: AsyncSession,
    source_descriptor_id: uuid.UUID,
    *,
    resolver: ConflictResolver | None = None,
) -> int:
    """Re-resolve every canonical record the source ever contributed to (no re-fetch).

    The DM-SUB-R5 / ING-SUB-R3 fan-out: each affected activity and wellness day
    re-resolves to the best AVAILABLE contributor under the current policy — downgrades
    surface as ``substituted``/typed-gap coverage, re-connections re-resolve UPWARD.
    Returns the number of canonical records re-resolved.
    """
    rows = (
        await session.execute(
            select(SourceCandidate).where(
                SourceCandidate.source_descriptor_id == source_descriptor_id
            )
        )
    ).scalars().all()
    count = 0
    seen_acts: set[uuid.UUID] = set()
    seen_days: set[tuple[uuid.UUID, _dt.date]] = set()
    for row in rows:
        if row.gbo_type == GboType.ACTIVITY and row.resolved_activity_id is not None:
            key = row.resolved_activity_id
            if key not in seen_acts:
                seen_acts.add(key)
                await re_resolve_activity(session, row.athlete_id, key, resolver=resolver)
                count += 1
        elif row.gbo_type == GboType.DAILY_WELLNESS:
            local_date = _parse_date(row.payload.get("local_date"))
            day = (row.athlete_id, local_date)
            if day not in seen_days:
                seen_days.add(day)
                await re_resolve_wellness(
                    session, row.athlete_id, local_date, resolver=resolver
                )
                count += 1
    return count


async def deactivate_source(
    session: AsyncSession,
    source_descriptor_id: uuid.UUID,
    *,
    resolver: ConflictResolver | None = None,
) -> int:
    """Disable a source as a CONFIGURATION action and degrade gracefully (EVOL-R2).

    Flips ``source_descriptor.is_active`` off — no candidate row is deleted
    (ING-SUB-R2: fully reversible) — then re-resolves every affected canonical record
    from the REMAINING contributors (CONF-R6, no re-fetch). Coverage updates to the
    reduced fidelity honestly: a displaced top member surfaces as ``substituted``
    (DM-SUB-R4) and an emptied channel as a typed gap — never a fabricated value.
    """
    await _set_active(session, source_descriptor_id, active=False)
    return await re_resolve_source_records(
        session, source_descriptor_id, resolver=resolver
    )


async def reactivate_source(
    session: AsyncSession,
    source_descriptor_id: uuid.UUID,
    *,
    resolver: ConflictResolver | None = None,
) -> int:
    """Re-enable a source and re-resolve affected channels UPWARD (DM-SUB-R5/ING-SUB-R7).

    The retained candidates contribute again with zero re-fetch; a higher-fidelity
    member that wins again clears the ``substitution`` marker automatically.
    """
    await _set_active(session, source_descriptor_id, active=True)
    return await re_resolve_source_records(
        session, source_descriptor_id, resolver=resolver
    )


async def _set_active(
    session: AsyncSession, source_descriptor_id: uuid.UUID, *, active: bool
) -> None:
    """Flip the descriptor's ``is_active`` flag (fail-closed on an unknown descriptor)."""
    descriptor = await session.get(SourceDescriptor, source_descriptor_id)
    if descriptor is None:
        raise LookupError(f"unknown source_descriptor {source_descriptor_id}")
    descriptor.is_active = active
    await session.flush()


async def ingest_tombstone(
    session: AsyncSession,
    athlete: uuid.UUID,
    source_descriptor_id: uuid.UUID,
    source_native_id: str,
    gbo_type: GboType,
    *,
    resolver: ConflictResolver | None = None,
) -> bool:
    """Apply a source-side DELETION as a typed tombstone candidate (UPS-R5).

    The current candidate version is superseded by a tombstone row (the deletion is
    itself a versioned, auditable candidate), removing that source's contribution.
    Affected canonical records then re-resolve from the remaining contributors —
    a multi-source record PERSISTS (never cascade-deleted); only when NO contributor
    remains is the canonical record removed (the datum existed solely at the deleting
    source). Returns ``True`` when a live candidate version was found and tombstoned.
    """
    prior = (
        await session.execute(
            select(SourceCandidate).where(
                SourceCandidate.athlete_id == athlete,
                SourceCandidate.source_descriptor_id == source_descriptor_id,
                SourceCandidate.source_native_id == source_native_id,
                SourceCandidate.gbo_type == gbo_type,
                SourceCandidate.is_superseded.is_(False),
                SourceCandidate.is_tombstone.is_(False),
            )
        )
    ).scalar_one_or_none()
    if prior is None:
        return False  # nothing live to delete; idempotent no-op
    prior.is_superseded = True
    tomb_native = prior.source_native_id
    prior.source_native_id = _superseded_native_id(prior)
    session.add(
        SourceCandidate(
            athlete_id=athlete,
            source_descriptor_id=source_descriptor_id,
            source_native_id=tomb_native,
            gbo_type=gbo_type,
            content_hash=f"tombstone:{prior.content_hash[:96]}",
            payload={},
            is_tombstone=True,
            fetched_at=utcnow(),
            resolved_activity_id=prior.resolved_activity_id,
        )
    )
    await session.flush()
    if gbo_type == GboType.ACTIVITY and prior.resolved_activity_id is not None:
        await _resolve_after_activity_deletion(
            session, athlete, prior.resolved_activity_id, resolver=resolver
        )
    elif gbo_type == GboType.DAILY_WELLNESS:
        await _resolve_after_wellness_deletion(session, athlete, prior, resolver=resolver)
    return True


async def _resolve_after_activity_deletion(
    session: AsyncSession,
    athlete: uuid.UUID,
    activity_id: uuid.UUID,
    *,
    resolver: ConflictResolver | None,
) -> None:
    """Re-resolve or remove ONE canonical activity after a tombstone (UPS-R5)."""
    remaining = await _activity_candidates(session, athlete, activity_id)
    if remaining:
        # Other sources still contribute: the record persists and re-resolves —
        # never a cascade delete of a multi-source record (UPS-R5).
        await re_resolve_activity(session, athlete, activity_id, resolver=resolver)
        return
    await _delete_activity_record(session, activity_id)


async def _resolve_after_wellness_deletion(
    session: AsyncSession,
    athlete: uuid.UUID,
    prior: SourceCandidate,
    *,
    resolver: ConflictResolver | None,
) -> None:
    """Re-resolve or remove ONE canonical wellness day after a tombstone (UPS-R5)."""
    local_date = _parse_date(prior.payload.get("local_date"))
    svc = IngestService(session, resolver=resolver)
    remaining = await _wellness_candidates(session, athlete, local_date)
    if remaining:
        await svc._write_wellness(athlete, local_date)
        await session.flush()
        return
    await session.execute(
        delete(DailyWellness).where(
            DailyWellness.athlete_id == athlete,
            DailyWellness.local_date == local_date,
        )
    )
    await session.flush()


async def _delete_activity_record(session: AsyncSession, activity_id: uuid.UUID) -> None:
    """Remove ONE canonical activity + its dependents (last contributor deleted, UPS-R5)."""
    set_ids = (
        await session.execute(
            select(ActivityStreamSet.stream_set_id).where(
                ActivityStreamSet.activity_id == activity_id
            )
        )
    ).scalars().all()
    if set_ids:
        await session.execute(
            delete(StreamChannel).where(StreamChannel.stream_set_id.in_(set_ids))
        )
        await session.execute(
            delete(ActivityStreamSet).where(ActivityStreamSet.activity_id == activity_id)
        )
    await session.execute(delete(ActivityLap).where(ActivityLap.activity_id == activity_id))
    await session.execute(
        delete(ActivityFile).where(ActivityFile.activity_id == activity_id)
    )
    await session.execute(delete(Activity).where(Activity.activity_id == activity_id))
    await session.flush()


async def split_activity(
    session: AsyncSession,
    source_candidate_id: uuid.UUID,
    *,
    resolver: ConflictResolver | None = None,
) -> uuid.UUID:
    """Undo a mistaken identity merge by an EXPLICIT, recorded split (MAP-R12).

    Re-points ONE contributing candidate at a fresh canonical ``activity_id``, records
    the split decision on the candidate's ``identity_resolution`` (auditable, like the
    merge it reverses), and re-resolves BOTH canonical activities from their remaining
    candidates — non-destructive: every contributing source value is retained.
    Returns the new canonical activity id.
    """
    row = await session.get(SourceCandidate, source_candidate_id)
    if row is None or row.resolved_activity_id is None:
        raise LookupError(
            f"candidate {source_candidate_id} is unknown or not resolved to an activity"
        )
    old_activity_id = row.resolved_activity_id
    athlete = row.athlete_id  # captured BEFORE re-resolution expires ORM state
    new_activity_id = uuid7()
    row.resolved_activity_id = new_activity_id
    row.identity_resolution = {
        "rule": "explicit_split",
        "match_score": 0.0,
        "split_from_activity_id": str(old_activity_id),
        "previous_decision": row.identity_resolution,
    }
    await session.flush()
    await re_resolve_activity(session, athlete, new_activity_id, resolver=resolver)
    remaining = await _activity_candidates(session, athlete, old_activity_id)
    if remaining:
        await re_resolve_activity(session, athlete, old_activity_id, resolver=resolver)
    else:
        await _delete_activity_record(session, old_activity_id)
    return new_activity_id


__all__ = [
    "deactivate_source",
    "ingest_tombstone",
    "re_resolve_activity",
    "re_resolve_source_records",
    "re_resolve_wellness",
    "reactivate_source",
    "split_activity",
]
