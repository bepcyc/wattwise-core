"""Canonical-write helpers for the ingest path (CONF-R*, ING-UPS-R*, FIL-R*).

Stateless write helpers the :class:`wattwise_core.ingestion.ingest.IngestService`
delegates to, kept apart to bound the writer module's size (QUAL-R9). Each takes the
session explicitly and writes the resolved canonical record:

* field/coverage resolution (CONF-R2/R5) — resolve every field across ALL contributing
  candidates and surface material disagreement as ``coverage.disputed`` (never hidden);
* daily-wellness resolution (CONF-R2 / ING-UPS-R5) — resolve across candidates by the
  CONF-R2 total order, NOT last-writer-wins;
* tier-1 verbatim-file capture (ING-R8 / FIL-R1/R5) — store the original bytes in the
  object store and create the dedup-idempotent ``activity_file`` reference.
"""

from __future__ import annotations

import datetime as _dt
import uuid
from dataclasses import dataclass
from typing import Any, cast

from sqlalchemy import Table, select
from sqlalchemy.ext.asyncio import AsyncSession

from wattwise_core.domain.candidate import FieldCandidate
from wattwise_core.domain.coverage import Coverage
from wattwise_core.domain.enums import (
    ActivityFileFormat,
    Fidelity,
    SampleBasis,
    StreamChannelName,
    StreamSetKind,
    trust_rank,
)
from wattwise_core.ingestion.dedup import resolve_field
from wattwise_core.ingestion.trust import TrustPolicy
from wattwise_core.persistence.models import (
    ActivityFile,
    ActivityLap,
    ActivityStreamSet,
    DailyWellness,
    SourceCandidate,
    StreamChannel,
)
from wattwise_core.persistence.types import utcnow, uuid7
from wattwise_core.persistence.upsert import upsert, upsert_many
from wattwise_core.storage import ObjectStore, content_hash

# Daily-wellness canonical scalar fields resolved across candidates (a subset of the
# DailyWellness columns the OSS adapters actually map; absent fields stay NULL/typed-gap).
WELLNESS_SCALARS = (
    "resting_hr_bpm", "hrv_rmssd_ms", "hrv_sdnn_ms", "sleep_score",
    "sleep_duration_s", "steps", "vo2max",
)

# Per-field numeric dispute tolerance (fraction of the larger magnitude). Two sources
# disagreeing beyond this on a numeric field set ``coverage.disputed`` (CONF-R5); the
# best value is still selected (never averaged or hidden).
_DEFAULT_DISPUTE_TOLERANCE = 0.05


@dataclass(frozen=True, slots=True)
class OriginalFile:
    """A verbatim original recording artifact to capture in tier-1 storage (FIL-R1).

    Carried INTO the ingest write path alongside the mapped candidates so the bytes
    are stored verbatim and an ``activity_file`` reference is created against the
    resolved canonical activity. ``source_native_id`` ties it to the candidate it was
    decoded from (a direct-API source with no original file supplies none).
    """

    data: bytes
    file_format: ActivityFileFormat
    source_native_id: str


def dispute_tolerance(field_name: str) -> float | None:
    """The numeric dispute tolerance for a field (CONF-R5); ``None`` for non-numeric."""
    if field_name in ("start_time", "sport", "sub_sport", "device_class"):
        return None
    return _DEFAULT_DISPUTE_TOLERANCE


def coverage_for(
    present: bool, fidelity: Fidelity, *, disputed: bool, failed: bool = False
) -> Coverage:
    """A real :class:`Coverage` for a resolved canonical field (CONF-R5/GAP-R2/GAP-R3).

    Carries no source identity; ``disputed`` surfaces material multi-source
    disagreement without hiding it (the winner is still the resolved value).

    On a typed absence (``present=False``), ``failed`` selects the GAP-R3 distinction:
    ``absent_failed`` (a source that SHOULD have supplied the channel failed to fetch — an
    open gap) versus ``absent_true`` (no source supplies it at all). The caller passes
    ``failed=True`` only when it holds a real fetch-failure signal; the default
    ``absent_true`` is the honest state for a field no contributor provided.
    """
    if not present:
        return Coverage.absent(failed=failed)
    return Coverage(present=True, fidelity=fidelity, disputed=disputed)


def field_candidates(
    candidates: list[SourceCandidate], fname: str, tier_of: Any
) -> list[FieldCandidate]:
    """Build the contributing :class:`FieldCandidate` list for one canonical field.

    ``completeness`` is higher for a stream-backed contribution than a summary-only
    scalar so the CONF-R2 step-4 completeness tiebreaker is actually applied.
    """
    out: list[FieldCandidate] = []
    for c in candidates:
        if c.payload.get(fname) is None:
            continue
        streams = c.payload.get("streams") or {}
        completeness = 2.0 if streams else 1.0
        out.append(
            FieldCandidate(
                value=c.payload[fname],
                trust_tier=tier_of(c),
                source_descriptor_id=str(c.source_descriptor_id),
                confidence=float(c.confidence) if c.confidence is not None else 1.0,
                observed_at=c.observed_at,
                fetched_at=c.fetched_at,
                completeness=completeness,
            )
        )
    return out


def resolve_streams(
    candidates: list[SourceCandidate], policy: TrustPolicy
) -> dict[str, dict[str, Any]]:
    """Resolve each stream channel across candidates by per-channel trust (CONF-R3/PRV-R6).

    Per-channel (not per-record) AND per-channel TRUST: each channel is resolved under its
    OWN effective tier ``policy.tier(candidate, channel)`` (mirroring the scalar path,
    PRV-R7/SF-3), so a descriptor ``trust_profile {power_w: SUMMARY_ONLY}`` or a per-athlete
    ``power_w`` override changes which source wins THAT stream channel — not just the
    whole-source ``"*"`` tier. A channel a higher-trust source lacks is filled from a
    lower-trust one rather than dropped, and a higher-trust channel is never regressed.
    Each winning channel carries its effective ``_fidelity`` plus a safe ``_coverage`` built
    through :class:`Coverage` (present/fidelity invariant enforced uniformly, D5). Ties
    break on the stable source_descriptor_id (deterministic, CONF-R4).
    """
    out: dict[str, dict[str, Any]] = {}
    # Discover every channel any candidate carries, then resolve each independently under
    # its own effective per-channel tier (the whole-source "*" tier no longer decides).
    names = {n for c in candidates for n in _candidate_streams(c)}
    for name in names:
        ordered = sorted(
            candidates,
            key=lambda c: (trust_rank(policy.tier(c, name)), str(c.source_descriptor_id)),
        )
        for c in ordered:  # highest-trust-for-this-channel first; first writer wins
            chan = _candidate_streams(c).get(name)
            if chan is None:
                continue
            fidelity = policy.tier(c, name)
            out[name] = {
                **chan,
                "_fidelity": fidelity.value,
                "_coverage": coverage_for(True, fidelity, disputed=False).to_jsonable(),
            }
            break
    return out


def _candidate_streams(candidate: SourceCandidate) -> dict[str, Any]:
    """The candidate's per-channel ``streams`` payload mapping (``{}`` when absent)."""
    return cast("dict[str, Any]", candidate.payload.get("streams") or {})


async def upsert_stream_set(
    session: AsyncSession, activity_id: uuid.UUID, streams: dict[str, Any]
) -> None:
    """Atomically upsert the activity stream set + each channel (UPS-R2, trust-guarded).

    The stream set is a single atomic insert-or-update keyed on its natural key
    ``activity_id`` through the sanctioned seam — never a ``select`` then ``add``
    check-then-write — so two sync runs landing the same activity's streams cannot race
    (UPS-R2). The set scalars (``sample_*``/``t0``) are NOT refreshed when the natural key
    already exists, so an existing set keeps its identity; only the per-channel values
    follow the trust guard.
    """
    first = next(iter(streams.values()))
    await upsert(
        session,
        cast("Table", ActivityStreamSet.__table__),
        {
            "stream_set_id": uuid7(),
            "activity_id": activity_id,
            "sample_basis": SampleBasis(first.get("sample_basis", "time")),
            "sample_rate_hz": first.get("sample_rate_hz", 1.0),
            "sample_count": len(first.get("values", [])),
            "t0": utcnow(),
        },
        conflict_keys=["activity_id"],
        update_columns=[],  # insert-or-keep: never regress an existing set's identity
    )
    stream_set_id = (
        await session.execute(
            select(ActivityStreamSet.stream_set_id).where(
                ActivityStreamSet.activity_id == activity_id
            )
        )
    ).scalar_one()
    for name, chan in streams.items():
        await _upsert_channel(session, stream_set_id, name, chan)


def _coerce_fidelity(raw: object) -> Fidelity:
    """Coerce a stored ``_fidelity`` token to ``Fidelity`` (worst tier on absence/garbage)."""
    if not isinstance(raw, str):
        return Fidelity.SUMMARY_ONLY
    try:
        return Fidelity(raw)
    except ValueError:
        return Fidelity.SUMMARY_ONLY


def _channel_rank(coverage: dict[str, object] | None) -> int:
    """The trust rank persisted on a channel's coverage (worst if absent)."""
    fid = (coverage or {}).get("fidelity")
    if not isinstance(fid, str):
        return trust_rank(Fidelity.SUMMARY_ONLY) + 1
    try:
        return trust_rank(Fidelity(fid))
    except ValueError:
        return trust_rank(Fidelity.SUMMARY_ONLY) + 1


async def _upsert_channel(
    session: AsyncSession, stream_set_id: uuid.UUID, name: str, chan: dict[str, Any]
) -> None:
    """Atomically upsert one stream channel on ``(stream_set_id, channel)`` (UPS-R2/ING-UPS-R5).

    The write is a single atomic insert-or-update through the sanctioned seam (no
    ``select`` then ``add`` race, UPS-R2). The existing channel's coverage is read ONLY to
    decide whether the incoming value may win: a lower-trust value never regresses a
    higher-trust stored channel (ING-UPS-R5), so when the incoming rank loses the upsert
    refreshes nothing (insert-or-keep), otherwise it refreshes values + coverage.
    """
    existing_rank = (
        await session.execute(
            select(StreamChannel.coverage).where(
                StreamChannel.stream_set_id == stream_set_id,
                StreamChannel.channel == StreamChannelName(name),
            )
        )
    ).scalar_one_or_none()
    values = chan.get("values", [])
    # D5: route channel coverage through Coverage(...).to_jsonable() so the
    # present/fidelity invariant is enforced uniformly (never a raw {present, fidelity}
    # dict that could persist a self-contradictory present=True + absent_* fidelity).
    coverage = chan.get("_coverage") or coverage_for(
        True, _coerce_fidelity(chan.get("_fidelity")), disputed=False
    ).to_jsonable()
    # ING-UPS-R5: when an existing channel outranks the incoming one, do NOT regress it —
    # insert-or-keep (refresh no columns on a key collision); else refresh values + coverage.
    wins = existing_rank is None or _channel_rank(coverage) <= _channel_rank(existing_rank)
    await upsert(
        session,
        cast("Table", StreamChannel.__table__),
        {
            "stream_channel_id": uuid7(),
            "stream_set_id": stream_set_id,
            "set_kind": StreamSetKind.ACTIVITY,
            "channel": StreamChannelName(name),
            "sample_basis": SampleBasis(chan.get("sample_basis", "time")),
            "values": values,
            "coverage": coverage,
        },
        conflict_keys=["stream_set_id", "channel"],
        update_columns=["values", "coverage"] if wins else [],
    )


async def upsert_laps(
    session: AsyncSession,
    activity_id: uuid.UUID,
    laps: list[dict[str, Any]],
    scalars: tuple[str, ...],
) -> None:
    """Batched atomic upsert of every lap on ``(activity_id, lap_index)`` (UPS-R2/PERF-R1).

    All laps for the activity are upserted in a SINGLE multi-row round-trip through the
    sanctioned seam — never a per-lap ``select`` then ``add`` loop (PERF-R1) and never a
    check-then-write race (UPS-R2). Re-ingest is idempotent on the natural key (GBO-R17).
    """
    if not laps:
        return
    rows = [
        {
            "activity_lap_id": uuid7(),
            "activity_id": activity_id,
            "lap_index": int(lap["lap_index"]),
            **{k: lap.get(k) for k in scalars},
        }
        for lap in laps
    ]
    await upsert_many(
        session,
        cast("Table", ActivityLap.__table__),
        rows,
        conflict_keys=["activity_id", "lap_index"],
    )


async def write_wellness_canonical(
    session: AsyncSession,
    athlete: uuid.UUID,
    local_date: _dt.date,
    candidates: list[SourceCandidate],
    tier_of: Any,
) -> None:
    """Resolve daily-wellness across ALL candidates and write the row (CONF-R2/ING-UPS-R5).

    Every field is resolved by the CONF-R2 total order (trust > confidence > recency >
    completeness > stable tiebreak) over the contributing candidates for this
    ``(athlete_id, local_date)`` — NOT last-writer-wins. A lower-trust newer candidate
    can therefore never clobber a higher-trust value (PRV-R6). The resolved row is then
    persisted through the sanctioned atomic upsert seam keyed on the natural key
    ``(athlete_id, local_date)`` — never a ``select`` then ``add`` race (UPS-R2). Only
    resolved fields are refreshed when the key already exists, so an unresolved field keeps
    its prior canonical value (no zero-filling, PRV-R6).
    """
    values: dict[str, Any] = {
        "daily_wellness_id": uuid7(),
        "athlete_id": athlete,
        "local_date": local_date,
    }
    update_columns: list[str] = []
    coverage: dict[str, object] = {}
    for fname in WELLNESS_SCALARS:
        contributors = field_candidates(candidates, fname, tier_of)
        winner = resolve_field(contributors, dispute_tolerance=dispute_tolerance(fname))
        if winner is None:
            # No contributor: a typed absent_true descriptor, not a silent skip
            # (GAP-R1/GAP-R3) — the wellness field is honestly "no data", never zero-filled.
            absent = coverage_for(False, Fidelity.ABSENT_TRUE, disputed=False)
            coverage[fname] = absent.to_jsonable()
            continue
        values[fname] = winner.value
        update_columns.append(fname)
        # Badge the RESOLVED WINNER's tier, NOT an arbitrary scanned contributor (PRV-R6).
        coverage[fname] = coverage_for(
            True, winner.winning_trust_tier, disputed=winner.disputed
        ).to_jsonable()
    values["coverage"] = coverage
    if coverage:
        update_columns.append("coverage")
    await upsert(
        session,
        cast("Table", DailyWellness.__table__),
        values,
        conflict_keys=["athlete_id", "local_date"],
        update_columns=update_columns,
    )
    await session.flush()


async def create_activity_file(
    session: AsyncSession,
    store: ObjectStore,
    *,
    athlete: uuid.UUID,
    activity_id: uuid.UUID,
    source_descriptor_id: uuid.UUID,
    original: OriginalFile,
    fetched_at: _dt.datetime | None,
) -> None:
    """Capture the verbatim original file in tier-1 storage + its reference (FIL-R1/R5).

    Stores the bytes byte-for-byte in the object store and inserts an ``activity_file``
    row linking the opaque ``object_ref`` to the resolved canonical activity. The row write
    is an atomic insert-or-ignore through the sanctioned seam keyed on the natural key
    ``(activity_id, source_descriptor_id, content_hash)`` (UPS-R2) — idempotent (FIL-R5):
    a re-ingest of the same artifact is a no-op (content-addressed store + the dedup
    uniqueness), and two concurrent runs cannot insert a duplicate row.
    """
    digest = content_hash(original.data)
    stmt = select(ActivityFile.activity_file_id).where(
        ActivityFile.activity_id == activity_id,
        ActivityFile.source_descriptor_id == source_descriptor_id,
        ActivityFile.content_hash == digest,
    )
    if (await session.execute(stmt)).scalar_one_or_none() is not None:
        return  # already captured (FIL-R5 dedup) — skip the redundant object-store write
    object_ref = store.put(original.data, suffix=f".{original.file_format.value}")
    await upsert(
        session,
        cast("Table", ActivityFile.__table__),
        {
            "activity_file_id": uuid7(),
            "activity_id": activity_id,
            "athlete_id": athlete,
            "object_ref": object_ref,
            "format": original.file_format,
            "byte_size": len(original.data),
            "content_hash": digest,
            "source_descriptor_id": source_descriptor_id,
            "fetched_at": fetched_at,
        },
        conflict_keys=["activity_id", "source_descriptor_id", "content_hash"],
        update_columns=[],  # insert-or-ignore: the artifact is immutable (FIL-R5)
    )
    await session.flush()


__all__ = [
    "WELLNESS_SCALARS",
    "OriginalFile",
    "coverage_for",
    "create_activity_file",
    "dispute_tolerance",
    "field_candidates",
    "resolve_streams",
    "upsert_laps",
    "upsert_stream_set",
    "write_wellness_canonical",
]
