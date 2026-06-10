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

The per-channel stream resolution + stream-set/channel writes live in the focused
:mod:`wattwise_core.ingestion._canonical_streams` sibling (QUAL-R9 size split).
"""

from __future__ import annotations

import datetime as _dt
import uuid
from dataclasses import dataclass
from typing import Any, cast

from sqlalchemy import Table, select
from sqlalchemy.ext.asyncio import AsyncSession

from wattwise_core.domain.candidate import FieldCandidate
from wattwise_core.domain.coverage import Coverage, Substitution
from wattwise_core.domain.enums import (
    ActivityFileFormat,
    Fidelity,
)
from wattwise_core.domain.equivalence import substitution_for
from wattwise_core.ingestion.dedup import ResolvedField, resolve_field
from wattwise_core.persistence.models import (
    ActivityFile,
    ActivityLap,
    DailyWellness,
    SourceCandidate,
)
from wattwise_core.persistence.types import uuid7
from wattwise_core.persistence.upsert import upsert, upsert_many
from wattwise_core.storage import ObjectStore, content_hash

# Daily-wellness canonical scalar fields resolved across candidates (a subset of the
# DailyWellness columns the OSS adapters actually map; absent fields stay NULL/typed-gap).
WELLNESS_SCALARS = (
    "resting_hr_bpm",
    "hrv_rmssd_ms",
    "hrv_sdnn_ms",
    "sleep_score",
    "sleep_duration_s",
    "steps",
    "vo2max",
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
    present: bool,
    fidelity: Fidelity,
    *,
    disputed: bool,
    failed: bool = False,
    substitution: Substitution | None = None,
) -> Coverage:
    """A real :class:`Coverage` for a resolved canonical field (CONF-R5/GAP-R2/GAP-R3).

    Carries no source identity; ``disputed`` surfaces material multi-source
    disagreement without hiding it (the winner is still the resolved value).

    On a typed absence (``present=False``), ``failed`` selects the GAP-R3 distinction:
    ``absent_failed`` (a source that SHOULD have supplied the channel failed to fetch — an
    open gap) versus ``absent_true`` (no source supplies it at all). The caller passes
    ``failed=True`` only when it holds a real fetch-failure signal; the default
    ``absent_true`` is the honest state for a field no contributor provided.

    A non-``None`` ``substitution`` (DM-SUB-R4: the winner sits below its declared
    equivalence-class top tier) forces ``fidelity=substituted`` and attaches the
    ``{class, from_fidelity}`` marker so a client badges reduced precision.
    """
    if not present:
        return Coverage.absent(failed=failed)
    if substitution is not None:
        return Coverage(
            present=True,
            fidelity=Fidelity.SUBSTITUTED,
            disputed=disputed,
            substitution=substitution,
        )
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
                candidate_id=str(c.source_candidate_id),  # LIN-R3 pointer
            )
        )
    return out


def resolution_record(winner: ResolvedField) -> dict[str, object]:
    """The LIN-R3 per-field resolution record for one resolved winner.

    Pointers into ``source_candidate`` (winner + every considered candidate) plus the
    CONF-R2 rule that decided — persisted beside the canonical value so any canonical
    scalar is traceable to its origin and re-decidable WITHOUT copying the
    source-shaped envelope onto the canonical record. Lineage-only (LIN-R4).
    """
    return {
        "winner_candidate_id": winner.winning_candidate_id,
        "considered_candidate_ids": list(winner.considered_candidate_ids),
        "rule": winner.deciding_rule,
    }


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
    policy_version: str | None = None,
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
    field_resolution: dict[str, object] = {}
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
        # Badge the RESOLVED WINNER's tier, NOT an arbitrary scanned contributor (PRV-R6);
        # a winner below its declared equivalence-class top tier badges SUBSTITUTED with
        # the displaced tier recorded (DM-SUB-R4), never the winner's own tier as-if-top.
        coverage[fname] = coverage_for(
            True,
            winner.winning_trust_tier,
            disputed=winner.disputed,
            substitution=substitution_for(fname, winner.winning_trust_tier),
        ).to_jsonable()
        field_resolution[fname] = resolution_record(winner)  # LIN-R3
    values["coverage"] = coverage
    if coverage:
        update_columns.append("coverage")
    values["field_resolution"] = field_resolution
    values["policy_version"] = policy_version  # CONF-R6: the policy that produced this
    update_columns += ["field_resolution", "policy_version"]
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
    "resolution_record",
    "upsert_laps",
    "write_wellness_canonical",
]
