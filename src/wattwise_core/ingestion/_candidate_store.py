"""Atomic candidate persistence for the ingest path (UPS-R2, ING-UPS-R1, UPS-R3/R5).

The single seam through which a :class:`SourceCandidate` (tier-2 mapped observation) is
written: a SINGLE atomic insert-or-update on the candidate natural key via the sanctioned
``persistence/upsert.py`` seam — never a ``select`` then ``add`` check-then-write, so
concurrent sync runs cannot race (UPS-R2). Kept apart from the resolution/canonical-write
service to bound the writer module's size (QUAL-R9).

Candidate writes are **bulk/batched** (ING-UPS-R1 / PERF-R1): a whole batch's prepared
value-dicts land in ONE multi-row ``VALUES`` upsert round-trip through the seam, not a
per-row insert loop. Per-candidate supersession of a CHANGED restatement (the
``select``-for-prior + version-tag of the prior row) runs first, since it depends on
each candidate's own prior version; only the actual INSERT of the new rows is batched.
"""

from __future__ import annotations

import datetime as _dt
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

from sqlalchemy import Table, select
from sqlalchemy.ext.asyncio import AsyncSession

from wattwise_core.domain.candidate import GboCandidate
from wattwise_core.domain.enums import GboType
from wattwise_core.persistence.models import SourceCandidate
from wattwise_core.persistence.upsert import upsert_many

# Candidate natural key (UPS-R1): the candidate-key columns PLUS content_hash, the UNIQUE
# on source_candidate. The atomic upsert keys on it so a byte-identical re-ingest is an
# idempotent no-op refresh and a changed restatement inserts a NEW version, never a
# check-then-write (UPS-R2/R3); only fetch metadata is refreshed on the no-op collision.
_CANDIDATE_KEY = (
    "athlete_id", "source_descriptor_id", "source_native_id", "gbo_type", "content_hash",
)
_CANDIDATE_REFRESH = ("fetched_at", "ingest_run_id")


@dataclass(slots=True)
class PreparedCandidate:
    """A candidate's prepared value-dict + the prior identity to carry forward.

    Produced by :func:`_prepare_candidate` (which runs the per-candidate supersession
    BEFORE the batched INSERT) and consumed by :func:`persist_candidates_bulk` (which
    issues ONE multi-row ``VALUES`` upsert for a whole batch, ING-UPS-R1 / PERF-R1).
    """

    cand: GboCandidate
    values: dict[str, Any]
    prior_activity_id: uuid.UUID | None


async def _prepare_candidate(
    session: AsyncSession,
    athlete: uuid.UUID,
    descriptor: uuid.UUID,
    cand: GboCandidate,
    connection_id: str | uuid.UUID | None,
    run_id: uuid.UUID,
) -> PreparedCandidate:
    """Build the candidate value-dict and supersede-and-version a CHANGED prior version.

    The supersession (locate the prior non-superseded version, mark it
    ``is_superseded=True`` and version-tag its ``source_native_id``) MUST run per
    candidate before the batched INSERT, because it depends on each candidate's own
    prior version (PRV-R2). Only the actual write of the new rows is batched. The prior
    version's ``resolved_activity_id`` is returned so it can be carried forward after the
    new row exists (ING-R6).
    """
    prior = await _current_version(session, athlete, descriptor, cand)
    values = _candidate_values(athlete, descriptor, cand, connection_id, run_id)
    if prior is not None and prior.content_hash == cand.content_hash:
        # UPS-R3 no-op: the row is byte-identical; only fetch metadata will be refreshed.
        return PreparedCandidate(cand=cand, values=values, prior_activity_id=None)
    if prior is not None:
        # PRV-R2: preserve the prior version for audit. The candidate-key unique constraint
        # admits only ONE row per key, so the superseded row's source_native_id is
        # version-tagged (its observed identity is retained in the untouched payload/
        # content_hash) and the NEW row reclaims the candidate key.
        prior.is_superseded = True
        prior.source_native_id = _superseded_native_id(prior)
        await session.flush()
        return PreparedCandidate(
            cand=cand, values=values, prior_activity_id=prior.resolved_activity_id
        )
    return PreparedCandidate(cand=cand, values=values, prior_activity_id=None)


async def prepare_batch(
    session: AsyncSession,
    athlete: uuid.UUID,
    descriptor: uuid.UUID,
    batch: list[GboCandidate],
    connection_id: str | uuid.UUID | None,
    run_id: uuid.UUID,
    *,
    validate: Callable[[GboCandidate], None],
) -> tuple[list[PreparedCandidate], int]:
    """Validate + prepare each candidate inside its OWN SAVEPOINT (ING-UPS-R3 isolation).

    ``validate`` parses the resolution-critical payload fields so a malformed record is
    rejected BEFORE the batched insert — it never lands an orphan candidate row, and
    rolling back its savepoint undoes any supersession it began. A failure is dropped and
    counted; the rest of the batch is unaffected (no whole-batch rollback on one bad
    record). Returns ``(prepared, failed_count)``. ING-UPS-R3's range-precise gap
    (ING-GAP-R5) for the failed record is DEFERRED to the watermark/gap model (ING-UPS-R2).
    """
    prepared: list[PreparedCandidate] = []
    failed = 0
    for cand in batch:
        try:
            async with session.begin_nested():
                validate(cand)
                prepared.append(
                    await _prepare_candidate(
                        session, athlete, descriptor, cand, connection_id, run_id
                    )
                )
        except Exception:
            failed += 1  # ING-UPS-R3 record isolation; keep the run
    return prepared, failed


async def persist_candidates_bulk(
    session: AsyncSession,
    athlete: uuid.UUID,
    descriptor: uuid.UUID,
    prepared: list[PreparedCandidate],
) -> dict[str, SourceCandidate]:
    """Upsert a whole batch's candidates in ONE multi-row round-trip (ING-UPS-R1/PERF-R1).

    Every prepared candidate value-dict is landed with a SINGLE multi-row ``VALUES``
    upsert through the sanctioned seam — never one INSERT per candidate — keyed on the
    candidate natural key ``(athlete_id, source_descriptor_id, source_native_id,
    gbo_type, content_hash)`` so a byte-identical re-ingest collides and refreshes ONLY
    fetch metadata (``fetched_at``/``ingest_run_id``, UPS-R3) and a changed restatement
    inserts a NEW version. Persisted rows are then reloaded by their full natural key and
    any carried-forward ``resolved_activity_id`` is applied (ING-R6). Returns the rows
    keyed by ``source_native_id`` so the caller can resolve each candidate's canonical
    write against the row it just persisted.
    """
    if not prepared:
        return {}
    table = cast("Table", SourceCandidate.__table__)
    # A multi-row ON CONFLICT upsert cannot touch the same conflict key twice in one
    # statement; collapse any same-natural-key duplicates within the batch to the last
    # occurrence (sequential last-write-wins) so the single round-trip stays valid.
    by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    for prep in prepared:
        by_key[tuple(prep.values[k] for k in _CANDIDATE_KEY)] = prep.values
    await upsert_many(
        session,
        table,
        list(by_key.values()),
        conflict_keys=_CANDIDATE_KEY,
        update_columns=_CANDIDATE_REFRESH,
    )
    rows: dict[str, SourceCandidate] = {}
    for prep in prepared:
        row = await _reload_candidate(session, athlete, descriptor, prep.cand)
        if prep.prior_activity_id is not None and row.resolved_activity_id is None:
            # Carry the prior version's resolved identity forward (ING-R6).
            row.resolved_activity_id = prep.prior_activity_id
        rows[prep.cand.source_native_id] = row
    if any(p.prior_activity_id is not None for p in prepared):
        await session.flush()
    return rows


async def persist_candidate(
    session: AsyncSession,
    athlete: uuid.UUID,
    descriptor: uuid.UUID,
    cand: GboCandidate,
    connection_id: str | uuid.UUID | None,
    run_id: uuid.UUID,
) -> SourceCandidate:
    """Atomically upsert ONE source candidate; supersede-and-version a CHANGED re-ingest.

    A single-candidate convenience over the bulk path: the write is a SINGLE atomic
    insert-or-update on the candidate natural key through the sanctioned
    ``persistence/upsert.py`` seam — never a check-then-write, so concurrent sync runs
    cannot race (UPS-R2). A byte-identical re-ingest collides and refreshes ONLY fetch
    metadata (UPS-R3); a CHANGED re-ingest inserts a NEW version, supersedes the prior,
    and carries its ``resolved_activity_id`` forward (PRV-R2/ING-R6).
    """
    prepared = await _prepare_candidate(
        session, athlete, descriptor, cand, connection_id, run_id
    )
    rows = await persist_candidates_bulk(session, athlete, descriptor, [prepared])
    return rows[cand.source_native_id]


async def _current_version(
    session: AsyncSession, athlete: uuid.UUID, descriptor: uuid.UUID, cand: GboCandidate
) -> SourceCandidate | None:
    """The current non-superseded candidate for this candidate key (NOT keyed on hash).

    This read locates a prior version to supersede / a no-op match; it is NOT the write,
    so it introduces no check-then-write of the row being persisted (the write itself is
    the atomic upsert keyed on the FULL natural key including ``content_hash``, UPS-R2).
    """
    stmt = select(SourceCandidate).where(
        SourceCandidate.athlete_id == athlete,
        SourceCandidate.source_descriptor_id == descriptor,
        SourceCandidate.source_native_id == cand.source_native_id,
        SourceCandidate.gbo_type == GboType(cand.gbo_type),
        SourceCandidate.is_superseded.is_(False),
    )
    return (await session.execute(stmt)).scalar_one_or_none()


async def _reload_candidate(
    session: AsyncSession, athlete: uuid.UUID, descriptor: uuid.UUID, cand: GboCandidate
) -> SourceCandidate:
    """Read back the row the atomic upsert just persisted, by its full natural key.

    A read AFTER the atomic write (not a TOCTOU window): the row is guaranteed present.
    """
    stmt = select(SourceCandidate).where(
        SourceCandidate.athlete_id == athlete,
        SourceCandidate.source_descriptor_id == descriptor,
        SourceCandidate.source_native_id == cand.source_native_id,
        SourceCandidate.gbo_type == GboType(cand.gbo_type),
        SourceCandidate.content_hash == cand.content_hash,
    )
    return (await session.execute(stmt)).scalars().first()  # type: ignore[return-value]


def _superseded_native_id(prior: SourceCandidate) -> str:
    """A version-tagged native id freeing the candidate key for the new version (PRV-R2).

    The prior row stays fully readable for audit (its payload/content_hash are untouched);
    only its candidate-key slot is vacated so the new version can hold the key under the
    single-row unique constraint.
    """
    tag = f"#superseded:{prior.content_hash[:16]}"
    base = prior.source_native_id.split("#superseded:", 1)[0]
    return f"{base}{tag}"


def _candidate_values(
    athlete: uuid.UUID,
    descriptor: uuid.UUID,
    cand: GboCandidate,
    connection_id: str | uuid.UUID | None,
    run_id: uuid.UUID,
) -> dict[str, Any]:
    """The candidate row value-dict for the atomic upsert seam (UPS-R2).

    Only the candidate's own columns; the surrogate PK and ``created_at``/``updated_at`` are
    filled by their Core column defaults, so the insert path matches the prior ORM
    construction byte-for-byte while the WRITE itself is the atomic insert-or-update.
    """
    return {
        "athlete_id": athlete,
        "source_descriptor_id": descriptor,
        "connection_id": _uid(connection_id) if connection_id else None,
        "source_native_id": cand.source_native_id,
        "gbo_type": GboType(cand.gbo_type),
        "observed_at": cand.observed_at,
        "fetched_at": cand.fetched_at,
        "content_hash": cand.content_hash,
        "adapter_version": cand.adapter_version,
        "mapping_version": cand.mapping_version,
        "trust_profile": {"tier": cand.trust_tier.value},
        "payload": _jsonsafe(cand.payload),
        "confidence": cand.confidence,
        "ingest_run_id": run_id,
        "untrusted_content": cand.untrusted_content,
    }


def _jsonsafe(value: Any) -> Any:
    """Coerce a mapped payload to JSON-storable form (datetimes/dates -> ISO strings)."""
    if isinstance(value, _dt.datetime | _dt.date):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: _jsonsafe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonsafe(v) for v in value]
    return value


def _uid(value: str | uuid.UUID) -> uuid.UUID:
    return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))


__all__ = [
    "PreparedCandidate",
    "persist_candidate",
    "persist_candidates_bulk",
    "prepare_batch",
]
