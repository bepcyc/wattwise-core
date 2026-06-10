"""Canonical stream-set/channel resolution + writes (CONF-R3, UPS-R2, ING-UPS-R5).

The focused sibling of :mod:`wattwise_core.ingestion._canonical` (QUAL-R9 size split) that owns
the per-channel stream path of the canonical write: resolving each stream channel across ALL
contributing candidates under its OWN effective per-channel trust tier (CONF-R3 / PRV-R6/R7),
and atomically upserting the activity stream set + each channel through the sanctioned upsert
seam with the never-regress trust guard (UPS-R2 / ING-UPS-R5). Stateless like its sibling: each
function takes the session explicitly; behavior is unchanged from the pre-split module.
"""

from __future__ import annotations

import uuid
from typing import Any, cast

from sqlalchemy import Table, select
from sqlalchemy.ext.asyncio import AsyncSession

from wattwise_core.domain.enums import (
    Fidelity,
    SampleBasis,
    StreamChannelName,
    StreamSetKind,
    trust_rank,
)
from wattwise_core.ingestion._canonical import coverage_for
from wattwise_core.ingestion.trust import TrustPolicy
from wattwise_core.persistence.models import (
    ActivityStreamSet,
    SourceCandidate,
    StreamChannel,
)
from wattwise_core.persistence.types import utcnow, uuid7
from wattwise_core.persistence.upsert import upsert


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
    coverage = (
        chan.get("_coverage")
        or coverage_for(True, _coerce_fidelity(chan.get("_fidelity")), disputed=False).to_jsonable()
    )
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


__all__ = ["resolve_streams", "upsert_stream_set"]
