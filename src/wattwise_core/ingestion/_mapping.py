"""Pure candidate->canonical scalar mapping, trust-tier seams, and value parsing.

A cohesive set of module-level PURE helpers used by the ingest write path
(``ingest.py``): per-field scalar resolution across candidates (CONF-R2/R5), the
effective-tier seams that bind a :class:`~wattwise_core.ingestion.trust.TrustPolicy`
to a channel for the ``_canonical`` helpers (PRV-R7), config-independent trust
selection (``_tier_of`` / ``_highest_trust``), the activity row value-dict + atomic
upsert update-set (UPS-R2), payload validation, and the stored-value parsers.

These are extracted into a focused sibling module (QUAL-R9 "focused modules"); none
of them touch the DB or import ``ingest``, so there is no circular import.
"""

from __future__ import annotations

import datetime as _dt
import uuid
from collections.abc import Callable
from typing import Any

from wattwise_core.domain.candidate import GboCandidate
from wattwise_core.domain.enums import Fidelity, GboType, trust_rank
from wattwise_core.ingestion import _canonical as _cw
from wattwise_core.ingestion.trust import TrustPolicy
from wattwise_core.persistence.models import Activity, SourceCandidate
from wattwise_core.persistence.models.athlete_preference import WHOLE_SOURCE_CHANNEL
from wattwise_core.persistence.types import utcnow
from wattwise_core.seams import ConflictResolver

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

_ACTIVITY_COLUMNS = frozenset(Activity.__table__.columns.keys())


def _validate_payload(cand: GboCandidate) -> None:
    """Parse the resolution-critical payload fields, raising on a malformed candidate."""
    if cand.gbo_type == GboType.ACTIVITY.value:
        _parse_start_time(cand.payload["start_time"])
    elif cand.gbo_type == GboType.DAILY_WELLNESS.value:
        _parse_date(cand.payload["local_date"])


def _resolve_scalars(
    candidates: list[SourceCandidate],
    fields: tuple[str, ...],
    policy: TrustPolicy,
    resolver: ConflictResolver,
) -> tuple[dict[str, Any], dict[str, object]]:
    """Resolve each scalar field across candidates + build its coverage (CONF-R2/R5).

    Returns ``(resolved_values, coverage)``. Each field is resolved with its EFFECTIVE
    per-channel trust tier (``policy.tier(candidate, fname)`` — the configurable PRV-R7
    re-rank, defaulting to the adapter tier when unconfigured). A field whose >=2
    contributors materially disagree beyond the per-field dispute tolerance gets
    ``coverage.disputed=True`` — the best value is still selected, the disagreement is
    surfaced not hidden (CONF-R5). Field resolution runs through the INJECTED resolver
    seam (CONF-R7/DEDUP-R6), not a directly-imported function — so the advanced
    commercial resolver (DEDUP-R8) rides the same seam without editing this consumer.
    """
    resolved: dict[str, Any] = {}
    coverage: dict[str, object] = {}
    for fname in fields:
        tier_of = _channel_tier_of(policy, fname)  # effective per-channel tier (PRV-R7)
        contributors = _cw.field_candidates(candidates, fname, tier_of)
        winner = resolver.resolve_field(
            contributors, dispute_tolerance=_cw.dispute_tolerance(fname)
        )
        if winner is None:
            # No contributor supplied this field: a typed absence, NOT a silent skip
            # (GAP-R1/GAP-R3). absent_true (no source provides it) — never zero-filled.
            coverage[fname] = _cw.coverage_for(
                False, Fidelity.ABSENT_TRUE, disputed=False
            ).to_jsonable()
            continue
        resolved[fname] = winner.value
        # Badge the RESOLVED WINNER's tier, NOT an arbitrary scanned contributor (PRV-R6).
        coverage[fname] = _cw.coverage_for(
            True, winner.winning_trust_tier, disputed=winner.disputed
        ).to_jsonable()
    return resolved, coverage


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


__all__ = [
    "_ACTIVITY_COLUMNS",
    "_ACTIVITY_SCALARS",
    "_LAP_SCALARS",
    "_activity_values",
    "_channel_tier_of",
    "_highest_trust",
    "_parse_date",
    "_parse_start_time",
    "_resolve_scalars",
    "_tier_of",
    "_validate_payload",
    "_whole_source_tier_of",
]
