"""The OSS default dedup / conflict resolver (CONF-R*, DEDUP-R7, MAP-R9..R12).

Two responsibilities, both pure and deterministic (CONF-R4):

1. :func:`resolve_field` — pick the canonical value for one field from contributing
   candidates, applying the CONF-R2 total order strictly. Field-granular (CONF-R3):
   power from one source, GPS from another for the same activity.
2. :func:`resolve_activity_identity` — decide whether two activity candidates are the
   same real-world session (MAP-R9..R12), using the policy-driven default matcher
   (MAP-R10): start-time overlap window + duration tolerance, or a strong shared
   fingerprint regardless of window.

This is the conservative OSS resolver (DEDUP-R7): it collapses only high-confidence
identity matches and keeps ambiguous candidates separate rather than fabricating a
merge. The advanced fuzzy resolver is commercial (DEDUP-R8) and rides the same seam.
The single-count invariant (DEDUP-R1/R4) holds regardless of which resolver is active.
No branch here hardcodes a source name (CONF-R1).
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass

from wattwise_core.domain.candidate import FieldCandidate
from wattwise_core.domain.enums import Fidelity, trust_rank

# Default identity-resolution thresholds (MAP-R10) — policy-driven, configurable.
DEFAULT_START_WINDOW_S = 120.0  # ±120 s start-time overlap
DEFAULT_DURATION_TOL_FRAC = 0.05  # ±5%
DEFAULT_DURATION_TOL_S = 60.0  # or ±60 s, whichever is looser


@dataclass(frozen=True, slots=True)
class ResolvedField:
    """The outcome of :func:`resolve_field` for one canonical field."""

    value: object
    winning_source_descriptor_id: str
    winning_trust_tier: Fidelity
    disputed: bool
    considered_source_ids: tuple[str, ...]


def _sort_key(c: FieldCandidate) -> tuple[int, float, float, float, str]:
    """The CONF-R2 total order as a sortable key (lower tuple sorts first = better).

    1. trust_tier (lower rank = higher trust); 2. confidence (higher better →
    negate); 3. recency (more recent observed_at/fetched_at → negate epoch); 4.
    completeness (higher better → negate); 5. stable tiebreak: lowest
    source_descriptor_id.
    """
    ts = c.observed_at or c.fetched_at
    epoch = ts.timestamp() if isinstance(ts, _dt.datetime) else 0.0
    return (
        trust_rank(c.trust_tier),
        -c.confidence,
        -epoch,
        -c.completeness,
        c.source_descriptor_id,
    )


def resolve_field(
    candidates: list[FieldCandidate], *, dispute_tolerance: float | None = None
) -> ResolvedField | None:
    """Resolve one canonical field from contributing candidates (CONF-R2/R3/R5).

    Returns ``None`` when there is no contributor (the caller then records a typed
    gap — never a zero-fill, CONF-R5). Absent/field-gap candidates must already have
    been filtered out by the caller (CONF-R2b): a non-contribution is never a value.

    ``disputed`` is set when at least two candidates materially disagree beyond
    ``dispute_tolerance`` (numeric fields only); the best value is still chosen and
    never averaged or hidden (CONF-R5).
    """
    if not candidates:
        return None
    ordered = sorted(candidates, key=_sort_key)
    winner = ordered[0]
    considered = tuple(c.source_descriptor_id for c in ordered)
    disputed = _is_disputed(ordered, dispute_tolerance)
    return ResolvedField(
        value=winner.value,
        winning_source_descriptor_id=winner.source_descriptor_id,
        winning_trust_tier=winner.trust_tier,
        disputed=disputed,
        considered_source_ids=considered,
    )


def _is_disputed(ordered: list[FieldCandidate], tolerance: float | None) -> bool:
    if tolerance is None or len(ordered) < 2:
        return False
    numeric = [c.value for c in ordered if isinstance(c.value, int | float)]
    if len(numeric) < 2:
        return False
    lo, hi = min(numeric), max(numeric)
    scale = max(1.0, abs(hi))
    return (hi - lo) / scale > tolerance


def resolve_activity_identity(
    a_start: _dt.datetime,
    a_duration_s: float,
    a_sport: str,
    a_fingerprint: str | None,
    b_start: _dt.datetime,
    b_duration_s: float,
    b_sport: str,
    b_fingerprint: str | None,
    *,
    start_window_s: float = DEFAULT_START_WINDOW_S,
    duration_tol_frac: float = DEFAULT_DURATION_TOL_FRAC,
    duration_tol_s: float = DEFAULT_DURATION_TOL_S,
) -> bool:
    """Decide whether two activity candidates are the same session (MAP-R10).

    COMPATIBLE SPORT is required FIRST: two incompatible-sport candidates are never the
    same session, even if they share a fingerprint (a shared fingerprint must never
    short-circuit the sport gate — two unrelated sessions that collide on a fingerprint
    token must stay separate). Then, with compatible sport: a shared strong fingerprint
    (device/file UUID, FIT fingerprint) matches regardless of the time window; otherwise
    start-time overlap within ``start_window_s`` AND duration similarity within the looser
    of ``duration_tol_frac`` or ``duration_tol_s``. Conservative (DEDUP-R7): anything
    short of these stays separate.
    """
    if a_sport != b_sport:
        return False
    if a_fingerprint is not None and b_fingerprint is not None and a_fingerprint == b_fingerprint:
        return True
    if abs((a_start - b_start).total_seconds()) > start_window_s:
        return False
    tol = max(duration_tol_s, duration_tol_frac * max(a_duration_s, b_duration_s))
    return abs(a_duration_s - b_duration_s) <= tol


__all__ = [
    "DEFAULT_DURATION_TOL_FRAC",
    "DEFAULT_DURATION_TOL_S",
    "DEFAULT_START_WINDOW_S",
    "ResolvedField",
    "resolve_activity_identity",
    "resolve_field",
]
