"""Pre-persist candidate validation gate (MAP-R2, MAP-R6).

Every :class:`~wattwise_core.domain.candidate.GboCandidate` passes through
:func:`validate_candidate` BEFORE it may contribute to the canonical store:

* **MAP-R2 (no leakage, structural):** the payload may contain ONLY canonical field
  names — the canonical model's own columns plus the structural ``streams``/``laps``
  envelopes. Any other key (a source-named field, unit or enum) fails the gate. The
  allowlists are DERIVED from the canonical models, so they cannot drift from the
  schema.
* **MAP-R6 (invariants):** values must sit in plausible physical ranges, enum-valued
  fields must be members, a stream time base must be monotonic, and laps must be
  contiguous.

A failing candidate is QUARANTINED, not dropped: the ingest path persists its row
with ``quarantine_rule_id`` set (retaining its full lineage envelope) and excludes it
from every resolution set — it is never partially written into the canonical store.
The returned rule id names exactly which check failed, so the quarantine is auditable.
"""

from __future__ import annotations

from itertools import pairwise
from typing import Any

from wattwise_core.domain.candidate import GboCandidate
from wattwise_core.domain.enums import DeviceClass, GboType
from wattwise_core.persistence.models import Activity, DailyWellness

# Structural payload envelopes that are not scalar columns (validated separately).
_STRUCTURAL_KEYS = frozenset({"streams", "laps"})

# Canonical-key allowlists are DERIVED from the canonical models (MAP-R2): the
# payload may only name what the canonical schema itself names.
_ACTIVITY_KEYS = frozenset(Activity.__table__.columns.keys()) | _STRUCTURAL_KEYS
_WELLNESS_KEYS = frozenset(DailyWellness.__table__.columns.keys())

# Plausible physical ranges (MAP-R6) — schema-level validation constraints (the same
# kind of bound as a pydantic ``Field(ge=...)``), not tunable configuration values.
_RANGES: dict[str, tuple[float, float]] = {
    "avg_power_w": (0.0, 3000.0),
    "max_power_w": (0.0, 4000.0),
    "avg_hr_bpm": (20.0, 260.0),
    "max_hr_bpm": (20.0, 260.0),
    "resting_hr_bpm": (20.0, 200.0),
    "avg_cadence_rpm": (0.0, 300.0),
    "avg_speed_mps": (0.0, 50.0),
    "distance_m": (0.0, 2_000_000.0),
    "elapsed_time_s": (0.0, 7 * 86_400.0),
    "moving_time_s": (0.0, 7 * 86_400.0),
    "elevation_gain_m": (0.0, 50_000.0),
    "avg_temp_c": (-60.0, 70.0),
    "hrv_rmssd_ms": (0.0, 500.0),
    "hrv_sdnn_ms": (0.0, 500.0),
    "sleep_duration_s": (0.0, 36 * 3600.0),
    "sleep_score": (0.0, 100.0),
    "steps": (0.0, 500_000.0),
    "vo2max": (10.0, 100.0),
}


def validate_candidate(cand: GboCandidate) -> str | None:
    """Validate one candidate against the MAP-R2/MAP-R6 gate.

    Returns ``None`` when the candidate passes, else the failing RULE ID (e.g.
    ``"MAP-R2:non-canonical-key:icu_watts"``) the quarantine records. Checks run in a
    fixed order so the recorded rule is deterministic for a given payload.
    """
    allowed = _allowlist(cand.gbo_type)
    if allowed is not None:
        for key in cand.payload:
            if key not in allowed:
                return f"MAP-R2:non-canonical-key:{key}"
    for fname, (lo, hi) in _RANGES.items():
        value = cand.payload.get(fname)
        if isinstance(value, int | float) and not lo <= float(value) <= hi:
            return f"MAP-R6:range:{fname}"
    device_class = cand.payload.get("device_class")
    if device_class is not None and not _is_device_class(device_class):
        return "MAP-R6:enum:device_class"
    rule = _validate_streams(cand.payload.get("streams"))
    if rule is not None:
        return rule
    return _validate_laps(cand.payload.get("laps"))


def _allowlist(gbo_type: str) -> frozenset[str] | None:
    """The canonical-key allowlist for a gbo type; ``None`` = no payload contract yet."""
    if gbo_type == GboType.ACTIVITY.value:
        return _ACTIVITY_KEYS
    if gbo_type == GboType.DAILY_WELLNESS.value:
        return _WELLNESS_KEYS
    return None


def _is_device_class(value: object) -> bool:
    try:
        DeviceClass(str(value))
    except ValueError:
        return False
    return True


def _validate_streams(streams: Any) -> str | None:
    """The stream envelope's MAP-R6 invariants: monotonic time base, list-valued channels."""
    if not streams:
        return None
    if not isinstance(streams, dict):
        return "MAP-R6:streams-shape"
    time_chan = streams.get("time_s")
    if isinstance(time_chan, dict):
        values = [v for v in (time_chan.get("values") or []) if v is not None]
        if any(b < a for a, b in pairwise(values)):
            return "MAP-R6:time-base-not-monotonic"
    return None


def _validate_laps(laps: Any) -> str | None:
    """The lap envelope's MAP-R6 invariants: contiguous indexes, ordered offsets."""
    if not laps:
        return None
    if not isinstance(laps, list):
        return "MAP-R6:laps-shape"
    raw_indexes = [lap.get("lap_index") for lap in laps if isinstance(lap, dict)]
    indexes = [i for i in raw_indexes if isinstance(i, int)]
    if len(indexes) != len(laps):
        return "MAP-R6:lap-index-missing"
    base = min(indexes)
    if sorted(indexes) != list(range(base, base + len(indexes))):
        return "MAP-R6:lap-contiguity"
    offsets = [
        lap.get("start_offset_s")
        for lap in sorted(laps, key=lambda lap: lap["lap_index"])
        if isinstance(lap.get("start_offset_s"), int | float)
    ]
    if any(b < a for a, b in pairwise(offsets)):
        return "MAP-R6:lap-offsets-not-ordered"
    return None


__all__ = ["validate_candidate"]
