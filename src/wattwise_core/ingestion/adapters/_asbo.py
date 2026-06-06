"""Shared intermediate ASBO for the file-upload adapter (ADP-R8, FIL-R3).

The three decoders (FIT/GPX/TCX) each turn verbatim bytes into ONE
:class:`ActivityAsbo` — a typed, source-shaped object (records + session summary +
laps + RR + a native fingerprint). It is NOT canonical: it keeps source-ish numbers
in their already-SI form so the pure ``map`` (MAP-R1) is the single place that emits
canonical GBO fields. A malformed file raises the typed :class:`FileDecodeError` so
arbitrary bytes fail closed (TIER-R5 / CLI-R2), never a bare crash or a
wrong-but-plausible record.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field


class FileDecodeError(Exception):
    """A FIT/GPX/TCX file could not be parsed into an :class:`ActivityAsbo`.

    Raised by every decoder for malformed/corrupt/empty input (TIER-R5). The
    adapter surfaces this as a typed decode failure; it is NEVER a partial
    coercion into a canonical record (CLI-R2).
    """


@dataclass(frozen=True, slots=True)
class AsboRecord:
    """One per-sample observation, units already SI (no source vocab)."""

    timestamp: _dt.datetime | None
    power_w: float | None = None
    hr_bpm: float | None = None
    cadence_rpm: float | None = None
    speed_mps: float | None = None
    altitude_m: float | None = None
    distance_m: float | None = None
    latlng: tuple[float, float] | None = None
    temp_c: float | None = None


@dataclass(frozen=True, slots=True)
class AsboLap:
    """One lap summary, units already SI."""

    lap_index: int
    start_time: _dt.datetime | None = None
    duration_s: float | None = None
    distance_m: float | None = None
    avg_power_w: float | None = None
    max_power_w: float | None = None
    avg_hr_bpm: float | None = None
    max_hr_bpm: float | None = None
    avg_cadence_rpm: float | None = None


@dataclass(frozen=True, slots=True)
class ActivityAsbo:
    """A decoded activity file in source-shaped-but-SI form (one per file).

    ``session`` holds optional file-level summary fields (e.g. ``sport``,
    ``total_distance``, ``avg_power``, ``title``) keyed by the decoder's own names;
    ``map`` is the only place these are translated to canonical fields (MAP-R1/R2).
    ``native_fingerprint`` is the per-format stable identity used to derive
    ``source_native_id`` (LIN-R1.1); ``None`` falls back to the content hash.
    """

    records: tuple[AsboRecord, ...] = ()
    session: dict[str, object] = field(default_factory=dict)
    laps: tuple[AsboLap, ...] = ()
    rr_intervals_ms: tuple[float, ...] = ()
    native_fingerprint: str | None = None


__all__ = [
    "ActivityAsbo",
    "AsboLap",
    "AsboRecord",
    "FileDecodeError",
]
