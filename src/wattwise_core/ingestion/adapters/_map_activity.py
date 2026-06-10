"""Pure ASBO -> canonical-activity payload mapping (MAP-R1..R8, FIL-R3).

The single place a decoded :class:`ActivityAsbo` is translated into the canonical
``activity`` payload: canonical field names only (MAP-R2), SI units (MAP-R3),
canonical sport codes / stream channels / device class (MAP-R4), real gaps as
``None`` never ``0`` (MAP-R5). Every function here is **pure and deterministic** (no
clock, no randomness, no I/O) so the adapter's ``map`` stays golden-testable and a
byte-identical re-decode re-maps identically (GBO-AC-1). Kept separate from the
adapter/decode dispatch (``file_upload.py``) to bound module size (QUAL-R9).
"""

from __future__ import annotations

import datetime as _dt
import json
import math
from collections.abc import Mapping
from typing import Any, Final

from wattwise_core.domain.enums import (
    DeviceClass,
    SampleBasis,
    StreamChannelName,
)
from wattwise_core.ingestion.adapters._asbo import ActivityAsbo
from wattwise_core.storage import content_hash

# Per-sample record attribute -> canonical scalar channel (MAP-R4). ``latlng`` and
# ``rr_intervals_ms`` are handled separately (paired / event-spaced).
_SCALAR_CHANNELS: Final[tuple[tuple[str, StreamChannelName], ...]] = (
    ("power_w", StreamChannelName.POWER_W),
    ("hr_bpm", StreamChannelName.HR_BPM),
    ("cadence_rpm", StreamChannelName.CADENCE_RPM),
    ("speed_mps", StreamChannelName.SPEED_MPS),
    ("altitude_m", StreamChannelName.ALTITUDE_M),
    ("distance_m", StreamChannelName.DISTANCE_M),
    ("temp_c", StreamChannelName.TEMP_C),
)

# Source ``sub_sport`` vocab -> canonical sub_sport registry code (MAP-R4/R2). Values
# are SEEDED ``{sport}_other`` codes (migration 0001 §_seed) so the emitted value is a
# valid FK and never a raw source token; an unmapped token resolves to ``None``.
_SUB_SPORT_CODES: Final[dict[str, str]] = {
    "road": "cycling_other",
    "gravel": "cycling_other",
    "mountain": "cycling_other",
    "mountain_bike": "cycling_other",
    "cyclocross": "cycling_other",
    "indoor_cycling": "cycling_other",
    "spin": "cycling_other",
    "track_cycling": "cycling_other",
    "trail": "running_other",
    "track": "running_other",
    "treadmill": "running_other",
    "road_running": "running_other",
    "street": "running_other",
    "lap_swimming": "swimming_other",
    "open_water": "swimming_other",
    "indoor_rowing": "rowing_other",
    "classic": "xc_ski_other",
    "skate": "xc_ski_other",
}

# Source ``sport`` vocab -> canonical sport registry code (MAP-R4); unknown -> "other".
_SPORT_CODES: Final[dict[str, str]] = {
    "cycling": "cycling",
    "biking": "cycling",
    "ebiking": "cycling",
    "virtualride": "cycling",
    "ride": "cycling",
    "bike": "cycling",
    "running": "running",
    "run": "running",
    "treadmill": "running",
    "trail_running": "running",
    "swimming": "swimming",
    "swim": "swimming",
    "lap_swimming": "swimming",
    "open_water": "swimming",
    "rowing": "rowing",
    "kayaking": "rowing",
    "paddling": "rowing",
    "cross_country_skiing": "xc_ski",
    "nordic_skiing": "xc_ski",
    "xc_ski": "xc_ski",
    "training": "strength",
    "strength_training": "strength",
    "fitness_equipment": "strength",
    "generic": "other",
    "walking": "other",
    "hiking": "other",
}


def start_time(asbo: ActivityAsbo) -> _dt.datetime | None:
    """The session start instant: session field, else first record/lap (IDS-R3)."""
    session_start = as_dt(asbo.session.get("start_time"))
    if session_start is not None:
        return session_start
    # Record/lap fallbacks route through the SAME naive==UTC normalization as the
    # session field, so every instant the map emits or subtracts is UTC-aware
    # (TIER-R5 mapping-fuzz distilled failure: a naive record timestamp must never
    # surface as a naive canonical start_time or poison a datetime subtraction).
    for rec in asbo.records:
        normalized = as_dt(rec.timestamp)
        if normalized is not None:
            return normalized
    for lap in asbo.laps:
        normalized = as_dt(lap.start_time)
        if normalized is not None:
            return normalized
    return None


def build_streams(asbo: ActivityAsbo) -> dict[str, dict[str, Any]]:
    """Build canonical per-sample streams from the records (MAP-R5: gaps as ``None``)."""
    out: dict[str, dict[str, Any]] = {}
    records = asbo.records
    for attr, channel in _SCALAR_CHANNELS:
        values = [_finite(getattr(rec, attr)) for rec in records]
        if any(v is not None for v in values):
            out[channel.value] = _channel(values, SampleBasis.TIME)
    latlng = [_finite_latlng(rec.latlng) for rec in records]
    if any(v is not None for v in latlng):
        out[StreamChannelName.LATLNG.value] = _channel(latlng, SampleBasis.TIME)
    rr = [_finite(v) for v in asbo.rr_intervals_ms]
    if any(v is not None for v in rr):
        out[StreamChannelName.RR_INTERVALS_MS.value] = _channel(rr, SampleBasis.EVENT)
    return out


def _channel(values: list[Any], basis: SampleBasis) -> dict[str, Any]:
    return {"values": values, "sample_basis": basis.value, "sample_rate_hz": 1.0}


def has_per_sample_stream(streams: Mapping[str, Any]) -> bool:
    """True if any non-event per-sample stream exists (drives RAW_STREAM fidelity)."""
    return any(ch != StreamChannelName.RR_INTERVALS_MS.value for ch in streams)


def build_laps(asbo: ActivityAsbo, session_start: _dt.datetime) -> list[dict[str, Any]]:
    """Build canonical contiguous 0-based laps with relative offsets (MAP-R2/R3)."""
    laps: list[dict[str, Any]] = []
    for lap in asbo.laps:
        # Normalize the lap instant through the same naive==UTC convention as every
        # other decoded instant (``as_dt``): a decoder emitting a NAIVE lap start next
        # to an aware session start must yield a typed offset, never a TypeError from
        # naive-aware subtraction (TIER-R5 mapping-fuzz distilled failure).
        lap_start = as_dt(lap.start_time)
        offset = lap_start - session_start if lap_start is not None else None
        laps.append(
            {
                "lap_index": lap.lap_index,
                "start_offset_s": None if offset is None else int(offset.total_seconds()),
                # Route every lap scalar through the finite sink (MAP-R5): a non-finite
                # lap aggregate (e.g. a corrupt-FIT recovery field that bypasses the FIT
                # decoder's lenient float parse) must become a typed gap here too, not leak
                # into the payload as NaN/inf (invalid JSONB, non-deterministic hash) — and
                # ``int(inf/nan)`` must not raise an uncaught OverflowError in the pure map.
                "duration_s": _int(lap.duration_s),
                "distance_m": _num(lap.distance_m),
                "avg_power_w": _num(lap.avg_power_w),
                "max_power_w": _num(lap.max_power_w),
                "avg_hr_bpm": _num(lap.avg_hr_bpm),
                "max_hr_bpm": _num(lap.max_hr_bpm),
                "avg_cadence_rpm": _num(lap.avg_cadence_rpm),
            }
        )
    return laps


def activity_payload(
    asbo: ActivityAsbo,
    session_start: _dt.datetime,
    streams: dict[str, dict[str, Any]],
    laps: list[dict[str, Any]],
) -> dict[str, Any]:
    """Assemble the canonical ``activity`` payload (MAP-R2/R3; SI units, no source keys)."""
    s = asbo.session
    joules = _num(s.get("total_work")) or _num(s.get("total_joules"))
    return {
        "start_time": session_start,
        "sport": sport_code(s.get("sport")),
        "sub_sport": _sub_sport(s.get("sub_sport")),
        "elapsed_time_s": _int(s.get("total_elapsed_time")),
        "moving_time_s": _int(s.get("total_timer_time")),
        "distance_m": _num(s.get("total_distance")),
        "total_work_j": joules,
        "energy_kj": None if joules is None else joules / 1000.0,
        "avg_power_w": _num(s.get("avg_power")),
        "max_power_w": _num(s.get("max_power")),
        "avg_hr_bpm": _num(s.get("avg_heart_rate")),
        "max_hr_bpm": _num(s.get("max_heart_rate")),
        "avg_cadence_rpm": _num(s.get("avg_cadence")),
        "avg_speed_mps": _num(s.get("avg_speed") or s.get("enhanced_avg_speed")),
        "elevation_gain_m": _num(s.get("total_ascent")),
        "avg_temp_c": _num(s.get("avg_temperature")),
        "device_class": _device_class(streams),
        "has_power": StreamChannelName.POWER_W.value in streams
        or _num(s.get("avg_power")) is not None,
        "has_hr": StreamChannelName.HR_BPM.value in streams
        or _num(s.get("avg_heart_rate")) is not None,
        "has_gps": StreamChannelName.LATLNG.value in streams,
        "has_cadence": StreamChannelName.CADENCE_RPM.value in streams
        or _num(s.get("avg_cadence")) is not None,
        "streams": streams,
        "laps": laps,
    }


def _device_class(streams: Mapping[str, Any]) -> str:
    """Infer canonical device class from the channels present (MAP-R2; never a name)."""
    if StreamChannelName.POWER_W.value in streams:
        return DeviceClass.POWERMETER.value
    if StreamChannelName.LATLNG.value in streams:
        return DeviceClass.GPS_WATCH.value
    if StreamChannelName.HR_BPM.value in streams:
        return DeviceClass.GPS_WATCH.value
    return DeviceClass.UNKNOWN.value


def sport_code(raw: Any) -> str:
    """Map a source sport token to a canonical sport code (MAP-R4); unknown -> 'other'."""
    if not isinstance(raw, str):
        return "other"
    return _SPORT_CODES.get(raw.strip().lower(), "other")


def _sub_sport(raw: Any) -> str | None:
    """Map a source sub_sport token to a canonical registry code (MAP-R4/R2).

    Translates through ``_SUB_SPORT_CODES`` (seeded codes); an unmapped/absent token
    yields ``None`` (a typed gap). NEVER echoes the raw lowercased source token into a
    canonical FK field.
    """
    if not isinstance(raw, str):
        return None
    token = raw.strip().lower().replace(" ", "_").replace("-", "_")
    if not token or token in ("generic", "all"):
        return None
    return _SUB_SPORT_CODES.get(token)


def has_free_text(asbo: ActivityAsbo) -> bool:
    """True if the file carries a title/description (tagged untrusted, MAP-R7)."""
    return bool(asbo.session.get("title")) or bool(asbo.session.get("description"))


def stable_hash(payload: Mapping[str, Any]) -> str:
    """Deterministic sha256 over the canonical payload (MAP-R8; stable across runs)."""
    encoded = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return content_hash(encoded.encode("utf-8"))


def _num(value: Any) -> float | None:
    """Coerce to a FINITE float, else ``None`` (the canonical session-scalar sink, MAP-R5).

    The one place every adapter's session scalars (avg_power, total_work, ...) become
    canonical payload numbers, so the non-finite guard lives here: a NaN/inf — from a raw
    FIT field, an Intervals.icu value, or a malformed XML token — becomes a typed gap rather
    than a value that makes the payload invalid JSON (Postgres JSONB rejects NaN/Infinity)
    and non-deterministic (``nan != nan`` breaks the byte-identical re-decode, GBO-AC-1).
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        result = float(value)
        return result if math.isfinite(result) else None
    return None


def _finite(value: Any) -> Any:
    """Drop a non-finite float to ``None``; pass everything else through (MAP-R5)."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _finite_latlng(latlng: tuple[float, float] | None) -> list[float] | None:
    """A ``[lat, lon]`` pair only when BOTH coordinates are finite, else ``None`` (MAP-R5)."""
    if latlng is None:
        return None
    lat, lon = latlng
    if not (math.isfinite(lat) and math.isfinite(lon)):
        return None
    return [lat, lon]


def _int(value: Any) -> int | None:
    num = _num(value)
    return None if num is None else int(num)


def as_dt(value: Any) -> _dt.datetime | None:
    if isinstance(value, _dt.datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=_dt.UTC)
        return value.astimezone(_dt.UTC)
    return None


__all__ = [
    "activity_payload",
    "as_dt",
    "build_laps",
    "build_streams",
    "has_free_text",
    "has_per_sample_stream",
    "sport_code",
    "stable_hash",
    "start_time",
]
