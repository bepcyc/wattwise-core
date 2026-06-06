"""Pure TCX decode layer for the file-upload adapter (CLI-R13, FIL-R*, TIER-R5).

Decodes verbatim TCX (Garmin TrainingCenterDatabase v2 XML) bytes into the shared
:class:`ActivityAsbo` via ``lxml`` (no maintained TCX-specific library exists).
Surfaces ``Trackpoint`` (Time/Position/Altitude/Distance/HeartRate/Cadence) plus
``Extensions/TPX`` (``Watts``/``Speed``) and ``Lap`` summaries. Decoding is impure
I/O kept OUT of the pure ``map`` (MAP-R1). Malformed XML raises the typed
:class:`FileDecodeError` (TIER-R5 / CLI-R2), never a bare crash.

XXE-safe: the parser is configured with entity resolution and network access
DISABLED (no external DTD/entity fetch) — untrusted upload bytes never trigger I/O.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any

from lxml import etree

from wattwise_core.ingestion.adapters._asbo import (
    ActivityAsbo,
    AsboLap,
    AsboRecord,
    FileDecodeError,
)


def decode_tcx(data: bytes) -> ActivityAsbo:
    """Decode verbatim TCX bytes into an :class:`ActivityAsbo` (impure I/O)."""
    root = _parse_xml(data)
    activity = _find(_find(root, "Activities"), "Activity")
    if activity is None:
        raise FileDecodeError("TCX file contained no Activity element")
    laps_el = _findall(activity, "Lap")
    records: list[AsboRecord] = []
    laps: list[AsboLap] = []
    for lap_index, lap_el in enumerate(laps_el):
        for track_el in _findall(lap_el, "Track"):
            for tp_el in _findall(track_el, "Trackpoint"):
                records.append(_record_from_tp(tp_el))
        laps.append(_lap_from_el(lap_el, lap_index))
    if not records and not laps:
        raise FileDecodeError("TCX file contained no trackpoints or laps")
    return ActivityAsbo(
        records=tuple(records),
        session=_session_from_activity(activity),
        laps=tuple(laps),
        rr_intervals_ms=(),
        native_fingerprint=_tcx_fingerprint(laps, records),
    )


def _parse_xml(data: bytes) -> Any:
    try:
        parser = etree.XMLParser(
            resolve_entities=False, no_network=True, dtd_validation=False, load_dtd=False
        )
        return etree.fromstring(data, parser=parser)
    except Exception as exc:
        raise FileDecodeError(f"TCX parse failed: {exc}") from exc


def _record_from_tp(tp_el: Any) -> AsboRecord:
    pos = _find(tp_el, "Position")
    ext = _find(tp_el, "Extensions")
    return AsboRecord(
        timestamp=_as_dt(_text(_find(tp_el, "Time"))),
        power_w=_tpx_value(ext, "Watts"),
        hr_bpm=_as_float(_text(_find(_find(tp_el, "HeartRateBpm"), "Value"))),
        cadence_rpm=_as_float(_text(_find(tp_el, "Cadence"))),
        speed_mps=_tpx_value(ext, "Speed"),
        altitude_m=_as_float(_text(_find(tp_el, "AltitudeMeters"))),
        distance_m=_as_float(_text(_find(tp_el, "DistanceMeters"))),
        latlng=_latlng(pos),
        temp_c=None,
    )


def _lap_from_el(lap_el: Any, lap_index: int) -> AsboLap:
    return AsboLap(
        lap_index=lap_index,
        start_time=_as_dt(lap_el.get("StartTime")),
        duration_s=_as_float(_text(_find(lap_el, "TotalTimeSeconds"))),
        distance_m=_as_float(_text(_find(lap_el, "DistanceMeters"))),
        avg_power_w=None,
        max_power_w=None,
        avg_hr_bpm=_as_float(_text(_find(_find(lap_el, "AverageHeartRateBpm"), "Value"))),
        max_hr_bpm=_as_float(_text(_find(_find(lap_el, "MaximumHeartRateBpm"), "Value"))),
        avg_cadence_rpm=_as_float(_text(_find(lap_el, "Cadence"))),
    )


def _session_from_activity(activity: Any) -> dict[str, object]:
    session: dict[str, object] = {}
    sport = activity.get("Sport")
    if sport:
        session["sport"] = sport
    notes = _text(_find(activity, "Notes"))
    if notes:
        session["title"] = notes
    return session


def _tcx_fingerprint(laps: list[AsboLap], records: list[AsboRecord]) -> str | None:
    """LIN-R1.1 TCX fingerprint: first lap/track start + total elapsed + total distance."""
    start = _first_instant(laps, records)
    if start is None:
        return None
    elapsed = sum(lap.duration_s or 0.0 for lap in laps)
    distance = sum(lap.distance_m or 0.0 for lap in laps)
    return f"{start.astimezone(_dt.UTC).isoformat()}|{elapsed:.3f}|{distance:.3f}"


def _first_instant(laps: list[AsboLap], records: list[AsboRecord]) -> _dt.datetime | None:
    for lap in laps:
        if lap.start_time is not None:
            return lap.start_time
    for rec in records:
        if rec.timestamp is not None:
            return rec.timestamp
    return None


def _tpx_value(extensions: Any, local_name: str) -> float | None:
    if extensions is None:
        return None
    for element in extensions.iter():
        if _local(element.tag) == local_name:
            return _as_float(_text(element))
    return None


def _latlng(pos: Any) -> tuple[float, float] | None:
    if pos is None:
        return None
    lat = _as_float(_text(_find(pos, "LatitudeDegrees")))
    lon = _as_float(_text(_find(pos, "LongitudeDegrees")))
    if lat is None or lon is None:
        return None
    return (lat, lon)


def _find(parent: Any, local_name: str) -> Any:
    if parent is None:
        return None
    for child in parent:
        if _local(child.tag) == local_name:
            return child
    return None


def _findall(parent: Any, local_name: str) -> list[Any]:
    if parent is None:
        return []
    return [child for child in parent if _local(child.tag) == local_name]


def _local(tag: Any) -> str:
    if not isinstance(tag, str):
        return ""
    return tag.rsplit("}", 1)[-1]


def _text(element: Any) -> str | None:
    if element is None:
        return None
    text = getattr(element, "text", None)
    return text.strip() if isinstance(text, str) else None


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_dt(value: Any) -> _dt.datetime | None:
    if not isinstance(value, str):
        return None
    raw = value.strip().replace("Z", "+00:00")
    try:
        parsed = _dt.datetime.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=_dt.UTC)
    return parsed.astimezone(_dt.UTC)


__all__ = ["decode_tcx"]
