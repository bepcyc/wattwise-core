"""Pure PWX decode layer for the file-upload adapter (CLI-R13, FIL-R*, TIER-R5).

Decodes verbatim PWX (TrainingPeaks PWX/1/0 XML) bytes into the shared
:class:`ActivityAsbo` via ``lxml`` (same backend/helpers as ``_decode_tcx``). PWX
uses the default namespace ``http://www.peaksware.com/PWX/1/0`` so every lookup is
namespace-agnostic (local-name only). Surfaces the workout ``<summarydata>`` summary,
``<segment>`` laps, and per-``<sample>`` streams (timeoffset/hr/spd/pwr/cad/dist/alt/
lat/lon). Decoding is impure I/O kept OUT of the pure ``map`` (MAP-R1). Malformed XML
raises the typed :class:`FileDecodeError` (TIER-R5 / CLI-R2), never a bare crash.

XXE-safe: a DTD/entity declaration is rejected before parse, and the parser is
configured with entity resolution and network access DISABLED — untrusted upload bytes
never trigger external I/O.
"""

from __future__ import annotations

import datetime as _dt
import math
from typing import Any

from lxml import etree

from wattwise_core.ingestion.adapters._asbo import (
    ActivityAsbo,
    AsboLap,
    AsboRecord,
    FileDecodeError,
)


def decode_pwx(data: bytes) -> ActivityAsbo:
    """Decode verbatim PWX bytes into an :class:`ActivityAsbo` (impure I/O)."""
    root = _parse_xml(data)
    if _local(root.tag) != "pwx":
        raise FileDecodeError("PWX file root element is not <pwx>")
    workout = _find(root, "workout")
    if workout is None:
        raise FileDecodeError("PWX file contained no workout element")
    session = _session_from_workout(workout)
    start = _as_dt(session.get("start_time"))
    summary = _find(workout, "summarydata")
    records = [_record_from_sample(s, start) for s in _findall(workout, "sample")]
    laps = [
        _lap_from_segment(seg, idx, start) for idx, seg in enumerate(_findall(workout, "segment"))
    ]
    if not records and not laps and summary is None:
        # A populated workout <summarydata> with no <sample>/<segment> is a legitimate
        # summary-only export (maps to a SUMMARY_ONLY-tier candidate); only a workout with
        # NO samples, NO segments, AND no summary at all is genuinely empty -> fail closed.
        raise FileDecodeError("PWX file contained no samples, segments, or summary")
    return ActivityAsbo(
        records=tuple(records),
        session=session,
        laps=tuple(laps),
        rr_intervals_ms=(),
        native_fingerprint=_pwx_fingerprint(start, summary, records),
    )


def _parse_xml(data: bytes) -> Any:
    head = data[:4096].lstrip().lower()
    if b"<!doctype" in head or b"<!entity" in data[:65536].lower():
        raise FileDecodeError("PWX with a DTD/entity declaration is rejected (XXE guard)")
    try:
        parser = etree.XMLParser(
            resolve_entities=False, no_network=True, dtd_validation=False, load_dtd=False
        )
        return etree.fromstring(data, parser=parser)
    except FileDecodeError:
        raise
    except Exception as exc:
        raise FileDecodeError(f"PWX parse failed: {exc}") from exc


def _record_from_sample(sample: Any, start: _dt.datetime | None) -> AsboRecord:
    offset = _as_float(_text(_find(sample, "timeoffset")))
    timestamp = None
    if start is not None and offset is not None:
        timestamp = start + _dt.timedelta(seconds=offset)
    return AsboRecord(
        timestamp=timestamp,
        power_w=_as_float(_text(_find(sample, "pwr"))),
        hr_bpm=_as_float(_text(_find(sample, "hr"))),
        cadence_rpm=_as_float(_text(_find(sample, "cad"))),
        speed_mps=_as_float(_text(_find(sample, "spd"))),
        altitude_m=_as_float(_text(_find(sample, "alt"))),
        distance_m=_as_float(_text(_find(sample, "dist"))),
        latlng=_latlng(sample),
        temp_c=None,
    )


def _lap_from_segment(segment: Any, lap_index: int, start: _dt.datetime | None) -> AsboLap:
    summary = _find(segment, "summarydata")
    beginning = _as_float(_text(_find(summary, "beginning")))
    lap_start = None
    if start is not None and beginning is not None:
        lap_start = start + _dt.timedelta(seconds=beginning)
    return AsboLap(
        lap_index=lap_index,
        start_time=lap_start,
        duration_s=_as_float(_text(_find(summary, "duration"))),
        distance_m=_as_float(_text(_find(summary, "dist"))),
        avg_power_w=_summary_attr(summary, "pwr", "avg"),
        max_power_w=_summary_attr(summary, "pwr", "max"),
        avg_hr_bpm=_summary_attr(summary, "hr", "avg"),
        max_hr_bpm=_summary_attr(summary, "hr", "max"),
        avg_cadence_rpm=_summary_attr(summary, "cad", "avg"),
    )


def _session_from_workout(workout: Any) -> dict[str, object]:
    session: dict[str, object] = {}
    _put(session, "sport", _text(_find(workout, "sportType")))
    _put(session, "title", _text(_find(workout, "title")))
    _put(session, "description", _text(_find(workout, "cmt")))
    _put(session, "start_time", _as_dt(_text(_find(workout, "time"))))
    summary = _find(workout, "summarydata")
    if summary is not None:
        _put(session, "total_elapsed_time", _as_float(_text(_find(summary, "duration"))))
        _put(session, "total_distance", _as_float(_text(_find(summary, "dist"))))
        # PWX <work> is documented in KILOJOULES; the canonical convention (session
        # "total_work" -> total_work_j) is JOULES, so convert at decode time (MAP-R3 SI
        # normalisation), matching FIT's native-joules work the canonical map assumes.
        work_kj = _as_float(_text(_find(summary, "work")))
        _put(session, "total_work", None if work_kj is None else work_kj * 1000.0)
        _put(session, "avg_power", _summary_attr(summary, "pwr", "avg"))
        _put(session, "max_power", _summary_attr(summary, "pwr", "max"))
        _put(session, "avg_heart_rate", _summary_attr(summary, "hr", "avg"))
        _put(session, "max_heart_rate", _summary_attr(summary, "hr", "max"))
        _put(session, "avg_cadence", _summary_attr(summary, "cad", "avg"))
        _put(session, "avg_speed", _summary_attr(summary, "spd", "avg"))
    return session


def _put(session: dict[str, object], key: str, value: object | None) -> None:
    if value is not None:
        session[key] = value


def _summary_attr(summary: Any, child_local: str, attr: str) -> float | None:
    child = _find(summary, child_local)
    if child is None:
        return None
    return _as_float(child.get(attr))


def _pwx_fingerprint(
    start: _dt.datetime | None, summary: Any, records: list[AsboRecord]
) -> str | None:
    """LIN-R1.1 PWX fingerprint: workout start + elapsed + a content discriminator."""
    if start is None:
        return None
    elapsed = _as_float(_text(_find(summary, "duration"))) or 0.0
    distance = _as_float(_text(_find(summary, "dist")))
    discriminator = f"{distance:.3f}" if distance is not None else str(len(records))
    return f"{start.astimezone(_dt.UTC).isoformat()}|{elapsed:.3f}|{discriminator}"


def _latlng(sample: Any) -> tuple[float, float] | None:
    lat = _as_float(_text(_find(sample, "lat")))
    lon = _as_float(_text(_find(sample, "lon")))
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
    """Parse a finite float, else ``None`` (a NaN/inf token is a typed gap, not a value).

    Rejecting non-finite values keeps the canonical payload JSON-valid (``NaN``/``Infinity``
    are not legal JSON and Postgres JSONB rejects them) and deterministic (``nan != nan``
    would break the byte-identical re-decode guarantee, GBO-AC-1 / MAP-R5).
    """
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _as_dt(value: Any) -> _dt.datetime | None:
    if isinstance(value, _dt.datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=_dt.UTC)
        return value.astimezone(_dt.UTC)
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


__all__ = ["decode_pwx"]
