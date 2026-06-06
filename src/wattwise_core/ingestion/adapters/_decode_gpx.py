"""Pure GPX decode layer for the file-upload adapter (CLI-R13, FIL-R*, TIER-R5).

Decodes verbatim GPX (XML) bytes into the shared :class:`ActivityAsbo` via ``gpxpy``.
Surfaces track points (lat/lon/elevation/time) plus ``gpxtpx`` extensions
(``hr``/``cad``/``power``/``atemp``/``speed``). Decoding is impure I/O kept OUT of the
pure ``map`` (MAP-R1). Malformed XML raises the typed :class:`FileDecodeError`
(TIER-R5 / CLI-R2), never a bare crash.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any

import gpxpy

from wattwise_core.ingestion.adapters._asbo import (
    ActivityAsbo,
    AsboRecord,
    FileDecodeError,
)

# Namespace-agnostic local tags read from a point's extension subtree.
_EXT_HR = ("hr", "heartrate")
_EXT_CAD = ("cad", "cadence")
_EXT_POWER = ("power", "watts", "pwr")
_EXT_TEMP = ("atemp", "temp", "temperature")
_EXT_SPEED = ("speed",)


def decode_gpx(data: bytes) -> ActivityAsbo:
    """Decode verbatim GPX bytes into an :class:`ActivityAsbo` (impure I/O).

    Hardened against XXE / entity-expansion (CLI-R2/ING-R7, TIER-R5): a DTD or entity
    declaration is rejected before parsing — a valid GPX never carries one, and refusing
    them closes the external-entity / billion-laughs vector independent of which XML
    backend ``gpxpy`` selects (a fail-closed guard, not a reliance on a safe default).
    """
    head = data[:4096].lstrip().lower()
    if b"<!doctype" in head or b"<!entity" in data[:65536].lower():
        raise FileDecodeError("GPX with a DTD/entity declaration is rejected (XXE guard)")
    try:
        gpx = gpxpy.parse(data.decode("utf-8", errors="strict"))
    except FileDecodeError:
        raise
    except Exception as exc:
        raise FileDecodeError(f"GPX parse failed: {exc}") from exc
    records = _collect_points(gpx)
    if not records:
        raise FileDecodeError("GPX file contained no track points")
    return ActivityAsbo(
        records=tuple(records),
        session=_session_from_gpx(gpx, records),
        laps=(),
        rr_intervals_ms=(),
        native_fingerprint=_gpx_fingerprint(records),
    )


def _collect_points(gpx: Any) -> list[AsboRecord]:
    records: list[AsboRecord] = []
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                records.append(_record_from_point(point))
    return records


def _record_from_point(point: Any) -> AsboRecord:
    ext = _extensions_map(getattr(point, "extensions", None))
    return AsboRecord(
        timestamp=_as_dt(getattr(point, "time", None)),
        power_w=_pick(ext, _EXT_POWER),
        hr_bpm=_pick(ext, _EXT_HR),
        cadence_rpm=_pick(ext, _EXT_CAD),
        speed_mps=_pick(ext, _EXT_SPEED),
        altitude_m=_as_float(getattr(point, "elevation", None)),
        distance_m=None,
        latlng=_latlng(getattr(point, "latitude", None), getattr(point, "longitude", None)),
        temp_c=_pick(ext, _EXT_TEMP),
    )


def _extensions_map(extensions: Any) -> dict[str, float]:
    """Flatten a point's gpxtpx extension subtree into ``{local_tag: float}``."""
    out: dict[str, float] = {}
    if not extensions:
        return out
    for element in extensions:
        _walk_extension(element, out)
    return out


def _walk_extension(element: Any, out: dict[str, float]) -> None:
    tag = getattr(element, "tag", None)
    if isinstance(tag, str):
        local = tag.rsplit("}", 1)[-1].lower()
        value = _as_float_text(getattr(element, "text", None))
        if value is not None and local not in out:
            out[local] = value
    for child in list(element) if _iterable_element(element) else ():
        _walk_extension(child, out)


def _iterable_element(element: Any) -> bool:
    try:
        iter(element)
        return not isinstance(element, str | bytes)
    except TypeError:
        return False


def _pick(ext: dict[str, float], names: tuple[str, ...]) -> float | None:
    for name in names:
        if name in ext:
            return ext[name]
    return None


def _session_from_gpx(gpx: Any, records: list[AsboRecord]) -> dict[str, object]:
    session: dict[str, object] = {}
    name = None
    for track in gpx.tracks:
        if getattr(track, "name", None):
            name = track.name
            break
    if name is None:
        name = getattr(gpx, "name", None)
    if name:
        session["title"] = name
    typ = next((t.type for t in gpx.tracks if getattr(t, "type", None)), None)
    if typ:
        session["sport"] = typ
    return session


def _gpx_fingerprint(records: list[AsboRecord]) -> str | None:
    """LIN-R1.1 GPX fingerprint: first point start instant + elapsed + point count."""
    start = next((r.timestamp for r in records if r.timestamp is not None), None)
    end = next((r.timestamp for r in reversed(records) if r.timestamp is not None), None)
    if start is None:
        return None
    elapsed = (end - start).total_seconds() if end is not None else 0.0
    return f"{start.astimezone(_dt.UTC).isoformat()}|{elapsed:.3f}|{len(records)}"


def _latlng(lat: Any, lon: Any) -> tuple[float, float] | None:
    flat = _as_float(lat)
    flon = _as_float(lon)
    if flat is None or flon is None:
        return None
    return (flat, flon)


def _as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _as_float_text(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _as_dt(value: Any) -> _dt.datetime | None:
    if isinstance(value, _dt.datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=_dt.UTC)
        return value.astimezone(_dt.UTC)
    return None


__all__ = ["decode_gpx"]
