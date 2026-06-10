"""Pure FIT decode layer for the file-upload adapter (CLI-R13, FIL-R*, TIER-R5).

Decodes verbatim FIT bytes into the shared :class:`ActivityAsbo` (records list +
session summary + laps + native fingerprint). Decoding is impure I/O kept OUT of the
adapter's pure ``map`` (ADP-R10/MAP-R1): this module turns bytes into a typed,
source-shaped object only — it never reads the clock, network, or randomness, and
never builds a canonical GBO.

Primary decoder is ``garmin-fit-sdk`` (official Garmin, stable cross-generation field
names); ``fitdecode`` is the corrupt/truncated-file recovery fallback (CLI-R13).
Any malformed input raises a TYPED :class:`FileDecodeError` (never a bare crash), so
arbitrary bytes fail closed at the boundary (TIER-R5 fuzz / CLI-R2).
"""

from __future__ import annotations

import datetime as _dt
import io
import warnings
from typing import Any

import fitdecode
from garmin_fit_sdk import Decoder, Stream

from wattwise_core.ingestion.adapters._asbo import (
    ActivityAsbo,
    AsboLap,
    AsboRecord,
    FileDecodeError,
)

# Garmin semicircle → WGS84 degrees: degrees = semicircles * 180 / 2**31 (FIT profile).
_SEMICIRCLE_TO_DEG = 180.0 / (2**31)


def decode_fit(data: bytes) -> ActivityAsbo:
    """Decode verbatim FIT bytes into an :class:`ActivityAsbo` (impure I/O).

    Tries the official SDK first; on any SDK failure falls back to ``fitdecode``
    for corrupt/truncated recovery (CLI-R13). Both paths raise a typed
    :class:`FileDecodeError` on malformed input — never a bare exception (TIER-R5).
    """
    try:
        return _decode_with_sdk(data)
    except FileDecodeError:
        return _decode_with_fitdecode(data)


def _decode_with_sdk(data: bytes) -> ActivityAsbo:
    try:
        decoder = Decoder(Stream.from_byte_array(data))
        if not decoder.is_fit():
            raise FileDecodeError("not a FIT file (bad header)")
        messages, errors = decoder.read()
    except FileDecodeError:
        raise
    except Exception as exc:
        raise FileDecodeError(f"FIT decode failed: {exc}") from exc
    return _build_asbo(messages, errors)


def _decode_with_fitdecode(data: bytes) -> ActivityAsbo:
    records: list[AsboRecord] = []
    session: dict[str, Any] = {}
    laps: list[AsboLap] = []
    file_id: dict[str, Any] = {}
    rr_ms: list[float] = []
    try:
        # fitdecode emits a UserWarning per malformed field on corrupt input; that is
        # the EXPECTED recovery path here (CLI-R13), so the noise is suppressed — the
        # bad bytes still surface as a typed error or are skipped, never fabricated.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with fitdecode.FitReader(
                io.BytesIO(data), check_crc=fitdecode.CrcCheck.DISABLED
            ) as reader:
                for frame in reader:
                    if not isinstance(frame, fitdecode.FitDataMessage):
                        continue
                    _ingest_fitdecode_frame(frame, records, session, laps, file_id, rr_ms)
    except FileDecodeError:
        raise
    except Exception as exc:
        raise FileDecodeError(f"FIT recovery decode failed: {exc}") from exc
    if not records and not session:
        raise FileDecodeError("FIT file contained no usable records or session")
    return ActivityAsbo(
        records=tuple(records),
        session=session,
        laps=tuple(laps),
        rr_intervals_ms=tuple(rr_ms),
        native_fingerprint=_fit_fingerprint(file_id),
        strong_fingerprint=_fit_strong_fingerprint(file_id),
    )


def _ingest_fitdecode_frame(
    frame: Any,
    records: list[AsboRecord],
    session: dict[str, Any],
    laps: list[AsboLap],
    file_id: dict[str, Any],
    rr_ms: list[float],
) -> None:
    fields = {f.name: f.value for f in frame.fields}
    name = frame.name
    if name == "record":
        records.append(_record_from_fields(fields))
    elif name == "session" and not session:
        session.update(fields)
    elif name == "lap":
        laps.append(_lap_from_fields(fields, len(laps)))
    elif name == "file_id" and not file_id:
        file_id.update(fields)
    elif name == "hrv":
        rr_ms.extend(_rr_from_hrv(fields.get("time")))


def _build_asbo(messages: dict[str, Any], errors: list[Any]) -> ActivityAsbo:
    record_mesgs = messages.get("record_mesgs") or []
    session_mesgs = messages.get("session_mesgs") or []
    lap_mesgs = messages.get("lap_mesgs") or []
    file_id_mesgs = messages.get("file_id_mesgs") or []
    hrv_mesgs = messages.get("hrv_mesgs") or []
    records = [_record_from_fields(m) for m in record_mesgs]
    laps = [_lap_from_fields(m, i) for i, m in enumerate(lap_mesgs)]
    session = dict(session_mesgs[0]) if session_mesgs else {}
    file_id = dict(file_id_mesgs[0]) if file_id_mesgs else {}
    rr_ms: list[float] = []
    for m in hrv_mesgs:
        rr_ms.extend(_rr_from_hrv(m.get("time")))
    if not records and not session:
        raise FileDecodeError("FIT file contained no usable records or session")
    return ActivityAsbo(
        records=tuple(records),
        session=session,
        laps=tuple(laps),
        rr_intervals_ms=tuple(rr_ms),
        native_fingerprint=_fit_fingerprint(file_id),
        strong_fingerprint=_fit_strong_fingerprint(file_id),
    )


def _record_from_fields(m: dict[str, Any]) -> AsboRecord:
    return AsboRecord(
        timestamp=_as_dt(m.get("timestamp")),
        power_w=_as_float(m.get("power")),
        hr_bpm=_as_float(m.get("heart_rate")),
        cadence_rpm=_as_float(m.get("cadence")),
        speed_mps=_as_float(_first_present(m, "enhanced_speed", "speed")),
        altitude_m=_as_float(_first_present(m, "enhanced_altitude", "altitude")),
        distance_m=_as_float(m.get("distance")),
        latlng=_latlng(m.get("position_lat"), m.get("position_long")),
        temp_c=_as_float(m.get("temperature")),
    )


def _lap_from_fields(m: dict[str, Any], index: int) -> AsboLap:
    return AsboLap(
        lap_index=index,
        start_time=_as_dt(m.get("start_time")),
        duration_s=_as_float(_first_present(m, "total_timer_time", "total_elapsed_time")),
        distance_m=_as_float(m.get("total_distance")),
        avg_power_w=_as_float(m.get("avg_power")),
        max_power_w=_as_float(m.get("max_power")),
        avg_hr_bpm=_as_float(m.get("avg_heart_rate")),
        max_hr_bpm=_as_float(m.get("max_heart_rate")),
        avg_cadence_rpm=_as_float(m.get("avg_cadence")),
    )


def _rr_from_hrv(time_field: Any) -> list[float]:
    """FIT ``hrv`` message ``time`` is RR in SECONDS → canonical milliseconds (MAP-R3)."""
    if not isinstance(time_field, list | tuple):
        return []
    out: list[float] = []
    for v in time_field:
        f = _as_float(v)
        if f is not None and f > 0:
            out.append(f * 1000.0)
    return out


def _fit_fingerprint(file_id: dict[str, Any]) -> str | None:
    """LIN-R1.1 FIT fingerprint: manufacturer+product+serial_number+time_created."""
    parts = [
        file_id.get("manufacturer"),
        _first_present(file_id, "product", "garmin_product"),
        file_id.get("serial_number"),
        file_id.get("time_created"),
    ]
    if all(p is None for p in parts):
        return None
    return "|".join("" if p is None else _stable_str(p) for p in parts)


def _fit_strong_fingerprint(file_id: dict[str, Any]) -> str | None:
    """The MAP-R10 STRONG fingerprint: a real shared device identity, or ``None``.

    Strong only when the FIT ``file_id`` carries BOTH a device ``serial_number`` AND a
    ``time_created`` instant — a genuine "this device recorded this session" identity
    two platforms exporting the same ride share. A stripped/degenerate ``file_id``
    (missing either) yields ``None``: it must NEVER merge unrelated sessions
    cross-window, so it falls back to the conservative windowed matcher.
    """
    serial = file_id.get("serial_number")
    created = file_id.get("time_created")
    if serial is None or created is None:
        return None
    return _fit_fingerprint(file_id)


def _stable_str(value: Any) -> str:
    if isinstance(value, _dt.datetime):
        return value.astimezone(_dt.UTC).isoformat()
    return str(value)


def _first_present(m: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if m.get(key) is not None:
            return m[key]
    return None


def _latlng(lat_semi: Any, long_semi: Any) -> tuple[float, float] | None:
    lat = _as_float(lat_semi)
    lon = _as_float(long_semi)
    if lat is None or lon is None:
        return None
    return (lat * _SEMICIRCLE_TO_DEG, lon * _SEMICIRCLE_TO_DEG)


def _as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _as_dt(value: Any) -> _dt.datetime | None:
    if isinstance(value, _dt.datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=_dt.UTC)
        return value.astimezone(_dt.UTC)
    return None


__all__ = ["decode_fit"]
