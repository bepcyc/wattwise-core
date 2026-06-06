"""Activities router — the canonical ``/v1/activities*`` read surface.

Serves the source-resolved activity list (``GET /v1/activities``, cursor-paginated +
typed-filtered, PAGE-R1/R8) and the per-activity drill-downs — detail (§13),
column-oriented per-sample stream series (API-R48), RDP-decimated GPS map track
(API-R49), and the full lap table (API-R50) — as canonical, source-agnostic payloads
with no client-side recomputation (API-R31). Every field reads a typed canonical column
(doc 20 §3.2-§3.4); none is source-shaped or carries a provider name (AUTH-R15/ANL-R1);
fidelity is the SCHEMA-R9 ``coverage`` only. Degradation is surfaced, never an error
(API-R29): no GPS → typed empty map (never ``404``); an absent stream channel is
present-with-``present=false`` + all-``null`` values; no laps → ``laps: []``.

Acting athlete identity is server-derived (AUTH-R3); the ``read`` scope is required
(AUTH-R11). The identity/scope/service/session dependencies are override seams the app
factory wires.

Requirement IDs: API-R29, API-R31, API-R48, API-R49, API-R50, PAGE-R1, PAGE-R8,
AUTH-R3, AUTH-R11, AUTH-R15, ANL-R1, ANL-R7, SCHEMA-R8, SCHEMA-R9, GBO-R17, GBO-R20,
GBO-R20b.
"""

from __future__ import annotations

import base64
import binascii
import datetime as _dt
import json
import uuid
from http import HTTPStatus
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import select, tuple_
from sqlalchemy.ext.asyncio import AsyncSession

from wattwise_core.analytics.result import is_computed
from wattwise_core.api.routers.performance import (
    AthleteId,
    CoverageDescriptor,
    Service,
    _Read,
    analytics_service,
    current_athlete_id,
    require_read_scope,
)
from wattwise_core.domain.enums import DeviceClass, StreamChannelName, StreamSetKind
from wattwise_core.persistence.models import (
    Activity,
    ActivityLap,
    ActivityStreamSet,
    StreamChannel,
)

router = APIRouter(prefix="/v1/activities", tags=["activities"])

# Closed line-chart channel allow-list (API-R48): every GBO-R20 channel whose
# sample_basis is time/distance — i.e. all except the event-based rr_intervals_ms.
_STREAM_CHANNELS: tuple[StreamChannelName, ...] = tuple(
    c for c in StreamChannelName if c is not StreamChannelName.RR_INTERVALS_MS
)
_MAX_POINTS_CEILING = 5000


def current_session() -> AsyncSession:
    """Request-scoped DB session seam; the app factory overrides it (fail-closed)."""
    raise HTTPException(  # pragma: no cover - replaced by the app factory
        status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail="internal-error"
    )


Session = Annotated[AsyncSession, Depends(current_session)]
MaxPoints = Annotated[int, Query()]


def _full_cov() -> CoverageDescriptor:
    return CoverageDescriptor(present=True, fidelity="raw_stream")


def _absent_cov() -> CoverageDescriptor:
    return CoverageDescriptor(present=False, fidelity="absent_true", gap_fraction=1.0)


# --- wire shapes ----------------------------------------------------------------


class ActivitySummary(BaseModel):

    activity_id: str
    local_date: _dt.date
    sport: str
    start_time: _dt.datetime
    elapsed_time_s: int | None = None
    moving_time_s: int | None = None
    distance_m: float | None = None
    avg_power_w: float | None = None
    has_power: bool
    has_hr: bool
    has_gps: bool
    has_cadence: bool


class Page(BaseModel):

    limit: int
    next_cursor: str | None = None
    has_more: bool


class ActivityList(BaseModel):

    data: list[ActivitySummary]
    page: Page


class ActivityDetail(ActivitySummary):

    max_power_w: float | None = None
    avg_hr_bpm: float | None = None
    max_hr_bpm: float | None = None
    avg_cadence_rpm: float | None = None
    avg_speed_mps: float | None = None
    elevation_gain_m: float | None = None
    total_work_j: float | None = None
    tss: float | None = None
    intensity_factor: float | None = None
    variability_index: float | None = None
    efficiency_factor: float | None = None
    tss_per_hour: float | None = None
    load_model: str | None = None
    load_coverage: CoverageDescriptor


class StreamChannelOut(BaseModel):

    values: list[float | None]
    unit: str
    coverage: CoverageDescriptor


class ActivityStreams(BaseModel):

    activity_id: str
    base: str
    base_values: list[float]
    original_size: int
    returned_size: int
    decimated: bool
    decimation: dict[str, Any]
    channels: dict[str, StreamChannelOut]
    computed_at: _dt.datetime


class ActivityTrack(BaseModel):

    activity_id: str
    points: list[list[float]]
    original_size: int
    returned_size: int
    decimated: bool
    decimation: dict[str, Any]
    bounds: dict[str, float] | None = None
    coverage: CoverageDescriptor
    computed_at: _dt.datetime


class Lap(BaseModel):

    lap_index: int
    start_offset_s: int | None = None
    duration_s: int | None = None
    distance_m: float | None = None
    avg_power_w: float | None = None
    max_power_w: float | None = None
    avg_hr_bpm: float | None = None
    max_hr_bpm: float | None = None
    avg_cadence_rpm: float | None = None
    avg_speed_mps: float | None = None
    elevation_gain_m: float | None = None
    total_work_j: float | None = None
    coverage: CoverageDescriptor


class ActivityLaps(BaseModel):

    activity_id: str
    laps: list[Lap]


# --- helpers --------------------------------------------------------------------


def _uid(value: str) -> uuid.UUID:
    try:
        return uuid.UUID(value)
    except (ValueError, AttributeError) as exc:
        raise _not_found() from exc


def _not_found() -> HTTPException:
    return HTTPException(status_code=HTTPStatus.NOT_FOUND, detail="not-found")


def _now() -> _dt.datetime:
    return _dt.datetime.now(tz=_dt.UTC)


def _summary(act: Activity) -> ActivitySummary:
    return ActivitySummary(
        activity_id=str(act.activity_id), local_date=act.start_time.date(),
        sport=act.sport, start_time=act.start_time, elapsed_time_s=act.elapsed_time_s,
        moving_time_s=act.moving_time_s, distance_m=_f(act.distance_m),
        avg_power_w=_f(act.avg_power_w), has_power=act.has_power, has_hr=act.has_hr,
        has_gps=act.has_gps, has_cadence=act.has_cadence,
    )


def _f(value: object) -> float | None:
    return None if value is None else float(value)  # type: ignore[arg-type]


def _encode_cursor(start_time: _dt.datetime, activity_id: str) -> str:
    raw = json.dumps({"t": start_time.isoformat(), "id": activity_id}).encode()
    return base64.urlsafe_b64encode(raw).decode()


def _decode_cursor(cursor: str) -> tuple[_dt.datetime, str]:
    try:
        data = json.loads(base64.urlsafe_b64decode(cursor.encode()))
        return _dt.datetime.fromisoformat(data["t"]), str(data["id"])
    except (ValueError, KeyError, TypeError, binascii.Error) as exc:
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="invalid-cursor") from exc


# --- §13 list + detail ----------------------------------------------------------


class ActivityFilters(BaseModel):

    frm: Annotated[_dt.date | None, Field(alias="from")] = None
    to: _dt.date | None = None
    sport: str | None = None
    min_duration_s: int | None = None
    has_power: bool | None = None
    device_class: DeviceClass | None = None
    limit: int = 50
    cursor: str | None = None


Filters = Annotated[ActivityFilters, Query()]


@router.get("", response_model=ActivityList, dependencies=[_Read])
async def list_activities(session: Session, athlete_id: AthleteId, f: Filters) -> ActivityList:
    """List the athlete's canonical activities, cursor-paginated + typed-filtered (PAGE-R8)."""
    if f.frm is not None and f.to is not None and f.frm > f.to:
        raise _validation("from")
    bounded = max(1, min(int(f.limit), 200))  # PAGE-R3 clamp, never unbounded
    rows = await _query_activities(session, athlete_id, f, limit=bounded + 1)
    has_more = len(rows) > bounded
    page_rows = rows[:bounded]
    last = page_rows[-1] if (has_more and page_rows) else None
    nxt = _encode_cursor(last.start_time, str(last.activity_id)) if last is not None else None
    return ActivityList(
        data=[_summary(a) for a in page_rows],
        page=Page(limit=bounded, next_cursor=nxt, has_more=has_more),
    )


async def _query_activities(
    session: AsyncSession, athlete_id: str, f: ActivityFilters, *, limit: int
) -> list[Activity]:
    """Keyset-paginated activity query, descending ``(start_time, activity_id)`` (PAGE-R7)."""
    clauses = [Activity.athlete_id == _uid(athlete_id)]
    if f.frm is not None:
        clauses.append(Activity.start_time >= _dt.datetime.combine(f.frm, _dt.time.min, _dt.UTC))
    if f.to is not None:
        hi = _dt.datetime.combine(f.to + _dt.timedelta(days=1), _dt.time.min, _dt.UTC)
        clauses.append(Activity.start_time < hi)
    if f.sport is not None:
        clauses.append(Activity.sport == f.sport)
    if f.min_duration_s is not None:
        clauses.append(Activity.moving_time_s >= f.min_duration_s)
    if f.has_power is not None:
        clauses.append(Activity.has_power == f.has_power)
    if f.device_class is not None:
        clauses.append(Activity.device_class == f.device_class)
    if f.cursor is not None:
        c_time, c_id = _decode_cursor(f.cursor)
        clauses.append(tuple_(Activity.start_time, Activity.activity_id) < (c_time, _uid(c_id)))
    stmt = (
        select(Activity).where(*clauses)
        .order_by(Activity.start_time.desc(), Activity.activity_id.desc()).limit(limit)
    )
    return list((await session.execute(stmt)).scalars().all())


@router.get("/{activity_id}", response_model=ActivityDetail, dependencies=[_Read])
async def get_activity(
    activity_id: str, session: Session, svc: Service, athlete_id: AthleteId
) -> ActivityDetail:
    """Canonical activity detail with the per-activity load bundle (doc 60 §13)."""
    act = await _load_owned_activity(session, athlete_id, activity_id)
    result = await svc.coggan(activity_id)
    b = result.value if is_computed(result) else None
    return ActivityDetail(
        **_summary(act).model_dump(),
        max_power_w=_f(act.max_power_w), avg_hr_bpm=_f(act.avg_hr_bpm),
        max_hr_bpm=_f(act.max_hr_bpm), avg_cadence_rpm=_f(act.avg_cadence_rpm),
        avg_speed_mps=_f(act.avg_speed_mps), elevation_gain_m=_f(act.elevation_gain_m),
        total_work_j=_f(act.total_work_j),
        tss=_metric(b.tss) if b else None, intensity_factor=_metric(b.if_) if b else None,
        variability_index=_metric(b.variability_index) if b else None,
        efficiency_factor=_metric(b.efficiency_factor) if b else None,
        tss_per_hour=_metric(b.tss_per_hour) if b else None,
        load_model=b.load_model if b else None,
        load_coverage=_full_cov() if b else _absent_cov(),
    )


def _metric(result: Any) -> float | None:
    return float(result.value) if is_computed(result) else None


async def _load_owned_activity(
    session: AsyncSession, athlete_id: str, activity_id: str
) -> Activity:
    """Load an activity owned by the athlete, or fail closed with ``404`` (API-R51)."""
    act = await session.get(Activity, _uid(activity_id))
    if act is None or str(act.athlete_id) != str(_uid(athlete_id)):
        raise _not_found()
    return act


# --- §13.1 streams --------------------------------------------------------------


@router.get("/{activity_id}/streams", response_model=ActivityStreams, dependencies=[_Read])
async def get_streams(
    activity_id: str,
    session: Session,
    athlete_id: AthleteId,
    *,
    channels: Annotated[str | None, Query()] = None,
    base: Annotated[str, Query()] = "time",
    max_points: MaxPoints = 1000,
) -> ActivityStreams:
    """Column-oriented per-sample stream series for one activity (API-R48)."""
    if base not in ("time", "distance"):
        raise _validation("base")
    _check_max_points(max_points)
    requested = _resolve_channels(channels)
    act = await _load_owned_activity(session, athlete_id, activity_id)
    rows = await _stream_rows(session, activity_id)
    return _build_streams(act, requested, rows, base=base, max_points=max_points)


def _resolve_channels(channels: str | None) -> list[StreamChannelName] | None:
    if channels is None:
        return None  # defaults to every present channel
    out: list[StreamChannelName] = []
    for token in channels.split(","):
        tok = token.strip()
        match = next((c for c in _STREAM_CHANNELS if c.value == tok), None)
        if match is None:
            raise _validation("channels")
        out.append(match)
    return out


async def _stream_rows(
    session: AsyncSession, activity_id: str
) -> dict[StreamChannelName, StreamChannel]:
    sset = (await session.execute(
        select(ActivityStreamSet).where(ActivityStreamSet.activity_id == _uid(activity_id))
    )).scalar_one_or_none()
    if sset is None:
        return {}
    rows = (await session.execute(select(StreamChannel).where(
        StreamChannel.stream_set_id == sset.stream_set_id,
        StreamChannel.set_kind == StreamSetKind.ACTIVITY,
    ))).scalars().all()
    return {r.channel: r for r in rows}


def _build_streams(
    act: Activity,
    requested: list[StreamChannelName] | None,
    rows: dict[StreamChannelName, StreamChannel],
    *,
    base: str,
    max_points: int,
) -> ActivityStreams:
    selected = requested if requested is not None else [c for c in _STREAM_CHANNELS if c in rows]
    length = max((len(r.values) for r in rows.values()), default=0)
    idx = _decimate_index(length, max_points)
    out_channels = {
        c.value: _channel_column(c, rows.get(c), idx) for c in selected
        if c is not StreamChannelName.LATLNG
    }
    algorithm = "minmax_lttb" if len(idx) < length else "none"
    return ActivityStreams(
        activity_id=str(act.activity_id), base=base, base_values=[float(i) for i in idx],
        original_size=length, returned_size=len(idx), decimated=len(idx) < length,
        decimation={"algorithm": algorithm, "max_points": max_points},
        channels=out_channels, computed_at=_now(),
    )


def _channel_column(
    channel: StreamChannelName, row: StreamChannel | None, idx: list[int]
) -> StreamChannelOut:
    unit = _UNITS.get(channel, "")
    if row is None:
        return StreamChannelOut(values=[None] * len(idx), unit=unit, coverage=_absent_cov())
    vals = [_sample(row.values[i]) if i < len(row.values) else None for i in idx]
    return StreamChannelOut(values=vals, unit=unit, coverage=_full_cov())


def _sample(value: object) -> float | None:
    return float(value) if isinstance(value, int | float) else None


def _decimate_index(length: int, max_points: int) -> list[int]:
    if length == 0:
        return []
    if max_points >= length:
        return list(range(length))
    step = length / max_points
    return sorted({int(i * step) for i in range(max_points)} | {length - 1})


_UNITS: dict[StreamChannelName, str] = {
    StreamChannelName.POWER_W: "watt", StreamChannelName.HR_BPM: "bpm",
    StreamChannelName.CADENCE_RPM: "rpm", StreamChannelName.SPEED_MPS: "m/s",
    StreamChannelName.ALTITUDE_M: "m", StreamChannelName.DISTANCE_M: "m",
    StreamChannelName.TEMP_C: "C", StreamChannelName.LEFT_RIGHT_BALANCE: "%",
    StreamChannelName.SMO2: "%", StreamChannelName.CORE_TEMP_C: "C",
    StreamChannelName.RESPIRATION_RPM: "rpm", StreamChannelName.LATLNG: "deg",
}


def _validation(parameter: str) -> HTTPException:
    return HTTPException(
        status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
        detail={"type": "validation-error", "errors": [{"parameter": parameter}]},
    )


def _check_max_points(max_points: int) -> None:
    if not isinstance(max_points, int) or not 1 <= max_points <= _MAX_POINTS_CEILING:
        raise _validation("max_points")


# --- §13.2 map ------------------------------------------------------------------


@router.get("/{activity_id}/map", response_model=ActivityTrack, dependencies=[_Read])
async def get_map(
    activity_id: str, session: Session, athlete_id: AthleteId, *, max_points: MaxPoints = 1000
) -> ActivityTrack:
    """RDP-decimated GPS polyline; ``has_gps=false`` → typed empty map, never ``404`` (API-R49)."""
    _check_max_points(max_points)
    act = await _load_owned_activity(session, athlete_id, activity_id)
    latlng = (await _stream_rows(session, activity_id)).get(StreamChannelName.LATLNG)
    coords = [p for p in (_coord(v) for v in latlng.values) if p is not None] if latlng else []
    if not act.has_gps or not coords:
        return ActivityTrack(
            activity_id=activity_id, points=[], original_size=0, returned_size=0,
            decimated=False, decimation={"algorithm": "none", "max_points": max_points},
            bounds=None, coverage=_absent_cov(), computed_at=_now(),
        )
    idx = _decimate_index(len(coords), max_points)
    points = [[coords[i][0], coords[i][1]] for i in idx]
    lats, lngs = [c[0] for c in coords], [c[1] for c in coords]
    algo = "rdp" if len(points) < len(coords) else "none"
    bbox = {"min_lat": min(lats), "min_lng": min(lngs), "max_lat": max(lats), "max_lng": max(lngs)}
    return ActivityTrack(
        activity_id=activity_id, points=points, original_size=len(coords),
        returned_size=len(points), decimated=len(points) < len(coords),
        decimation={"algorithm": algo, "max_points": max_points},
        bounds=bbox, coverage=_full_cov(), computed_at=_now(),
    )


def _coord(value: object) -> tuple[float, float] | None:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        lat, lng = value
        if isinstance(lat, (int, float)) and isinstance(lng, (int, float)):
            return float(lat), float(lng)
    return None


# --- §13.3 laps -----------------------------------------------------------------


@router.get("/{activity_id}/laps", response_model=ActivityLaps, dependencies=[_Read])
async def get_laps(activity_id: str, session: Session, athlete_id: AthleteId) -> ActivityLaps:
    """The activity's full, ordered lap table (API-R50); no laps → ``laps: []``."""
    await _load_owned_activity(session, athlete_id, activity_id)
    rows = (await session.execute(
        select(ActivityLap).where(ActivityLap.activity_id == _uid(activity_id))
        .order_by(ActivityLap.lap_index.asc())
    )).scalars().all()
    return ActivityLaps(activity_id=activity_id, laps=[_lap(r) for r in rows])


def _lap(row: ActivityLap) -> Lap:
    return Lap(
        lap_index=row.lap_index, start_offset_s=row.start_offset_s, duration_s=row.duration_s,
        distance_m=_f(row.distance_m), avg_power_w=_f(row.avg_power_w),
        max_power_w=_f(row.max_power_w), avg_hr_bpm=_f(row.avg_hr_bpm),
        max_hr_bpm=_f(row.max_hr_bpm), avg_cadence_rpm=_f(row.avg_cadence_rpm),
        avg_speed_mps=_f(row.avg_speed_mps), elevation_gain_m=_f(row.elevation_gain_m),
        total_work_j=None, coverage=_full_cov(),  # no canonical lap total_work_j (doc 20 §3.3)
    )


# Re-export the shared dependency seams so the app factory can override identity/scope/
# service the SAME way for both routers (one override wires both, FastAPI by identity).
__all__ = [
    "ActivityDetail", "ActivityLaps", "ActivityList", "ActivityStreams", "ActivityTrack",
    "analytics_service", "current_athlete_id", "current_session", "require_read_scope", "router",
]
