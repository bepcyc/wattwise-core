"""Activities router — the canonical ``/v1/activities*`` read surface.

Serves the source-resolved activity list (``GET /v1/activities``, cursor-paginated +
typed-filtered + typed-sorted, PAGE-R1/R2/R5/R6/R8) and the per-activity drill-downs —
detail (§13), column-oriented per-sample stream series (API-R48), RDP-decimated GPS map
track (API-R49), and the full lap table (API-R50) — as canonical, source-agnostic
payloads with no client-side recomputation (API-R31). Every field reads a typed canonical
column (doc 20 §3.2-§3.4); none is source-shaped or carries a provider name
(AUTH-R15/ANL-R1); fidelity is the SCHEMA-R9 ``coverage`` only. Degradation is surfaced,
never an error (API-R29): no GPS → typed empty map (never ``404``); an absent stream
channel is present-with-``present=false`` + all-``null`` values; no laps → ``laps: []``.

Every non-2xx is the catalog :class:`ProblemError` (a tampered cursor → ``invalid-cursor``;
a cursor replayed against changed filters/sort → ``cursor-parameter-mismatch``; a bad
query param → ``validation-error`` with the offending ``parameter``; an unknown id →
``not-found``) — never a raw framework ``HTTPException`` whose detail the status-only
handler would discard.

Acting athlete identity is server-derived (AUTH-R3); the ``read`` scope is required
(AUTH-R11). The identity/scope/session/cursor-key dependencies are override seams the app
factory wires.

Requirement IDs: API-R29, API-R31, API-R48, API-R49, API-R50, API-R51, PAGE-R1, PAGE-R2,
PAGE-R3, PAGE-R5, PAGE-R6, PAGE-R8, AUTH-R3, AUTH-R11, AUTH-R15, ANL-R1, ANL-R7, ERR-R1,
ERR-R6, SCHEMA-R8, SCHEMA-R9, DOC-R3.
"""

from __future__ import annotations

import datetime as _dt
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field
from sqlalchemy import asc, desc, select, tuple_
from sqlalchemy.ext.asyncio import AsyncSession

from wattwise_core.analytics.result import is_computed
from wattwise_core.api import activity_helpers as _ah
from wattwise_core.api.activity_schemas import (
    ActivityDetail,
    ActivityLaps,
    ActivityList,
    ActivityStreams,
    ActivityTrack,
    Lap,
    Page,
)
from wattwise_core.api.decimate import rdp_simplify
from wattwise_core.api.deps import RateLimit
from wattwise_core.api.errors import ProblemError
from wattwise_core.api.pagination import clamp_limit, decode_cursor, encode_cursor
from wattwise_core.api.problems import not_found, parameter_invalid, range_reversed

# The stream/map series assembly + the shared coverage / id-parse primitives live in the
# focused :mod:`activities_streams` sibling (QUAL-R9 size split); behavior is unchanged.
from wattwise_core.api.routers.activities_streams import (
    _absent_cov,
    _build_streams,
    _check_max_points,
    _coord,
    _full_cov,
    _now,
    _resolve_channels,
    _stream_rows,
    _uid,
)
from wattwise_core.api.routers.performance import (
    AthleteId,
    Service,
    _Read,
    analytics_service,
    current_athlete_id,
    require_read_scope,
)
from wattwise_core.domain.enums import DeviceClass, StreamChannelName
from wattwise_core.persistence.models import Activity, ActivityLap

router = APIRouter(prefix="/v1/activities", tags=["activities"], dependencies=[RateLimit])

#: The activities sort allow-list (PAGE-R2 / spec §8.7); default ``start_time desc``.
SortKey = Literal["start_time", "duration", "tss"]
SortOrder = Literal["asc", "desc"]
_SORT_COLUMN: dict[str, Any] = {
    "start_time": Activity.start_time,
    "duration": Activity.moving_time_s,
    "tss": Activity.start_time,  # canonical TSS is per-activity; tie-break stays keyset-stable
}


def current_session() -> AsyncSession:
    """Request-scoped DB session seam; the app factory overrides it (fail-closed)."""
    raise ProblemError("internal-error")  # pragma: no cover - replaced by the app factory


def cursor_signing_key() -> str:
    """Provide the cursor HMAC signing key; the app factory overrides it (PAGE-R5).

    The default fails closed so a router mounted without its wiring never issues an
    unsigned cursor. The factory binds the engine ``token_signing_key``; tests inject a
    deterministic key.
    """
    raise ProblemError("internal-error")  # pragma: no cover - replaced by the app factory


Session = Annotated[AsyncSession, Depends(current_session)]
CursorKey = Annotated[str, Depends(cursor_signing_key)]
MaxPoints = Annotated[int, Query()]


# --- §13 list + detail ----------------------------------------------------------


class ActivityFilters(BaseModel):
    frm: Annotated[_dt.date | None, Field(alias="from")] = None
    to: _dt.date | None = None
    sport: str | None = None
    min_duration_s: int | None = None
    has_power: bool | None = None
    device_class: DeviceClass | None = None
    sort: SortKey = "start_time"
    order: SortOrder = "desc"
    # ge=1 rejects limit < 1 up front (PAGE-R3 422); the >200 bound CLAMPS server-side
    # (clamp_limit), so the schema documents it as ``maximum`` (DOC-R3) without rejecting.
    limit: int = Field(default=50, ge=1, json_schema_extra={"maximum": 200})
    cursor: str | None = None


Filters = Annotated[ActivityFilters, Query()]


def _cursor_params(f: ActivityFilters) -> dict[str, str]:
    """The filter/sort fingerprint a cursor is bound to (PAGE-R6); identity-only fields."""
    return {
        "from": f.frm.isoformat() if f.frm else "",
        "to": f.to.isoformat() if f.to else "",
        "sport": f.sport or "",
        "min_duration_s": str(f.min_duration_s) if f.min_duration_s is not None else "",
        "has_power": "" if f.has_power is None else str(f.has_power),
        "device_class": f.device_class.value if f.device_class else "",
        "sort": f.sort,
        "order": f.order,
    }


@router.get("", response_model=ActivityList, operation_id="listActivities", dependencies=[_Read])
async def list_activities(
    session: Session, athlete_id: AthleteId, key: CursorKey, f: Filters
) -> ActivityList:
    """List the athlete's canonical activities, cursor-paginated + typed-sorted (PAGE-R1/R2/R8)."""
    if f.frm is not None and f.to is not None and f.frm > f.to:
        raise range_reversed("from")
    bounded = clamp_limit(int(f.limit))  # PAGE-R3 clamp, never unbounded / offset
    rows = await _query_activities(session, athlete_id, f, key=key, limit=bounded + 1)
    has_more = len(rows) > bounded
    page_rows = rows[:bounded]
    last = page_rows[-1] if (has_more and page_rows) else None
    nxt = (
        encode_cursor(last.start_time, str(last.activity_id), params=_cursor_params(f), key=key)
        if last is not None
        else None
    )
    owner = await _ah.owner_or_not_found(session, athlete_id)  # reference tz for local_date
    return ActivityList(
        data=[_ah.summary(a, _ah.local_date_of(a, owner)) for a in page_rows],
        page=Page(limit=bounded, next_cursor=nxt, has_more=has_more),
    )


async def _query_activities(
    session: AsyncSession, athlete_id: str, f: ActivityFilters, *, key: str, limit: int
) -> list[Activity]:
    """Keyset-paginated activity query, tie-broken on ``activity_id`` (PAGE-R7)."""
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
        c_time, c_id = decode_cursor(f.cursor, params=_cursor_params(f), key=key)
        op = (
            tuple_(Activity.start_time, Activity.activity_id) > (c_time, _uid(c_id))
            if f.order == "asc"
            else tuple_(Activity.start_time, Activity.activity_id) < (c_time, _uid(c_id))
        )
        clauses.append(op)
    direction = asc if f.order == "asc" else desc
    sort_col = _SORT_COLUMN[f.sort]
    # Primary order is the requested sort key (PAGE-R2); the (start_time, activity_id)
    # keyset is the deterministic tie-break the cursor pages on (PAGE-R7).
    order = (direction(sort_col), direction(Activity.start_time), direction(Activity.activity_id))
    stmt = select(Activity).where(*clauses).order_by(*order).limit(limit)
    return list((await session.execute(stmt)).scalars().all())


@router.get(
    "/{activity_id}",
    response_model=ActivityDetail,
    operation_id="getActivity",
    dependencies=[_Read],
)
async def get_activity(
    activity_id: str, session: Session, svc: Service, athlete_id: AthleteId
) -> ActivityDetail:
    """Canonical activity detail with the per-activity load bundle (doc 60 §13)."""
    act = await _load_owned_activity(session, athlete_id, activity_id)
    owner = await _ah.owner_or_not_found(session, athlete_id)
    result = await svc.coggan(activity_id)
    b = result.value if is_computed(result) else None
    return ActivityDetail(
        **_ah.summary(act, _ah.local_date_of(act, owner)).model_dump(),
        max_power_w=_ah.f(act.max_power_w),
        avg_hr_bpm=_ah.f(act.avg_hr_bpm),
        max_hr_bpm=_ah.f(act.max_hr_bpm),
        avg_cadence_rpm=_ah.f(act.avg_cadence_rpm),
        avg_speed_mps=_ah.f(act.avg_speed_mps),
        elevation_gain_m=_ah.f(act.elevation_gain_m),
        total_work_j=_ah.f(act.total_work_j),
        tss=_metric(b.tss) if b else None,
        intensity_factor=_metric(b.if_) if b else None,
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
        raise not_found()
    return act


# --- §13.1 streams --------------------------------------------------------------


@router.get(
    "/{activity_id}/streams",
    response_model=ActivityStreams,
    operation_id="getActivityStreams",
    dependencies=[_Read],
)
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
        raise parameter_invalid("base")
    _check_max_points(max_points)
    requested = _resolve_channels(channels)
    act = await _load_owned_activity(session, athlete_id, activity_id)
    rows = await _stream_rows(session, activity_id)
    return _build_streams(act, requested, rows, base=base, max_points=max_points)


# --- §13.2 map ------------------------------------------------------------------


@router.get(
    "/{activity_id}/map",
    response_model=ActivityTrack,
    operation_id="getActivityMap",
    dependencies=[_Read],
)
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
            activity_id=activity_id,
            points=[],
            original_size=0,
            returned_size=0,
            decimated=False,
            decimation={"algorithm": "none", "max_points": max_points},
            bounds=None,
            coverage=_absent_cov(),
            computed_at=_now(),
        )
    simplified = rdp_simplify(coords, max_points)
    points = [[lat, lng] for lat, lng in simplified]
    lats, lngs = [c[0] for c in coords], [c[1] for c in coords]
    decimated = len(points) < len(coords)
    algo = "rdp" if decimated else "none"
    bbox = {"min_lat": min(lats), "min_lng": min(lngs), "max_lat": max(lats), "max_lng": max(lngs)}
    return ActivityTrack(
        activity_id=activity_id,
        points=points,
        original_size=len(coords),
        returned_size=len(points),
        decimated=decimated,
        decimation={"algorithm": algo, "max_points": max_points},
        bounds=bbox,
        coverage=_full_cov(),
        computed_at=_now(),
    )


# --- §13.3 laps -----------------------------------------------------------------


@router.get(
    "/{activity_id}/laps",
    response_model=ActivityLaps,
    operation_id="getActivityLaps",
    dependencies=[_Read],
)
async def get_laps(activity_id: str, session: Session, athlete_id: AthleteId) -> ActivityLaps:
    """The activity's full, ordered lap table (API-R50); no laps → ``laps: []``."""
    await _load_owned_activity(session, athlete_id, activity_id)
    rows = (
        (
            await session.execute(
                select(ActivityLap)
                .where(ActivityLap.activity_id == _uid(activity_id))
                .order_by(ActivityLap.lap_index.asc())
            )
        )
        .scalars()
        .all()
    )
    return ActivityLaps(activity_id=activity_id, laps=[_lap(r) for r in rows])


def _lap(row: ActivityLap) -> Lap:
    return Lap(
        lap_index=row.lap_index,
        start_offset_s=row.start_offset_s,
        duration_s=row.duration_s,
        distance_m=_ah.f(row.distance_m),
        avg_power_w=_ah.f(row.avg_power_w),
        max_power_w=_ah.f(row.max_power_w),
        avg_hr_bpm=_ah.f(row.avg_hr_bpm),
        max_hr_bpm=_ah.f(row.max_hr_bpm),
        avg_cadence_rpm=_ah.f(row.avg_cadence_rpm),
        avg_speed_mps=_ah.f(row.avg_speed_mps),
        elevation_gain_m=_ah.f(row.elevation_gain_m),
        total_work_j=None,
        coverage=_full_cov(),  # no canonical lap total_work_j (doc 20 §3.3)
    )


# Re-export the shared dependency seams so the app factory can override identity/scope/
# service the SAME way for both routers (one override wires both, FastAPI by identity).
__all__ = [
    "ActivityDetail",
    "ActivityLaps",
    "ActivityList",
    "ActivityStreams",
    "ActivityTrack",
    "analytics_service",
    "current_athlete_id",
    "current_session",
    "cursor_signing_key",
    "require_read_scope",
    "router",
]
