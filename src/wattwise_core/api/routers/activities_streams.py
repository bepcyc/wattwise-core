"""Stream/map series helpers for the activities router (API-R48/R49, QUAL-R9 split).

The focused sibling of :mod:`wattwise_core.api.routers.activities` that owns the
column-oriented stream-series assembly (the closed line-chart channel allow-list, channel
selection/validation, decimation-indexed column building and the X-axis base array, API-R48)
and the GPS coordinate / shared id-parse / coverage primitives the endpoints compose
(API-R49, API-R51). Pure functions plus one read-only query helper; behavior is unchanged
from the pre-split module. Degradation stays surfaced, never an error (API-R29): an absent
channel is present-with-``present=false`` + all-``null`` values, never a ``404``.
"""

from __future__ import annotations

import datetime as _dt
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from wattwise_core.api.activity_schemas import ActivityStreams, StreamChannelOut
from wattwise_core.api.chart_schemas import CoverageDescriptor
from wattwise_core.api.decimate import minmax_index
from wattwise_core.api.problems import not_found, parameter_invalid
from wattwise_core.domain.enums import StreamChannelName, StreamSetKind
from wattwise_core.persistence.models import (
    Activity,
    ActivityStreamSet,
    StreamChannel,
)

# Closed line-chart channel allow-list (API-R48): every GBO-R20 channel whose
# sample_basis is time/distance — i.e. all except the event-based rr_intervals_ms.
_STREAM_CHANNELS: tuple[StreamChannelName, ...] = tuple(
    c for c in StreamChannelName if c is not StreamChannelName.RR_INTERVALS_MS
)
_MAX_POINTS_CEILING = 5000

_UNITS: dict[StreamChannelName, str] = {
    StreamChannelName.POWER_W: "watt",
    StreamChannelName.HR_BPM: "bpm",
    StreamChannelName.CADENCE_RPM: "rpm",
    StreamChannelName.SPEED_MPS: "m/s",
    StreamChannelName.ALTITUDE_M: "m",
    StreamChannelName.DISTANCE_M: "m",
    StreamChannelName.TEMP_C: "C",
    StreamChannelName.LEFT_RIGHT_BALANCE: "%",
    StreamChannelName.SMO2: "%",
    StreamChannelName.CORE_TEMP_C: "C",
    StreamChannelName.RESPIRATION_RPM: "rpm",
    StreamChannelName.LATLNG: "deg",
}


def _full_cov() -> CoverageDescriptor:
    """A present, raw-stream coverage descriptor (SCHEMA-R9)."""
    return CoverageDescriptor(present=True, fidelity="raw_stream")


def _absent_cov() -> CoverageDescriptor:
    """A typed-absent coverage descriptor — surfaced degradation, never an error (API-R29)."""
    return CoverageDescriptor(present=False, fidelity="absent_true", gap_fraction=1.0)


def _uid(value: str) -> uuid.UUID:
    """Parse a path id; a malformed id fails closed as ``not-found`` (API-R51/ERR-R1)."""
    try:
        return uuid.UUID(value)
    except (ValueError, AttributeError) as exc:
        raise not_found() from exc


def _now() -> _dt.datetime:
    """The ``computed_at`` stamp for a freshly assembled payload."""
    return _dt.datetime.now(tz=_dt.UTC)


def _resolve_channels(channels: str | None) -> list[StreamChannelName] | None:
    """Validate the requested channel list against the closed allow-list (API-R48)."""
    if channels is None:
        return None  # defaults to every present (non-latlng) channel
    out: list[StreamChannelName] = []
    for token in channels.split(","):
        tok = token.strip()
        match = next((c for c in _STREAM_CHANNELS if c.value == tok), None)
        # ``latlng`` is a valid map channel but its [lat,lng] pairs cannot ride the
        # scalar ``values: list[float|None]`` shape here, so it is rejected on /streams
        # (the map track serves it) rather than silently omitted (API-R48).
        if match is None or match is StreamChannelName.LATLNG:
            raise parameter_invalid("channels")
        out.append(match)
    return out


async def _stream_rows(
    session: AsyncSession, activity_id: str
) -> dict[StreamChannelName, StreamChannel]:
    """The activity's persisted stream channels keyed by name (``{}`` when no set exists)."""
    sset = (
        await session.execute(
            select(ActivityStreamSet).where(ActivityStreamSet.activity_id == _uid(activity_id))
        )
    ).scalar_one_or_none()
    if sset is None:
        return {}
    rows = (
        (
            await session.execute(
                select(StreamChannel).where(
                    StreamChannel.stream_set_id == sset.stream_set_id,
                    StreamChannel.set_kind == StreamSetKind.ACTIVITY,
                )
            )
        )
        .scalars()
        .all()
    )
    return {r.channel: r for r in rows}


def _build_streams(
    act: Activity,
    requested: list[StreamChannelName] | None,
    rows: dict[StreamChannelName, StreamChannel],
    *,
    base: str,
    max_points: int,
) -> ActivityStreams:
    """Assemble the column-oriented, decimation-indexed stream payload (API-R48)."""
    selected = (
        requested
        if requested is not None
        else [c for c in _STREAM_CHANNELS if c in rows and c is not StreamChannelName.LATLNG]
    )
    sample_channels = [
        rows[c].values for c in selected if c in rows and c is not StreamChannelName.LATLNG
    ]
    length = max((len(r.values) for r in rows.values()), default=0)
    idx = minmax_index(length, max_points, sample_channels)
    out_channels = {c.value: _channel_column(c, rows.get(c), idx) for c in selected}
    decimated = len(idx) < length
    algorithm = "minmax_lttb" if decimated else "none"
    return ActivityStreams(
        activity_id=str(act.activity_id),
        base=base,
        base_values=_base_values(base, rows, idx),
        original_size=length,
        returned_size=len(idx),
        decimated=decimated,
        decimation={"algorithm": algorithm, "max_points": max_points},
        channels=out_channels,
        computed_at=_now(),
    )


def _base_values(
    base: str, rows: dict[StreamChannelName, StreamChannel], idx: list[int]
) -> list[float]:
    """The X-axis array the channels align to (API-R48).

    ``base=distance`` → the cumulative ``distance_m`` channel sampled at ``idx``;
    ``base=time`` → seconds from ``start_time`` (the 1 Hz sample index in seconds, the
    canonical time base). Computed from the actual channel, never the bare sample index.
    """
    if base == "distance":
        dist = rows.get(StreamChannelName.DISTANCE_M)
        if dist is not None:
            return [_distance_at(dist.values, i) for i in idx]
    return [float(i) for i in idx]


def _distance_at(values: list[object], i: int) -> float:
    """Cumulative distance at sample ``i`` (the channel value, else the index fallback)."""
    if i < len(values):
        v = values[i]
        if isinstance(v, int | float):
            return float(v)
    return float(i)


def _channel_column(
    channel: StreamChannelName, row: StreamChannel | None, idx: list[int]
) -> StreamChannelOut:
    """One output channel column; an absent channel is typed-absent nulls (API-R29/R48)."""
    unit = _UNITS.get(channel, "")
    if row is None:
        return StreamChannelOut(values=[None] * len(idx), unit=unit, coverage=_absent_cov())
    vals = [_sample(row.values[i]) if i < len(row.values) else None for i in idx]
    return StreamChannelOut(values=vals, unit=unit, coverage=_full_cov())


def _sample(value: object) -> float | None:
    """Coerce one persisted sample to a float, else a typed ``null`` (API-R29)."""
    return float(value) if isinstance(value, int | float) else None


def _check_max_points(max_points: int) -> None:
    """Bound ``max_points`` to the closed ceiling; out of range → ``validation-error``."""
    if not isinstance(max_points, int) or not 1 <= max_points <= _MAX_POINTS_CEILING:
        raise parameter_invalid("max_points")


def _coord(value: object) -> tuple[float, float] | None:
    """Parse one persisted ``latlng`` sample into a ``(lat, lng)`` pair, else ``None``."""
    if isinstance(value, (list, tuple)) and len(value) == 2:
        lat, lng = value
        if isinstance(lat, (int, float)) and isinstance(lng, (int, float)):
            return float(lat), float(lng)
    return None
