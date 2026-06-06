"""Wire shapes for the ``/v1/activities*`` surface (SCHEMA-R8/R9, API-R48/R49/R50).

Extracted from the activities router so the route logic stays within the module-size
ceiling (QUAL-R9). Every field reads a typed canonical column; none is source-shaped or
carries a provider name (AUTH-R15); fidelity is the SCHEMA-R9 ``coverage`` only.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any

from pydantic import BaseModel

from wattwise_core.api.chart_schemas import CoverageDescriptor


class ActivitySummary(BaseModel):
    """One activity's canonical list-row summary (§13)."""

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
    """The cursor-pagination page envelope (PAGE-R1): clamp + opaque next cursor."""

    limit: int
    next_cursor: str | None = None
    has_more: bool


class ActivityList(BaseModel):
    """The paginated activity list response (PAGE-R1/R8)."""

    data: list[ActivitySummary]
    page: Page


class ActivityDetail(ActivitySummary):
    """Canonical activity detail with the per-activity load bundle (§13)."""

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
    """One column-oriented per-sample stream channel (API-R48): gaps survive as null."""

    values: list[float | None]
    unit: str
    coverage: CoverageDescriptor


class ActivityStreams(BaseModel):
    """The index-aligned per-sample stream bundle (API-R48).

    ``base_values`` is the X-axis array the channels align to: seconds from
    ``start_time`` for ``base=time`` or cumulative metres for ``base=distance``.
    """

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
    """The RDP-simplified GPS map polyline (API-R49); no GPS → typed empty map."""

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
    """One lap row (API-R50): lap-scoped canonical scalars + coverage."""

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
    """The full, ordered lap table (API-R50); no laps → ``laps: []``."""

    activity_id: str
    laps: list[Lap]


__all__ = [
    "ActivityDetail",
    "ActivityLaps",
    "ActivityList",
    "ActivityStreams",
    "ActivitySummary",
    "ActivityTrack",
    "Lap",
    "Page",
    "StreamChannelOut",
]
