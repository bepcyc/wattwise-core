"""Intervals.icu source-shaped payload models (ASBOs) — the CLI-R2 ingress boundary.

The validated source-shaped objects the typed :class:`~wattwise_core.ingestion.adapters.
intervals_icu.IntervalsIcuClient` decodes a response into BEFORE the pure adapter map runs
(CLI-R2: a drifted payload fails closed as a typed error at this boundary, never partially
coerced into a GBO). These are pure schema (no I/O, no logic), split out of the adapter
module so the I/O client + the pure mapper each stay focused (QUAL-R9). The fields carry
the SOURCE vocabulary verbatim; the adapter's ``map`` translates them to canonical fields,
SI units, and canonical enums (MAP-R2/R3/R4).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class IntervalsActivityAsbo(BaseModel):
    """Validated source-shaped activity payload (CLI-R2; fail-closed at the boundary)."""

    model_config = ConfigDict(extra="allow", frozen=True)

    id: str
    type: str | None = None
    sub_type: str | None = None
    start_date: str | None = None
    start_date_local: str | None = None
    name: str | None = None
    description: str | None = None
    device_name: str | None = None
    source: str | None = None
    distance: float | None = None
    moving_time: int | None = None
    elapsed_time: int | None = None
    icu_recording_time: int | None = None
    total_elevation_gain: float | None = None
    icu_joules: float | None = None
    icu_average_watts: float | None = None
    icu_weighted_avg_watts: float | None = None
    p_max: float | None = None
    average_heartrate: float | None = None
    max_heartrate: float | None = None
    average_cadence: float | None = None
    average_speed: float | None = None
    max_speed: float | None = None
    average_temp: float | None = None
    calories: float | None = None
    device_watts: bool | None = None
    power_meter: bool | None = None
    trainer: bool | None = None
    has_heartrate: bool | None = None
    icu_lap_count: int | None = None
    icu_rpe: float | None = None
    feel: int | None = None


class IntervalsStreamAsbo(BaseModel):
    """One per-sample stream channel as Intervals returns it (CLI-R2)."""

    model_config = ConfigDict(extra="allow", frozen=True)

    type: str
    data: list[Any] = Field(default_factory=list)


class IntervalsWellnessAsbo(BaseModel):
    """Validated source-shaped daily-wellness payload (CLI-R2)."""

    model_config = ConfigDict(extra="allow", frozen=True)

    id: str  # the wellness record id IS the local ISO date (e.g. "2026-05-01")
    restingHR: int | None = None
    hrv: float | None = None  # rmssd
    hrvSDNN: float | None = None
    sleepScore: float | None = None
    sleepSecs: int | None = None
    steps: int | None = None
    weight: float | None = None
    readiness: float | None = None
    spO2: float | None = None
    respiration: float | None = None
    vo2max: float | None = None


class ActivityWithStreams(BaseModel):
    """A fetched activity plus its decoded streams — the unit the map consumes."""

    model_config = ConfigDict(frozen=True)

    activity: IntervalsActivityAsbo
    streams: list[IntervalsStreamAsbo] = Field(default_factory=list)


__all__ = [
    "ActivityWithStreams",
    "IntervalsActivityAsbo",
    "IntervalsStreamAsbo",
    "IntervalsWellnessAsbo",
]
