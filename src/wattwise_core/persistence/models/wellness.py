"""Daily wellness and non-activity (wellness) stream sets.

Owning requirements:

* ``daily_wellness`` — exactly ONE row per ``(athlete_id, local_date)`` (GBO-R24),
  a standardized source-neutral SUPERSET. Carries source-reported training-state
  fields that are NOT canonical PMC (GBO-R25); HRV field discipline per GBO-R24c
  (one field = one statistic + one unit; ``hrv_method`` is a pointer).
* ``wellness_stream_set`` — 0..* per day (GBO-R24b); key
  ``(athlete_id, local_date, recording_id)``; its channels live in the shared
  ``stream_channel`` table.
"""

from __future__ import annotations

import datetime as _dt
import uuid

from sqlalchemy import Date, Index, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from wattwise_core.domain.enums import (
    AcwrStatus,
    HrvMethod,
    HrvStatus,
    SampleBasis,
    TrainingStatus,
)
from wattwise_core.persistence.base import Base, TimestampMixin
from wattwise_core.persistence.types import (
    enum_column,
    fk_uuid_column,
    integer_column,
    json_column,
    numeric_column,
    pk_column,
    smallint_column,
    timestamptz_column,
)


class DailyWellness(Base, TimestampMixin):
    """One reconciled wellness row per athlete-local day (GBO-R24).

    Key ``(athlete_id, local_date)``. NEVER carries canonical CTL/ATL/form/TSB
    (GBO-R25 -> ``fitness_state_daily``); the training-state fields here are
    SOURCE-REPORTED only.
    """

    __tablename__ = "daily_wellness"
    __table_args__ = (
        UniqueConstraint("athlete_id", "local_date", name="uq_daily_wellness_athlete_local_date"),
        Index("ix_daily_wellness_athlete_local_date_desc", "athlete_id", "local_date"),
    )

    daily_wellness_id: Mapped[uuid.UUID] = pk_column()
    athlete_id: Mapped[uuid.UUID] = fk_uuid_column("athlete.athlete_id", nullable=False)
    local_date: Mapped[_dt.date] = mapped_column(Date, nullable=False)

    # --- HR / stress / energy / activity ---
    resting_hr_bpm: Mapped[int | None] = smallint_column(nullable=True)
    min_hr_bpm: Mapped[int | None] = smallint_column(nullable=True)
    max_hr_bpm: Mapped[int | None] = smallint_column(nullable=True)
    stress_avg: Mapped[int | None] = smallint_column(nullable=True)
    stress_max: Mapped[int | None] = smallint_column(nullable=True)
    body_battery_high: Mapped[int | None] = smallint_column(nullable=True)
    body_battery_low: Mapped[int | None] = smallint_column(nullable=True)
    steps: Mapped[int | None] = integer_column(nullable=True)
    active_s: Mapped[int | None] = integer_column(nullable=True)
    highly_active_s: Mapped[int | None] = integer_column(nullable=True)
    sedentary_s: Mapped[int | None] = integer_column(nullable=True)
    active_kcal: Mapped[float | None] = numeric_column(nullable=True)
    bmr_kcal: Mapped[float | None] = numeric_column(nullable=True)
    total_kcal: Mapped[float | None] = numeric_column(nullable=True)
    distance_m: Mapped[float | None] = numeric_column(nullable=True)
    intensity_minutes_moderate: Mapped[int | None] = smallint_column(nullable=True)
    intensity_minutes_vigorous: Mapped[int | None] = smallint_column(nullable=True)
    intensity_minutes_goal: Mapped[int | None] = smallint_column(nullable=True)
    floors_ascended: Mapped[int | None] = smallint_column(nullable=True)
    floors_descended: Mapped[int | None] = smallint_column(nullable=True)
    floors_ascended_m: Mapped[float | None] = numeric_column(nullable=True)
    floors_descended_m: Mapped[float | None] = numeric_column(nullable=True)

    # --- respiration / SpO2 ---
    respiration_avg_rpm: Mapped[int | None] = smallint_column(nullable=True)
    respiration_latest_rpm: Mapped[int | None] = smallint_column(nullable=True)
    respiration_lowest_rpm: Mapped[int | None] = smallint_column(nullable=True)
    respiration_highest_rpm: Mapped[int | None] = smallint_column(nullable=True)
    spo2_avg_pct: Mapped[float | None] = numeric_column(nullable=True)
    spo2_latest_pct: Mapped[float | None] = numeric_column(nullable=True)
    spo2_lowest_pct: Mapped[float | None] = numeric_column(nullable=True)

    # --- sleep ---
    sleep_score: Mapped[float | None] = numeric_column(nullable=True)
    sleep_duration_s: Mapped[int | None] = integer_column(nullable=True)
    sleep_start: Mapped[_dt.datetime | None] = timestamptz_column(nullable=True)
    sleep_end: Mapped[_dt.datetime | None] = timestamptz_column(nullable=True)
    sleep_deep_s: Mapped[int | None] = integer_column(nullable=True)
    sleep_light_s: Mapped[int | None] = integer_column(nullable=True)
    sleep_rem_s: Mapped[int | None] = integer_column(nullable=True)
    sleep_awake_s: Mapped[int | None] = integer_column(nullable=True)

    # --- HRV (day/session summaries); one field = one statistic + one unit ---
    hrv_rmssd_ms: Mapped[float | None] = numeric_column(nullable=True)
    hrv_sdnn_ms: Mapped[float | None] = numeric_column(nullable=True)
    hrv_pnn50_pct: Mapped[float | None] = numeric_column(nullable=True)
    hrv_weekly_avg_ms: Mapped[float | None] = numeric_column(nullable=True)
    hrv_baseline_low_ms: Mapped[float | None] = numeric_column(nullable=True)
    hrv_baseline_high_ms: Mapped[float | None] = numeric_column(nullable=True)
    hrv_status: Mapped[HrvStatus | None] = enum_column(HrvStatus, nullable=True)
    hrv_method: Mapped[HrvMethod | None] = enum_column(HrvMethod, nullable=True)

    # --- physiology / profile snapshot ---
    vo2max: Mapped[float | None] = numeric_column(nullable=True)
    fitness_age_years: Mapped[float | None] = numeric_column(nullable=True)
    body_mass_kg: Mapped[float | None] = numeric_column(nullable=True)
    height_cm: Mapped[float | None] = numeric_column(nullable=True)
    ftp_watts: Mapped[float | None] = numeric_column(nullable=True)
    lactate_threshold_hr_bpm: Mapped[int | None] = smallint_column(nullable=True)

    # --- source-reported training-state (NOT canonical PMC, GBO-R25) ---
    training_status: Mapped[TrainingStatus | None] = enum_column(TrainingStatus, nullable=True)
    training_load_balance: Mapped[float | None] = numeric_column(nullable=True)
    acute_load: Mapped[float | None] = numeric_column(nullable=True)
    chronic_load: Mapped[float | None] = numeric_column(nullable=True)
    acwr: Mapped[float | None] = numeric_column(nullable=True)
    acwr_status: Mapped[AcwrStatus | None] = enum_column(AcwrStatus, nullable=True)
    load_aerobic_low: Mapped[float | None] = numeric_column(nullable=True)
    load_aerobic_high: Mapped[float | None] = numeric_column(nullable=True)
    load_anaerobic: Mapped[float | None] = numeric_column(nullable=True)
    endurance_score: Mapped[float | None] = numeric_column(nullable=True)
    readiness_external: Mapped[float | None] = numeric_column(nullable=True)

    coverage: Mapped[dict[str, object]] = json_column(nullable=False, default=dict)
    # The conflict-resolution policy version that produced the resolved values (CONF-R6).
    policy_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    # Per-field resolution record (LIN-R3): candidate pointers + deciding rule; lineage
    # only, never exposed through consumer reads (LIN-R4).
    field_resolution: Mapped[dict[str, object] | None] = json_column(nullable=True)


class WellnessStreamSet(Base, TimestampMixin):
    """Non-activity (wellness) stream set, 0..* per day (GBO-R24b).

    Key ``(athlete_id, local_date, recording_id)``; ``recording_id`` is a per-day
    surrogate ordinal (NOT source-derived). Its channels live in the shared
    ``stream_channel`` table keyed ``(wellness_stream_set_id, channel)``.
    """

    __tablename__ = "wellness_stream_set"
    __table_args__ = (
        UniqueConstraint(
            "athlete_id",
            "local_date",
            "recording_id",
            name="uq_wellness_stream_set_athlete_date_recording",
        ),
        Index(
            "ix_wellness_stream_set_athlete_date_recording",
            "athlete_id",
            "local_date",
            "recording_id",
        ),
    )

    wellness_stream_set_id: Mapped[uuid.UUID] = pk_column()
    athlete_id: Mapped[uuid.UUID] = fk_uuid_column("athlete.athlete_id", nullable=False)
    local_date: Mapped[_dt.date] = mapped_column(Date, nullable=False)
    recording_id: Mapped[int] = smallint_column(nullable=False)
    sample_basis: Mapped[SampleBasis] = enum_column(SampleBasis, nullable=False)
    sample_count: Mapped[int | None] = integer_column(nullable=True)
    t0: Mapped[_dt.datetime] = timestamptz_column(nullable=False)


__all__ = ["DailyWellness", "WellnessStreamSet"]
