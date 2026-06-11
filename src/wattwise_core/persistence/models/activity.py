"""Activity cluster: sessions, laps, stream sets, channels, and raw files.

Owning requirements:

* ``activity`` — one continuous training session in canonical form (GBO-R14/R15);
  resolved ``activity_id``, source-independent; derived sports-science metrics are
  NOT authoritative columns here (GBO-R16 -> see ``models.derived``).
* ``activity_lap`` — child rows (GBO-R17); UNIQUE ``(activity_id, lap_index)``.
* ``activity_stream_set`` — 1:1 sampling contract (GBO-R19).
* ``stream_channel`` — ONE table serving BOTH activity and wellness stream sets
  (GBO-R20/R20b/R21/R22); UNIQUE ``(stream_set_id, channel)``; the parent is
  identified by a ``set_kind`` discriminator + an un-FK'd ``stream_set_id`` (two
  possible parents).
* ``activity_file`` — tier-1 object-store reference (RAW-R1/R3); dedup UNIQUE
  ``(activity_id, source_descriptor_id, content_hash)``.
"""

from __future__ import annotations

import datetime as _dt
import uuid

from sqlalchemy import Boolean, Date, ForeignKey, Index, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from wattwise_core.domain.enums import (
    ActivityFileFormat,
    DeviceClass,
    SampleBasis,
    StreamChannelName,
    StreamSetKind,
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


class Activity(Base, TimestampMixin):
    """One continuous training session in canonical form (GBO-R14).

    ``activity_id`` is the resolved canonical identity (§4.3), NEVER a per-source id.
    Summary scalars are reproducible from streams when present (GBO-R15). Source-
    reported fields (``training_load_source``, ``vo2max_estimate`` ...) are typed
    SUMMARIES only — analytics MUST NOT read them as canonical (GBO-R16/R25).
    """

    __tablename__ = "activity"
    __table_args__ = (
        Index("ix_activity_athlete_start_time", "athlete_id", "start_time"),
        Index("ix_activity_athlete_sport_start_time", "athlete_id", "sport", "start_time"),
        # The day-bucket query path filters by the athlete-LOCAL calendar day (GBO-R35).
        Index("ix_activity_athlete_local_date", "athlete_id", "local_date"),
    )

    activity_id: Mapped[uuid.UUID] = pk_column()
    athlete_id: Mapped[uuid.UUID] = fk_uuid_column("athlete.athlete_id", nullable=False)
    start_time: Mapped[_dt.datetime] = timestamptz_column(nullable=False)
    # derived display only — the athlete-LOCAL wall-clock of start_time (GBO-R13, §3.8).
    start_time_local: Mapped[_dt.datetime | None] = timestamptz_column(nullable=True)
    # the athlete-LOCAL calendar date of start_time (§3.8); the reproducible GBO-R35
    # day-attribution bucket, recomputable from start_time + the as-of reference tz
    # (GBO-R34). Mirrors daily_wellness.local_date; NOT a UTC date. Nullable so a row
    # ingested before a tz was configured surfaces a typed absence rather than a fake date.
    local_date: Mapped[_dt.date | None] = mapped_column(Date, nullable=True)
    elapsed_time_s: Mapped[int | None] = integer_column(nullable=True)
    moving_time_s: Mapped[int | None] = integer_column(nullable=True)
    # registry code (GBO-R16a); NOT NULL. Soft reference into the sport registry.
    sport: Mapped[str] = mapped_column(
        String(64), ForeignKey("sport.sport_code"), nullable=False, index=True
    )
    sub_sport: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("sub_sport.sub_sport_code"), nullable=True, index=True
    )
    is_indoor: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    distance_m: Mapped[float | None] = numeric_column(nullable=True)
    total_work_j: Mapped[float | None] = numeric_column(nullable=True)
    energy_kj: Mapped[float | None] = numeric_column(nullable=True)
    avg_power_w: Mapped[float | None] = numeric_column(nullable=True)
    max_power_w: Mapped[float | None] = numeric_column(nullable=True)
    avg_hr_bpm: Mapped[int | None] = smallint_column(nullable=True)
    max_hr_bpm: Mapped[int | None] = smallint_column(nullable=True)
    avg_cadence_rpm: Mapped[int | None] = smallint_column(nullable=True)
    avg_speed_mps: Mapped[float | None] = numeric_column(nullable=True)
    elevation_gain_m: Mapped[float | None] = numeric_column(nullable=True)
    avg_temp_c: Mapped[float | None] = numeric_column(nullable=True)
    # Athlete-reported session exertion on the CR-10 scale (0..10, SRPE-R1): the ONLY
    # internal-load input that exists for power-less, HR-less sessions (strength, most
    # swims). Captured from the source when it carries one (FIT perceived_exertion,
    # intervals.icu icu_rpe); NULL is a typed absence — never imputed to a default.
    perceived_exertion: Mapped[float | None] = numeric_column(nullable=True)
    # Athlete-reported session feel, the intervals.icu 1..5 ordinal (1 = strong,
    # 5 = weak). A subjective summary token, not a load input; NULL when unreported.
    feel: Mapped[int | None] = smallint_column(nullable=True)
    training_effect_aerobic: Mapped[float | None] = numeric_column(nullable=True)
    anaerobic_effect: Mapped[float | None] = numeric_column(nullable=True)
    vo2max_estimate: Mapped[float | None] = numeric_column(nullable=True)
    training_load_source: Mapped[float | None] = numeric_column(nullable=True)
    device_class: Mapped[DeviceClass | None] = enum_column(DeviceClass, nullable=True)
    has_power: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    has_hr: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    has_gps: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    has_cadence: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    coverage: Mapped[dict[str, object]] = json_column(nullable=False, default=dict)
    # The version of the conflict-resolution policy that produced the resolved values
    # (CONF-R6): recorded on every canonical write so a re-resolution under a changed
    # trust profile is auditable. NULL only for rows written before the column existed.
    policy_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    # Per-field resolution record (LIN-R3): winner/considered candidate POINTERS into
    # the tier-2 ``source_candidate`` store + the CONF-R2 rule that decided, keyed by
    # canonical field name. Lineage-only — NEVER exposed through consumer reads (LIN-R4).
    field_resolution: Mapped[dict[str, object] | None] = json_column(nullable=True)


class ActivityLap(Base, TimestampMixin):
    """Child lap row of an activity (GBO-R17).

    ``lap_index`` is 0-based and contiguous; UNIQUE ``(activity_id, lap_index)``.
    Carries the same canonical summary fields scoped to the lap.
    """

    __tablename__ = "activity_lap"
    __table_args__ = (
        UniqueConstraint("activity_id", "lap_index", name="uq_activity_lap_activity_lap_index"),
    )

    activity_lap_id: Mapped[uuid.UUID] = pk_column()
    activity_id: Mapped[uuid.UUID] = fk_uuid_column("activity.activity_id", nullable=False)
    lap_index: Mapped[int] = integer_column(nullable=False)
    start_offset_s: Mapped[int | None] = integer_column(nullable=True)
    duration_s: Mapped[int | None] = integer_column(nullable=True)
    distance_m: Mapped[float | None] = numeric_column(nullable=True)
    avg_power_w: Mapped[float | None] = numeric_column(nullable=True)
    max_power_w: Mapped[float | None] = numeric_column(nullable=True)
    avg_hr_bpm: Mapped[int | None] = smallint_column(nullable=True)
    max_hr_bpm: Mapped[int | None] = smallint_column(nullable=True)
    avg_cadence_rpm: Mapped[int | None] = smallint_column(nullable=True)
    avg_speed_mps: Mapped[float | None] = numeric_column(nullable=True)
    elevation_gain_m: Mapped[float | None] = numeric_column(nullable=True)
    coverage: Mapped[dict[str, object]] = json_column(nullable=False, default=dict)


class ActivityStreamSet(Base, TimestampMixin):
    """Per-activity sampling contract, 1:1 with activity (GBO-R19).

    Channels inherit the set ``sample_basis`` when their own is NULL (GBO-R21);
    ``t0`` equals ``activity.start_time``.
    """

    __tablename__ = "activity_stream_set"
    __table_args__ = (UniqueConstraint("activity_id", name="uq_activity_stream_set_activity"),)

    stream_set_id: Mapped[uuid.UUID] = pk_column()
    activity_id: Mapped[uuid.UUID] = fk_uuid_column("activity.activity_id", nullable=False)
    sample_basis: Mapped[SampleBasis] = enum_column(SampleBasis, nullable=False)
    sample_rate_hz: Mapped[float | None] = numeric_column(nullable=True)
    sample_count: Mapped[int | None] = integer_column(nullable=True)
    t0: Mapped[_dt.datetime] = timestamptz_column(nullable=False)


class StreamChannel(Base, TimestampMixin):
    """ONE canonical channel, serving BOTH activity and wellness sets (GBO-R20).

    ``stream_set_id`` references one of two possible parents (an
    ``activity_stream_set`` or a ``wellness_stream_set``); it carries NO hard FK and
    the parent is disambiguated by ``set_kind``. Key ``(stream_set_id, channel)``.
    Per-channel ``sample_basis`` overrides the set default when set (GBO-R20b/R21);
    ``values`` is a compressed typed array stored as a portable JSON list with typed
    missing samples preserved (GBO-R22).
    """

    __tablename__ = "stream_channel"
    __table_args__ = (
        UniqueConstraint("stream_set_id", "channel", name="uq_stream_channel_set_channel"),
        Index("ix_stream_channel_stream_set_id", "stream_set_id"),
    )

    stream_channel_id: Mapped[uuid.UUID] = pk_column()
    # un-FK'd: two possible parents (activity vs wellness stream set), see set_kind.
    stream_set_id: Mapped[uuid.UUID] = mapped_column(nullable=False)
    set_kind: Mapped[StreamSetKind] = enum_column(StreamSetKind, nullable=False)
    channel: Mapped[StreamChannelName] = enum_column(StreamChannelName, nullable=False)
    sample_basis: Mapped[SampleBasis | None] = enum_column(SampleBasis, nullable=True)
    # compressed typed array (typed missing preserved, GBO-R22) as a portable JSON list.
    values: Mapped[list[object]] = json_column(nullable=False, default=list)
    coverage: Mapped[dict[str, object]] = json_column(nullable=False, default=dict)


class ActivityFile(Base, TimestampMixin):
    """Tier-1 object-store reference to a verbatim original file (RAW-R1/R3).

    The relational store holds ONLY an opaque ``object_ref`` — never the bytes.
    Dedup UNIQUE ``(activity_id, source_descriptor_id, content_hash)``.
    """

    __tablename__ = "activity_file"
    __table_args__ = (
        UniqueConstraint(
            "activity_id",
            "source_descriptor_id",
            "content_hash",
            name="uq_activity_file_activity_source_hash",
        ),
        # NOTE: the `activity_id` single-column index (IDX-R3) is already provided by
        # the FK column (fk_uuid_column(index=True), IDX-R1) as ix_activity_file_activity_id.
    )

    activity_file_id: Mapped[uuid.UUID] = pk_column()
    activity_id: Mapped[uuid.UUID] = fk_uuid_column("activity.activity_id", nullable=False)
    athlete_id: Mapped[uuid.UUID] = fk_uuid_column("athlete.athlete_id", nullable=False)
    object_ref: Mapped[str] = mapped_column(String(1024), nullable=False)
    format: Mapped[ActivityFileFormat] = enum_column(ActivityFileFormat, nullable=False)
    byte_size: Mapped[int | None] = integer_column(nullable=True)
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    source_descriptor_id: Mapped[uuid.UUID] = fk_uuid_column(
        "source_descriptor.source_descriptor_id", nullable=False
    )
    fetched_at: Mapped[_dt.datetime | None] = timestamptz_column(nullable=True)


__all__ = [
    "Activity",
    "ActivityFile",
    "ActivityLap",
    "ActivityStreamSet",
    "StreamChannel",
]
