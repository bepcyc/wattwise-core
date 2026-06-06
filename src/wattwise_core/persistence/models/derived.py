"""Derived analytics store (separated from authoritative canonical rows).

Owning requirements:

* ``fitness_state_daily`` — the derived PMC store (GBO-R16/R25); key
  ``(athlete_id, local_date, load_model)``. Canonical CTL/ATL/form/TSB live HERE,
  NOT on ``daily_wellness``, and carry their own lineage + engine version.
* ``derived_activity_metric`` — per-activity derived sports-science metrics
  (NP/IF/TSS/decoupling ...) (GBO-R16); key ``(activity_id, load_model)``. These
  MUST NOT be authoritative columns on ``activity``.

``load_model`` is a free-text code validated against doc 40's ``load_model`` set
(this doc models only the field; the set + selection semantics are owned by doc 40).
"""

from __future__ import annotations

import datetime as _dt
import uuid

from sqlalchemy import Boolean, Date, Index, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from wattwise_core.persistence.base import Base, TimestampMixin
from wattwise_core.persistence.types import (
    fk_uuid_column,
    json_column,
    numeric_column,
    pk_column,
    timestamptz_column,
)


class FitnessStateDaily(Base, TimestampMixin):
    """Derived per-day PMC state (GBO-R16/R25).

    Key ``(athlete_id, local_date, load_model)``. Canonical CTL/ATL/form(TSB) with a
    ``load_model`` discriminator, a ``provisional`` flag (doc 40 rest-day semantics),
    an ``engine_version``, and a ``lineage`` JSON of canonical input-snapshot ids.
    """

    __tablename__ = "fitness_state_daily"
    __table_args__ = (
        UniqueConstraint(
            "athlete_id",
            "local_date",
            "load_model",
            name="uq_fitness_state_daily_athlete_date_model",
        ),
        Index(
            "ix_fitness_state_daily_athlete_date_model",
            "athlete_id",
            "local_date",
            "load_model",
        ),
    )

    fitness_state_daily_id: Mapped[uuid.UUID] = pk_column()
    athlete_id: Mapped[uuid.UUID] = fk_uuid_column("athlete.athlete_id", nullable=False)
    local_date: Mapped[_dt.date] = mapped_column(Date, nullable=False)
    load_model: Mapped[str] = mapped_column(String(64), nullable=False)
    ctl: Mapped[float | None] = numeric_column(nullable=True)
    atl: Mapped[float | None] = numeric_column(nullable=True)
    tsb: Mapped[float | None] = numeric_column(nullable=True)
    provisional: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    engine_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    lineage: Mapped[dict[str, object]] = json_column(nullable=False, default=dict)


class DerivedActivityMetric(Base, TimestampMixin):
    """Per-activity derived sports-science metrics (GBO-R16).

    Key ``(activity_id, load_model)``. NP/IF/TSS/decoupling and friends are NULLABLE
    numerics; carries an ``engine_version``, ``computed_at``, and a ``lineage`` JSON.
    Separated from ``activity`` so derived values are NEVER authoritative columns.
    """

    __tablename__ = "derived_activity_metric"
    __table_args__ = (
        UniqueConstraint(
            "activity_id",
            "load_model",
            name="uq_derived_activity_metric_activity_model",
        ),
        Index("ix_derived_activity_metric_activity_model", "activity_id", "load_model"),
    )

    derived_activity_metric_id: Mapped[uuid.UUID] = pk_column()
    activity_id: Mapped[uuid.UUID] = fk_uuid_column("activity.activity_id", nullable=False)
    load_model: Mapped[str] = mapped_column(String(64), nullable=False)
    np_w: Mapped[float | None] = numeric_column(nullable=True)
    if_: Mapped[float | None] = numeric_column(nullable=True)
    tss: Mapped[float | None] = numeric_column(nullable=True)
    decoupling_pct: Mapped[float | None] = numeric_column(nullable=True)
    variability_index: Mapped[float | None] = numeric_column(nullable=True)
    efficiency_factor: Mapped[float | None] = numeric_column(nullable=True)
    work_above_cp_j: Mapped[float | None] = numeric_column(nullable=True)
    engine_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    computed_at: Mapped[_dt.datetime | None] = timestamptz_column(nullable=True)
    lineage: Mapped[dict[str, object]] = json_column(nullable=False, default=dict)


__all__ = ["DerivedActivityMetric", "FitnessStateDaily"]
