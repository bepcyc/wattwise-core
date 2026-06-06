"""Athlete identity, sport/sub-sport registries, and training zones.

Owning requirements:

* ``athlete`` — exactly ONE row; profile PK + FK anchor (GBO-R13/R13b/R13c).
* ``sport`` / ``sub_sport`` — data-driven registries, NOT closed enums
  (GBO-R16a / GBO-R16a-i / GBO-R16a-ii). Seeded from ``SEED_SPORTS`` via migration.
* ``training_zone_set`` — effective-dated power/HR zones (GBO-R13d), key
  ``(athlete_id, zone_kind, effective_date)``.
"""

from __future__ import annotations

import datetime as _dt
import uuid

from sqlalchemy import Boolean, Date, ForeignKey, Index, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from wattwise_core.domain.enums import Sex, ZoneBasis, ZoneKind
from wattwise_core.persistence.base import Base, TimestampMixin
from wattwise_core.persistence.types import (
    enum_column,
    fk_uuid_column,
    json_column,
    numeric_column,
    pk_column,
    timestamptz_column,
)


class Sport(Base, TimestampMixin):
    """Data-driven sport registry (GBO-R16a / GBO-R16a-i).

    Adding a sport is a data/registration action — zero schema/consumer/agent/API
    change. ``sport_code`` is the stable lowercase-snake-case primary identifier.
    """

    __tablename__ = "sport"

    sport_code: Mapped[str] = mapped_column(String(64), primary_key=True)
    display_name: Mapped[str] = mapped_column(String(128), nullable=False)
    has_mechanical_power: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)


class SubSport(Base, TimestampMixin):
    """Data-driven sub-sport registry (GBO-R16a-ii).

    Each entry references a parent sport; the registry MUST include an ``other``
    fallback member (seeded via migration).
    """

    __tablename__ = "sub_sport"

    sub_sport_code: Mapped[str] = mapped_column(String(64), primary_key=True)
    sport_code: Mapped[str] = mapped_column(
        String(64), ForeignKey("sport.sport_code"), nullable=False, index=True
    )
    display_name: Mapped[str] = mapped_column(String(128), nullable=False)


class Athlete(Base, TimestampMixin):
    """Master identity record — exactly ONE row in OSS (GBO-R13).

    ``athlete_id`` is the profile PK + FK anchor for referential integrity, NEVER an
    isolation key.
    """

    __tablename__ = "athlete"

    athlete_id: Mapped[uuid.UUID] = pk_column()
    display_name: Mapped[str | None] = mapped_column(String(256), nullable=True)
    sex: Mapped[Sex] = enum_column(Sex, nullable=False, default=Sex.UNKNOWN)
    birth_date: Mapped[_dt.date | None] = mapped_column(Date, nullable=True)
    body_mass_kg: Mapped[float | None] = numeric_column(nullable=True)
    primary_locale: Mapped[str | None] = mapped_column(String(35), nullable=True)
    # current primary sport (registry code); may change over lifetime (GBO-R13b);
    # HINT only, not an analytics key. Soft reference into the sport registry.
    current_sport: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("sport.sport_code"), nullable=True, index=True
    )
    reference_timezone: Mapped[str] = mapped_column(String(64), nullable=False, default="UTC")
    reference_timezone_effective_from: Mapped[_dt.datetime | None] = timestamptz_column(
        nullable=True
    )
    # member of doc 40's load_model set; NULL = system default; HINT not identity.
    default_training_load_model: Mapped[str | None] = mapped_column(String(64), nullable=True)


class TrainingZoneSet(Base, TimestampMixin):
    """Effective-dated power/HR training zones (GBO-R13d).

    Key ``(athlete_id, zone_kind, effective_date)``. Intervals per
    ``(athlete_id, zone_kind, sport)`` MUST NOT overlap; at most one open interval.
    No source lineage, no coverage.
    """

    __tablename__ = "training_zone_set"
    __table_args__ = (
        UniqueConstraint(
            "athlete_id",
            "zone_kind",
            "effective_date",
            name="uq_training_zone_set_athlete_kind_effective",
        ),
        Index(
            "ix_training_zone_set_athlete_kind_effective_desc",
            "athlete_id",
            "zone_kind",
            "effective_date",
        ),
    )

    zone_set_id: Mapped[uuid.UUID] = pk_column()
    athlete_id: Mapped[uuid.UUID] = fk_uuid_column("athlete.athlete_id", nullable=False)
    zone_kind: Mapped[ZoneKind] = enum_column(ZoneKind, nullable=False)
    # NULL = all sports. Soft reference into the sport registry.
    sport: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("sport.sport_code"), nullable=True, index=True
    )
    effective_date: Mapped[_dt.date] = mapped_column(Date, nullable=False)
    effective_to: Mapped[_dt.datetime | None] = timestamptz_column(nullable=True)
    basis: Mapped[ZoneBasis] = enum_column(ZoneBasis, nullable=False)
    # ordered non-overlapping contiguous [{zone_index,label,lower,upper}] array.
    boundaries: Mapped[list[dict[str, object]]] = json_column(nullable=False)


__all__ = ["Athlete", "Sport", "SubSport", "TrainingZoneSet"]
