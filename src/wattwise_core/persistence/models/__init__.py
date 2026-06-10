"""Canonical GBO ORM models (doc 20: canonical data model).

Importing this package fully populates ``Base.metadata`` with every canonical
entity, so Alembic autogenerate and ``Base.metadata.create_all`` see the whole
schema. Every model is a SQLAlchemy 2.0 ``Mapped[...]``-typed declarative model
inheriting :class:`~wattwise_core.persistence.base.Base`.

The clusters (and their owning GBO-R* IDs) are:

* :mod:`.athlete`   — ``Athlete``, ``Sport``, ``SubSport``, ``TrainingZoneSet``.
* :mod:`.source`    — ``SourceDescriptor``, ``Connection``, ``SourceCandidate``,
  ``IngestionWatermark`` (SYN-R2), ``IngestionGap`` (ING-GAP-R2).
* :mod:`.athlete_preference` — ``AthleteSourcePreference`` (per-athlete trust
  override, PRV-R7).
* :mod:`.activity`  — ``Activity``, ``ActivityLap``, ``ActivityStreamSet``,
  ``StreamChannel``, ``ActivityFile``.
* :mod:`.wellness`  — ``DailyWellness``, ``WellnessStreamSet``.
* :mod:`.signature` — ``FitnessSignature``.
* :mod:`.derived`   — ``FitnessStateDaily``, ``DerivedActivityMetric``.
* :mod:`.planning`  — ``Workout``, ``Plan``, ``PlanDay``, ``Goal``,
  ``ScheduleAdjustment``.
* :mod:`.notify`    — ``DigestSubscription``, ``NotificationRoute``.
"""

from __future__ import annotations

from wattwise_core.persistence.base import Base
from wattwise_core.persistence.models.activity import (
    Activity,
    ActivityFile,
    ActivityLap,
    ActivityStreamSet,
    StreamChannel,
)
from wattwise_core.persistence.models.athlete import (
    Athlete,
    Sport,
    SubSport,
    TrainingZoneSet,
)
from wattwise_core.persistence.models.athlete_preference import (
    AthleteSourcePreference,
)
from wattwise_core.persistence.models.derived import (
    DerivedActivityMetric,
    FitnessStateDaily,
)
from wattwise_core.persistence.models.notify import (
    DigestSubscription,
    NotificationRoute,
)
from wattwise_core.persistence.models.planning import (
    Goal,
    Plan,
    PlanDay,
    ScheduleAdjustment,
    Workout,
)
from wattwise_core.persistence.models.signature import FitnessSignature
from wattwise_core.persistence.models.source import (
    Connection,
    IngestionGap,
    IngestionWatermark,
    SourceCandidate,
    SourceDescriptor,
)
from wattwise_core.persistence.models.wellness import (
    DailyWellness,
    WellnessStreamSet,
)

__all__ = [
    "Activity",
    "ActivityFile",
    "ActivityLap",
    "ActivityStreamSet",
    "Athlete",
    "AthleteSourcePreference",
    "Base",
    "Connection",
    "DailyWellness",
    "DerivedActivityMetric",
    "DigestSubscription",
    "FitnessSignature",
    "FitnessStateDaily",
    "Goal",
    "IngestionGap",
    "IngestionWatermark",
    "NotificationRoute",
    "Plan",
    "PlanDay",
    "ScheduleAdjustment",
    "SourceCandidate",
    "SourceDescriptor",
    "Sport",
    "StreamChannel",
    "SubSport",
    "TrainingZoneSet",
    "WellnessStreamSet",
    "Workout",
]
