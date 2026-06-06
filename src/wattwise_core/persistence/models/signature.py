"""Effective-dated fitness signatures (thresholds).

Owning requirements:

* ``fitness_signature`` — effective-dated versioned thresholds (GBO-R26/R27/R28);
  key ``(athlete_id, effective_date, signature_type)``. Intervals per
  ``(athlete_id, signature_type)`` MUST NOT overlap; modeled signatures carry
  ``fit_quality`` (analytics MAY fail-closed below a threshold).
"""

from __future__ import annotations

import datetime as _dt
import uuid

from sqlalchemy import Date, ForeignKey, Index, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from wattwise_core.domain.enums import SignatureOrigin
from wattwise_core.persistence.base import Base, TimestampMixin
from wattwise_core.persistence.types import (
    enum_column,
    fk_uuid_column,
    json_column,
    numeric_column,
    pk_column,
    smallint_column,
    timestamptz_column,
)


class FitnessSignature(Base, TimestampMixin):
    """Effective-dated versioned thresholds for an athlete (GBO-R26).

    Key ``(athlete_id, effective_date, signature_type)``; ``signature_type`` is a
    sport-registry code used as the scope discriminator.
    """

    __tablename__ = "fitness_signature"
    __table_args__ = (
        UniqueConstraint(
            "athlete_id",
            "effective_date",
            "signature_type",
            name="uq_fitness_signature_athlete_date_type",
        ),
        Index(
            "ix_fitness_signature_athlete_type_effective_desc",
            "athlete_id",
            "signature_type",
            "effective_date",
        ),
    )

    signature_id: Mapped[uuid.UUID] = pk_column()
    athlete_id: Mapped[uuid.UUID] = fk_uuid_column("athlete.athlete_id", nullable=False)
    # sport-registry code (GBO-R16c); canonical-key discriminator. Soft reference.
    signature_type: Mapped[str] = mapped_column(
        String(64), ForeignKey("sport.sport_code"), nullable=False, index=True
    )
    effective_date: Mapped[_dt.date] = mapped_column(Date, nullable=False)
    effective_to: Mapped[_dt.datetime | None] = timestamptz_column(nullable=True)
    cp_w: Mapped[float | None] = numeric_column(nullable=True)
    w_prime_j: Mapped[float | None] = numeric_column(nullable=True)
    ftp_w: Mapped[float | None] = numeric_column(nullable=True)
    threshold_hr_bpm: Mapped[int | None] = smallint_column(nullable=True)
    max_hr_bpm: Mapped[int | None] = smallint_column(nullable=True)
    resting_hr_bpm: Mapped[int | None] = smallint_column(nullable=True)
    vo2max: Mapped[float | None] = numeric_column(nullable=True)
    origin: Mapped[SignatureOrigin] = enum_column(SignatureOrigin, nullable=False)
    # modeled CP/W' fit metadata: R^2, n points, residuals (GBO-R28).
    fit_quality: Mapped[dict[str, object] | None] = json_column(nullable=True)


__all__ = ["FitnessSignature"]
