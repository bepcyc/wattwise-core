"""Per-athlete source-trust override (PRV-R7 / CONF-R1 configuration data).

The default per-channel trust ordering a source declares lives in
:class:`~wattwise_core.persistence.models.source.SourceDescriptor.trust_profile`
(LIN-R1). PRV-R7 additionally requires that ordering be *overridable per athlete
without code changes* — so an athlete who knows their phone GPS is better than a
platform's down-sampled track can pin that judgement as DATA, not a code branch.

:class:`AthleteSourcePreference` is exactly that data: ONE row binds an athlete +
source + ``channel`` to an effective ``trust_tier`` (the canonical ``Fidelity``
vocabulary, GAP-R2 — no second tier vocabulary, PRV-R7). ``channel = "*"`` is the
whole-source default for that athlete; a specific channel name (e.g. ``power_w``)
overrides only that field/channel. The conflict resolver reads these overrides as the
HIGHEST-precedence layer of the effective-tier resolution (see
:func:`wattwise_core.ingestion.ingest.effective_tier`); an empty table means no
overrides and the resolver falls through to the source descriptor / adapter tier, so
the default behaviour is byte-identical (the opt-in invariant).

CONF-R1: the policy is keyed by ``(source_descriptor_id, channel)`` — NEVER a source
*name*. The natural key ``(athlete_id, source_descriptor_id, channel)`` is the only
place source identity appears here, mirroring the candidate-key discipline (UPS-R1).
"""

from __future__ import annotations

import uuid

from sqlalchemy import String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, validates

from wattwise_core.domain.enums import TRUST_TIER_ORDER, Fidelity
from wattwise_core.persistence.base import Base, TimestampMixin
from wattwise_core.persistence.types import enum_column, fk_uuid_column, pk_column

# The sentinel ``channel`` value meaning "the whole-source default for this athlete"
# (the per-athlete analogue of the descriptor's ``trust_profile["*"]`` base). A
# concrete channel/field name overrides only that channel; "*" applies to all.
WHOLE_SOURCE_CHANNEL = "*"


class NonTierTrustError(ValueError):
    """A trust-tier config value decoded to a non-ranked ``Fidelity`` member (CONF-R2).

    Only the 5 RANKED tiers (``TRUST_TIER_ORDER``) are valid trust tiers; the 3 outcome
    states (``substituted`` / ``absent_true`` / ``absent_failed``) are NOT tiers and must
    be rejected at write time for any trust-tier configuration field.
    """


def ensure_ranked_tier(raw: object, *, field: str) -> Fidelity:
    """Coerce ``raw`` to a RANKED-tier ``Fidelity`` or raise :class:`NonTierTrustError`.

    Accepts only the 5 members of ``TRUST_TIER_ORDER`` (as a ``Fidelity`` or its string
    value). Anything else — an unknown token OR a non-tier ``Fidelity`` (``substituted`` /
    ``absent_*``) — raises a typed ``ValueError`` at construction/assignment, so a
    non-tier trust tier can never be persisted (the app-level complement of the resolver's
    coerce-clamp that makes the per-athlete ingest-DoS path unreachable).
    """
    try:
        fidelity = Fidelity(str(raw))
    except ValueError as exc:
        raise NonTierTrustError(
            f"{field}: {raw!r} is not a valid Fidelity trust tier"
        ) from exc
    if fidelity not in TRUST_TIER_ORDER:
        raise NonTierTrustError(
            f"{field}: {fidelity.value!r} is a non-tier outcome state, not one of the "
            f"ranked trust tiers {[t.value for t in TRUST_TIER_ORDER]}"
        )
    return fidelity


class AthleteSourcePreference(Base, TimestampMixin):
    """One athlete's trust override for a source channel (PRV-R7, configuration data).

    Natural key ``(athlete_id, source_descriptor_id, channel)`` is UNIQUE: at most one
    override per athlete/source/channel. ``channel`` is either a canonical
    field/channel name or :data:`WHOLE_SOURCE_CHANNEL` (``"*"``) for the whole-source
    default. ``trust_tier`` is the ranked-tier subset of the canonical :class:`Fidelity`
    vocabulary (the 5 members of ``TRUST_TIER_ORDER``; text + CHECK, portable across
    SQLite / PostgreSQL / MariaDB), the SAME tiers the resolver ranks on (CONF-R2) — never
    a second competing tier scale, and never one of the 3 non-tier outcome states
    (``substituted`` / ``absent_*``), which :func:`ensure_ranked_tier` rejects at write
    time (a non-tier tier would make a downstream coverage build raise and abort ingest).
    """

    __tablename__ = "athlete_source_preference"
    __table_args__ = (
        UniqueConstraint(
            "athlete_id",
            "source_descriptor_id",
            "channel",
            name="uq_athlete_source_preference_athlete_source_channel",
        ),
    )

    athlete_source_preference_id: Mapped[uuid.UUID] = pk_column()
    athlete_id: Mapped[uuid.UUID] = fk_uuid_column("athlete.athlete_id", nullable=False)
    source_descriptor_id: Mapped[uuid.UUID] = fk_uuid_column(
        "source_descriptor.source_descriptor_id", nullable=False
    )
    # A canonical field/channel name, or ``"*"`` for the whole-source default.
    channel: Mapped[str] = mapped_column(String(64), nullable=False)
    # The effective tier this athlete pins for the channel; canonical Fidelity vocab
    # (text + CHECK) so it ranks identically to a descriptor/adapter tier (CONF-R2).
    trust_tier: Mapped[Fidelity] = enum_column(Fidelity, nullable=False)

    @validates("trust_tier")
    def _validate_trust_tier(self, _key: str, value: object) -> Fidelity:
        """Reject a non-ranked-tier ``trust_tier`` at construction/assignment (CONF-R2)."""
        return ensure_ranked_tier(value, field="trust_tier")


__all__ = [
    "WHOLE_SOURCE_CHANNEL",
    "AthleteSourcePreference",
    "NonTierTrustError",
    "ensure_ranked_tier",
]
