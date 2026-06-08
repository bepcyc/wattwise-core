"""Configurable effective-trust resolution (CONF-R1, LIN-R1, PRV-R6/R7).

The conflict resolver (:mod:`wattwise_core.ingestion.dedup`) ranks candidates PRIMARILY
by a ``trust_tier`` (CONF-R2 step 1). Historically that tier was just the adapter's
observed fidelity carried on each candidate. This module turns it into CONFIGURATION
DATA layered on top of that observed fidelity, computing an EFFECTIVE tier per
``(athlete, candidate, field/channel)`` from, in strict first-hit-wins order:

1. a per-athlete override for ``(source_descriptor_id, field/channel)`` (PRV-R7); else
2. a per-athlete override for ``(source_descriptor_id, "*")`` (whole-source default for
   that athlete, PRV-R7); else
3. ``SourceDescriptor.trust_profile[field/channel]`` (the source's declared per-channel
   base, LIN-R1); else
4. ``SourceDescriptor.trust_profile["*"]`` / ``SourceDescriptor.default_fidelity`` (the
   source's whole-source declared base, LIN-R1); else
5. the candidate's ACTUAL adapter-assigned ``trust_tier`` (the real observed fidelity —
   the current, pre-config behaviour).

PRV-R6 INTERACTION (intentional, opt-in): with NO configuration (empty descriptor
``trust_profile`` AND no athlete override — the seeded state), every layer 1-4 misses
and the effective tier IS the candidate's adapter-assigned tier. So the actual
higher-fidelity observation wins by default (PRV-R6 preserved) and existing behaviour is
byte-identical. A configured profile / per-athlete override is an EXPLICIT opt-in
re-rank: configuring it is a deliberate statement that, for this athlete/source/channel,
the declared tier should decide instead of the raw observed fidelity. CONF-R1: the
policy is keyed by ``(source_descriptor_id, field/channel)`` — NEVER by a source NAME.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from wattwise_core.domain.enums import TRUST_TIER_ORDER, Fidelity
from wattwise_core.persistence.models import SourceCandidate, SourceDescriptor
from wattwise_core.persistence.models.athlete_preference import (
    WHOLE_SOURCE_CHANNEL,
    AthleteSourcePreference,
)

# The per-candidate adapter tier lives under this key in ``SourceCandidate.trust_profile``
# (written by the ingest map step). Layer-5 fallback when no config re-ranks the tier.
_ADAPTER_TIER_KEY = "tier"


@dataclass(frozen=True, slots=True)
class TrustPolicy:
    """A resolved, in-memory trust policy for ONE ingest resolution (CONF-R1 config).

    Pure data + a pure :meth:`tier` — holds no session and touches no DB, so the
    resolver stays DB-free. Built by :func:`load_trust_policy` from the descriptors and
    athlete overrides for the candidates in play; an EMPTY policy (no profiles, no
    overrides) makes :meth:`tier` return the candidate's adapter tier unchanged, so the
    default behaviour is byte-identical (the opt-in invariant).

    * ``profiles`` maps ``source_descriptor_id -> (per_channel_profile, default_fidelity)``
      where ``per_channel_profile`` is the descriptor's ``trust_profile`` dict (channel ->
      tier-token, possibly with a ``"*"`` whole-source entry).
    * ``overrides`` maps ``(source_descriptor_id, channel) -> Fidelity`` (the per-athlete
      override rows for THIS athlete; ``channel`` is a field/channel name or ``"*"``).
    """

    profiles: dict[str, tuple[dict[str, object], str | None]]
    overrides: dict[tuple[str, str], Fidelity]

    def tier(self, candidate: SourceCandidate, channel: str) -> Fidelity:
        """The EFFECTIVE tier for ``candidate`` on ``channel`` (the 5-layer order above).

        ``channel`` is a canonical field/channel name, or ``"*"`` to resolve the
        whole-source effective tier (used for record-level surfaces such as streams).
        First hit wins; the final fallback is the candidate's adapter-assigned tier so an
        empty policy is a no-op (PRV-R6 preserved by default).
        """
        descriptor_id = str(candidate.source_descriptor_id)
        # 1 + 2: per-athlete override (specific channel, then whole-source "*"). A stored
        # override that is NOT one of the 5 ranked tiers (a non-tier ``Fidelity`` such as
        # ``absent_true``) is ignored exactly like a missing override and falls through to
        # the next layer — a non-tier override can never become an effective tier (CONF-R2;
        # closes the per-athlete ingest-DoS path regardless of storage gating).
        override = _clamp_tier(self.overrides.get((descriptor_id, channel)))
        if override is None and channel != WHOLE_SOURCE_CHANNEL:
            override = _clamp_tier(self.overrides.get((descriptor_id, WHOLE_SOURCE_CHANNEL)))
        if override is not None:
            return override
        # 3 + 4: descriptor-declared base (specific channel, then "*"/default_fidelity).
        profile, default_fidelity = self.profiles.get(descriptor_id, ({}, None))
        declared = _profile_tier(profile, channel) or _profile_tier(
            profile, WHOLE_SOURCE_CHANNEL
        )
        if declared is None and default_fidelity is not None:
            declared = _coerce_fidelity(default_fidelity)
        if declared is not None:
            return declared
        # 5: the candidate's ACTUAL adapter-assigned tier (current behaviour, PRV-R6).
        return _adapter_tier(candidate)


def _clamp_tier(fidelity: Fidelity | None) -> Fidelity | None:
    """Pass through a RANKED tier; map a non-tier (or ``None``) to ``None`` (CONF-R2).

    Used to gate the per-athlete override layers: a non-tier ``Fidelity`` (``substituted``
    / ``absent_*``) must NEVER become an effective tier, so it is dropped and resolution
    falls through to the next layer.
    """
    if fidelity is None:
        return None
    return fidelity if fidelity in TRUST_TIER_ORDER else None


def _adapter_tier(candidate: SourceCandidate) -> Fidelity:
    """The candidate's adapter-assigned observed fidelity (layer-5 fallback)."""
    raw = candidate.trust_profile.get(_ADAPTER_TIER_KEY, Fidelity.PLATFORM_COMPUTED.value)
    return Fidelity(str(raw))


def _profile_tier(profile: dict[str, object], channel: str) -> Fidelity | None:
    """Read + coerce one channel's declared tier from a descriptor ``trust_profile``."""
    raw = profile.get(channel)
    return _coerce_fidelity(raw) if raw is not None else None


def _coerce_fidelity(raw: object) -> Fidelity | None:
    """Coerce a stored tier token to a RANKED ``Fidelity`` tier; else ignore it (CONF-R2).

    Configuration data is tolerant: a malformed token never crashes resolution, it just
    falls through to the next layer (ultimately the adapter tier), so a bad config can
    never silently corrupt the canonical record.

    Only the 5 RANKED trust tiers (``TRUST_TIER_ORDER``) are valid effective tiers
    (CONF-R2). The 3 non-tier ``Fidelity`` members — ``substituted`` / ``absent_true`` /
    ``absent_failed`` — are outcome states, NOT tiers: a stored token that decodes to one
    of them is treated EXACTLY like an unknown/garbage token (returns ``None`` → falls
    through to the next config layer). This guarantees a non-tier can never become an
    effective tier, so a downstream ``coverage_for(present=True, <non-tier>, ...)`` — which
    would raise and abort the whole ingest batch — is unreachable from configuration,
    regardless of any storage-level gating (a self-inflicted per-athlete ingest DoS).
    """
    try:
        fidelity = Fidelity(str(raw))
    except ValueError:
        return None
    return fidelity if fidelity in TRUST_TIER_ORDER else None


async def load_trust_policy(
    session: AsyncSession,
    athlete_id: uuid.UUID,
    candidates: Iterable[SourceCandidate],
) -> TrustPolicy:
    """Load the descriptor profiles + this athlete's overrides for the candidates in play.

    The DB read seam (the ingest layer owns the DB; the resolver stays pure). Scopes both
    reads to ONLY the ``source_descriptor_id``s actually contributing, so resolution cost
    is bounded by the candidate set, not the table. An empty override table (the default)
    yields an empty ``overrides`` map ⇒ :meth:`TrustPolicy.tier` returns the adapter tier
    ⇒ default behaviour is byte-identical (the opt-in invariant).
    """
    descriptor_ids = {c.source_descriptor_id for c in candidates}
    if not descriptor_ids:
        return TrustPolicy(profiles={}, overrides={})
    profiles: dict[str, tuple[dict[str, object], str | None]] = {}
    desc_stmt = select(
        SourceDescriptor.source_descriptor_id,
        SourceDescriptor.trust_profile,
        SourceDescriptor.default_fidelity,
    ).where(SourceDescriptor.source_descriptor_id.in_(descriptor_ids))
    for did, profile, default_fidelity in (await session.execute(desc_stmt)).all():
        profiles[str(did)] = (dict(profile or {}), default_fidelity)
    overrides: dict[tuple[str, str], Fidelity] = {}
    pref_stmt = select(AthleteSourcePreference).where(
        AthleteSourcePreference.athlete_id == athlete_id,
        AthleteSourcePreference.source_descriptor_id.in_(descriptor_ids),
    )
    for pref in (await session.execute(pref_stmt)).scalars().all():
        overrides[(str(pref.source_descriptor_id), pref.channel)] = pref.trust_tier
    return TrustPolicy(profiles=profiles, overrides=overrides)


__all__ = ["TrustPolicy", "load_trust_policy"]
