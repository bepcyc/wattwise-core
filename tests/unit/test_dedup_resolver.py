"""Unit tests for the OSS dedup/conflict resolver (CONF-R2, MAP-R10, DEDUP-R7)."""

from __future__ import annotations

import datetime as _dt
import uuid

import pytest

from wattwise_core.domain.candidate import FieldCandidate
from wattwise_core.domain.enums import TRUST_TIER_ORDER, Fidelity, SourceKind
from wattwise_core.ingestion.dedup import resolve_activity_identity, resolve_field
from wattwise_core.ingestion.trust import TrustPolicy
from wattwise_core.persistence.models import SourceCandidate, SourceDescriptor
from wattwise_core.persistence.models.athlete_preference import (
    AthleteSourcePreference,
    NonTierTrustError,
)

UTC = _dt.UTC


def _fc(value: object, tier: Fidelity, sid: str, **kw: object) -> FieldCandidate:
    return FieldCandidate(value=value, trust_tier=tier, source_descriptor_id=sid, **kw)  # type: ignore[arg-type]


def _cand(sid: str, adapter_tier: Fidelity) -> SourceCandidate:
    """A bare candidate carrying only its adapter-assigned tier (the layer-5 fallback)."""
    return SourceCandidate(
        source_descriptor_id=sid,
        source_native_id=sid,
        content_hash=sid,
        trust_profile={"tier": adapter_tier.value},
        payload={},
        confidence=1.0,
    )


def test_no_contributor_returns_none() -> None:
    """No candidate -> None so the caller records a typed gap, never a zero (CONF-R5)."""
    assert resolve_field([]) is None


def test_trust_tier_wins_first() -> None:
    """Highest fidelity wins regardless of recency/confidence (CONF-R2 step 1)."""
    raw = _fc(250.0, Fidelity.RAW_STREAM, "b", confidence=0.5)
    summ = _fc(260.0, Fidelity.SUMMARY_ONLY, "a", confidence=1.0)
    res = resolve_field([summ, raw])
    assert res is not None
    assert res.value == 250.0
    assert res.winning_source_descriptor_id == "b"


def test_confidence_breaks_tie_within_tier() -> None:
    """Within one tier, higher confidence wins (CONF-R2 step 2)."""
    a = _fc(100.0, Fidelity.DEVICE_COMPUTED, "a", confidence=0.6)
    b = _fc(110.0, Fidelity.DEVICE_COMPUTED, "b", confidence=0.9)
    res = resolve_field([a, b])
    assert res is not None
    assert res.value == 110.0


def test_recency_then_completeness_then_stable_tiebreak() -> None:
    """Recency, then completeness, then lowest source id as the final stable tiebreak."""
    older = _fc(
        1.0, Fidelity.MODELED, "z", confidence=1.0,
        observed_at=_dt.datetime(2026, 1, 1, tzinfo=UTC), completeness=1.0,
    )
    newer = _fc(
        2.0, Fidelity.MODELED, "y", confidence=1.0,
        observed_at=_dt.datetime(2026, 2, 1, tzinfo=UTC), completeness=1.0,
    )
    assert resolve_field([older, newer]).value == 2.0  # type: ignore[union-attr]

    # Identical on tiers 1-4 -> lowest source_descriptor_id wins (byte-reproducible).
    a = _fc(5.0, Fidelity.SUMMARY_ONLY, "aaa")
    b = _fc(6.0, Fidelity.SUMMARY_ONLY, "bbb")
    assert resolve_field([b, a]).winning_source_descriptor_id == "aaa"  # type: ignore[union-attr]


def test_disputed_flag_when_materially_disagree() -> None:
    """disputed=True when two numeric candidates differ beyond tolerance (CONF-R5)."""
    a = _fc(200.0, Fidelity.RAW_STREAM, "a")
    b = _fc(260.0, Fidelity.RAW_STREAM, "b")
    res = resolve_field([a, b], dispute_tolerance=0.1)
    assert res is not None
    assert res.disputed is True
    # Still picks the best, never averages.
    assert res.value in (200.0, 260.0)


def test_resolution_is_deterministic() -> None:
    """Same candidate set -> same winner regardless of input order (CONF-R4)."""
    cs = [
        _fc(1.0, Fidelity.MODELED, "m"),
        _fc(2.0, Fidelity.RAW_STREAM, "r"),
        _fc(3.0, Fidelity.SUMMARY_ONLY, "s"),
    ]
    first = resolve_field(cs)
    second = resolve_field(list(reversed(cs)))
    assert first == second


def test_identity_fingerprint_matches_regardless_of_window() -> None:
    """A shared strong fingerprint matches even outside the time window (MAP-R10)."""
    t1 = _dt.datetime(2026, 6, 1, 8, 0, tzinfo=UTC)
    t2 = _dt.datetime(2026, 6, 1, 9, 0, tzinfo=UTC)  # 1h apart
    assert resolve_activity_identity(
        t1, 3600, "cycling", "fit-abc", t2, 3600, "cycling", "fit-abc"
    )


def test_identity_window_and_duration_tolerance() -> None:
    """Within window + duration tolerance and same sport -> match; else not (MAP-R10)."""
    t1 = _dt.datetime(2026, 6, 1, 8, 0, 0, tzinfo=UTC)
    t2 = _dt.datetime(2026, 6, 1, 8, 1, 30, tzinfo=UTC)  # 90 s < 120 s window
    assert resolve_activity_identity(t1, 3600, "cycling", None, t2, 3650, "cycling", None)
    # Different sport never matches without a fingerprint.
    assert not resolve_activity_identity(t1, 3600, "cycling", None, t1, 3600, "running", None)
    # Outside the start window with no fingerprint -> no match.
    t3 = _dt.datetime(2026, 6, 1, 8, 5, 0, tzinfo=UTC)  # 300 s > window
    assert not resolve_activity_identity(t1, 3600, "cycling", None, t3, 3600, "cycling", None)


# ------------------------------------- CON-R4 wiring: configurable effective trust tier
#
# The effective tier is computed from CONFIGURATION DATA in this strict first-hit order
# (PRV-R7 / LIN-R1 / CONF-R1): per-athlete (source, channel) override > per-athlete
# (source, "*") override > descriptor trust_profile[channel] > descriptor
# trust_profile["*"]/default_fidelity > the candidate's adapter-assigned tier.


def test_effective_tier_empty_config_falls_through_to_adapter_tier() -> None:
    """INVARIANT: with NO profile and NO override the effective tier IS the adapter tier.

    This is the opt-in guarantee — unconfigured resolution is byte-identical to the prior
    behaviour, so PRV-R6 (actual higher-fidelity wins) is preserved by default.
    """
    policy = TrustPolicy(profiles={}, overrides={})
    cand = _cand("src-a", Fidelity.RAW_STREAM)
    assert policy.tier(cand, "avg_power_w") == Fidelity.RAW_STREAM
    assert policy.tier(cand, "*") == Fidelity.RAW_STREAM


def test_effective_tier_descriptor_profile_channel_then_star() -> None:
    """Descriptor trust_profile re-ranks: per-channel entry beats the "*" base (LIN-R1)."""
    # The adapter says RAW_STREAM, but the descriptor declares this channel SUMMARY_ONLY
    # and its whole-source base MODELED — the per-channel declaration wins over both.
    policy = TrustPolicy(
        profiles={
            "src-a": (
                {"avg_power_w": Fidelity.SUMMARY_ONLY.value, "*": Fidelity.MODELED.value},
                None,
            )
        },
        overrides={},
    )
    cand = _cand("src-a", Fidelity.RAW_STREAM)
    assert policy.tier(cand, "avg_power_w") == Fidelity.SUMMARY_ONLY  # per-channel base
    assert policy.tier(cand, "avg_hr_bpm") == Fidelity.MODELED  # falls to "*" base


def test_effective_tier_default_fidelity_is_the_whole_source_fallback() -> None:
    """default_fidelity stands in for a missing trust_profile["*"] (LIN-R1 layer 4)."""
    policy = TrustPolicy(
        profiles={"src-a": ({}, Fidelity.MODELED.value)}, overrides={}
    )
    cand = _cand("src-a", Fidelity.RAW_STREAM)
    assert policy.tier(cand, "avg_power_w") == Fidelity.MODELED


def test_effective_tier_athlete_override_beats_profile_and_adapter() -> None:
    """A per-athlete (source, channel) override is the HIGHEST-precedence layer (PRV-R7)."""
    policy = TrustPolicy(
        profiles={"src-a": ({"avg_power_w": Fidelity.SUMMARY_ONLY.value}, None)},
        overrides={("src-a", "avg_power_w"): Fidelity.RAW_STREAM},
    )
    cand = _cand("src-a", Fidelity.MODELED)
    # Override wins over both the descriptor per-channel base and the adapter tier.
    assert policy.tier(cand, "avg_power_w") == Fidelity.RAW_STREAM


def test_effective_tier_athlete_whole_source_override_applies_to_all_channels() -> None:
    """A per-athlete (source, "*") override is the whole-source default for that athlete."""
    policy = TrustPolicy(
        profiles={}, overrides={("src-a", "*"): Fidelity.SUMMARY_ONLY}
    )
    cand = _cand("src-a", Fidelity.RAW_STREAM)
    # No channel-specific override -> the "*" override applies to every channel.
    assert policy.tier(cand, "avg_power_w") == Fidelity.SUMMARY_ONLY
    assert policy.tier(cand, "avg_hr_bpm") == Fidelity.SUMMARY_ONLY


def test_effective_tier_channel_override_beats_whole_source_override() -> None:
    """The (source, channel) override is consulted BEFORE the (source, "*") override."""
    policy = TrustPolicy(
        profiles={},
        overrides={
            ("src-a", "avg_power_w"): Fidelity.RAW_STREAM,
            ("src-a", "*"): Fidelity.SUMMARY_ONLY,
        },
    )
    cand = _cand("src-a", Fidelity.MODELED)
    assert policy.tier(cand, "avg_power_w") == Fidelity.RAW_STREAM  # specific channel
    assert policy.tier(cand, "avg_hr_bpm") == Fidelity.SUMMARY_ONLY  # whole-source "*"


def test_effective_tier_keying_is_by_descriptor_id_not_source_name() -> None:
    """CONF-R1: the policy is keyed by source_descriptor_id, never a source NAME string.

    Two candidates with DIFFERENT descriptor ids resolve independently from the same
    policy; there is no source-name branch anywhere — only the opaque descriptor id keys.
    """
    policy = TrustPolicy(
        profiles={"desc-1": ({"avg_power_w": Fidelity.RAW_STREAM.value}, None)},
        overrides={("desc-2", "avg_power_w"): Fidelity.SUMMARY_ONLY},
    )
    c1 = _cand("desc-1", Fidelity.MODELED)
    c2 = _cand("desc-2", Fidelity.MODELED)
    # desc-1 picks up its descriptor profile; desc-2 picks up its athlete override; a
    # third, unkeyed descriptor falls through to its own adapter tier.
    assert policy.tier(c1, "avg_power_w") == Fidelity.RAW_STREAM
    assert policy.tier(c2, "avg_power_w") == Fidelity.SUMMARY_ONLY
    assert policy.tier(_cand("desc-3", Fidelity.DEVICE_COMPUTED), "avg_power_w") == (
        Fidelity.DEVICE_COMPUTED
    )


def test_effective_tier_malformed_config_token_falls_through() -> None:
    """A garbage profile token is ignored (tolerant config) -> next layer / adapter tier."""
    policy = TrustPolicy(profiles={"src-a": ({"avg_power_w": "not-a-tier"}, None)}, overrides={})
    cand = _cand("src-a", Fidelity.DEVICE_COMPUTED)
    assert policy.tier(cand, "avg_power_w") == Fidelity.DEVICE_COMPUTED


# ------------------------------------- CONF-R2: a NON-TIER Fidelity is never an effective
# tier (the per-athlete ingest-DoS guard).
#
# Only the 5 ranked tiers (TRUST_TIER_ORDER) are valid trust tiers. The 3 non-tier members
# (substituted / absent_true / absent_failed) are outcome states. If one of them ever
# became an effective tier, a downstream coverage_for(present=True, <non-tier>, ...) would
# raise and ABORT the whole ingest batch (a durable per-athlete ingest DoS). The resolver
# coerce-clamp must drop a non-tier at EVERY config layer so it can never reach coverage.

_NON_TIER_FIDELITIES = (
    Fidelity.SUBSTITUTED,
    Fidelity.ABSENT_TRUE,
    Fidelity.ABSENT_FAILED,
)


@pytest.mark.parametrize("non_tier", _NON_TIER_FIDELITIES)
def test_effective_tier_non_tier_athlete_override_falls_through_to_adapter(
    non_tier: Fidelity,
) -> None:
    """A per-athlete override set to a NON-TIER falls through to the adapter tier (CONF-R2).

    The returned tier is the candidate's adapter (ranked) tier, NEVER the non-tier — so a
    downstream coverage_for(True, tier) can never see a non-tier and abort the batch.
    """
    policy = TrustPolicy(profiles={}, overrides={("src-a", "avg_power_w"): non_tier})
    cand = _cand("src-a", Fidelity.DEVICE_COMPUTED)
    resolved = policy.tier(cand, "avg_power_w")
    assert resolved == Fidelity.DEVICE_COMPUTED
    assert resolved in TRUST_TIER_ORDER  # never a non-tier


@pytest.mark.parametrize("non_tier", _NON_TIER_FIDELITIES)
def test_effective_tier_non_tier_whole_source_override_falls_through(
    non_tier: Fidelity,
) -> None:
    """A whole-source ("*") per-athlete override set to a NON-TIER also falls through."""
    policy = TrustPolicy(profiles={}, overrides={("src-a", "*"): non_tier})
    cand = _cand("src-a", Fidelity.MODELED)
    resolved = policy.tier(cand, "avg_power_w")
    assert resolved == Fidelity.MODELED
    assert resolved in TRUST_TIER_ORDER


@pytest.mark.parametrize("non_tier", _NON_TIER_FIDELITIES)
def test_effective_tier_non_tier_descriptor_profile_falls_through(
    non_tier: Fidelity,
) -> None:
    """A descriptor trust_profile[channel] set to a NON-TIER falls through (CONF-R2)."""
    policy = TrustPolicy(
        profiles={"src-a": ({"avg_power_w": non_tier.value}, None)}, overrides={}
    )
    cand = _cand("src-a", Fidelity.PLATFORM_COMPUTED)
    resolved = policy.tier(cand, "avg_power_w")
    assert resolved == Fidelity.PLATFORM_COMPUTED
    assert resolved in TRUST_TIER_ORDER


@pytest.mark.parametrize("non_tier", _NON_TIER_FIDELITIES)
def test_effective_tier_non_tier_default_fidelity_falls_through(
    non_tier: Fidelity,
) -> None:
    """A descriptor default_fidelity set to a NON-TIER falls through to the adapter tier."""
    policy = TrustPolicy(profiles={"src-a": ({}, non_tier.value)}, overrides={})
    cand = _cand("src-a", Fidelity.SUMMARY_ONLY)
    resolved = policy.tier(cand, "avg_power_w")
    assert resolved == Fidelity.SUMMARY_ONLY
    assert resolved in TRUST_TIER_ORDER


@pytest.mark.parametrize("non_tier", _NON_TIER_FIDELITIES)
def test_effective_tier_non_tier_override_with_tier_profile_uses_profile(
    non_tier: Fidelity,
) -> None:
    """A non-tier override is dropped, so a valid descriptor profile (layer 3) wins.

    Proves the clamp falls to the NEXT layer (not straight to the adapter tier) — the
    override is ignored exactly like a missing override.
    """
    policy = TrustPolicy(
        profiles={"src-a": ({"avg_power_w": Fidelity.RAW_STREAM.value}, None)},
        overrides={("src-a", "avg_power_w"): non_tier},
    )
    cand = _cand("src-a", Fidelity.MODELED)
    assert policy.tier(cand, "avg_power_w") == Fidelity.RAW_STREAM


# ------------------------------------- CONF-R2: the write-time app validator rejects a
# non-tier token for the per-athlete override AND the descriptor config fields.


@pytest.mark.parametrize("ranked", TRUST_TIER_ORDER)
def test_app_validator_accepts_each_ranked_tier(ranked: Fidelity) -> None:
    """All 5 ranked tiers are accepted for trust_tier / default_fidelity / trust_profile."""
    pref = AthleteSourcePreference(
        athlete_id=uuid.uuid4(),
        source_descriptor_id=uuid.uuid4(),
        channel="avg_power_w",
        trust_tier=ranked,
    )
    assert pref.trust_tier == ranked

    desc = SourceDescriptor(
        source_key="s",
        display_name="S",
        kind=SourceKind.OAUTH_API,
        trust_profile={"avg_power_w": ranked.value, "*": ranked.value},
        default_fidelity=ranked.value,
    )
    assert desc.default_fidelity == ranked.value
    assert desc.trust_profile["avg_power_w"] == ranked.value


@pytest.mark.parametrize("non_tier", _NON_TIER_FIDELITIES)
def test_app_validator_rejects_non_tier_athlete_preference(non_tier: Fidelity) -> None:
    """Constructing an AthleteSourcePreference with a non-tier trust_tier raises (typed)."""
    with pytest.raises(NonTierTrustError):
        AthleteSourcePreference(
            athlete_id=uuid.uuid4(),
            source_descriptor_id=uuid.uuid4(),
            channel="avg_power_w",
            trust_tier=non_tier,
        )


@pytest.mark.parametrize("non_tier", _NON_TIER_FIDELITIES)
def test_app_validator_rejects_non_tier_default_fidelity(non_tier: Fidelity) -> None:
    """A SourceDescriptor.default_fidelity set to a non-tier token raises at write time."""
    with pytest.raises(NonTierTrustError):
        SourceDescriptor(
            source_key="s",
            display_name="S",
            kind=SourceKind.OAUTH_API,
            default_fidelity=non_tier.value,
        )


@pytest.mark.parametrize("non_tier", _NON_TIER_FIDELITIES)
def test_app_validator_rejects_non_tier_trust_profile_value(non_tier: Fidelity) -> None:
    """A SourceDescriptor.trust_profile carrying a non-tier tier token raises at write."""
    with pytest.raises(NonTierTrustError):
        SourceDescriptor(
            source_key="s",
            display_name="S",
            kind=SourceKind.OAUTH_API,
            trust_profile={"avg_power_w": non_tier.value},
        )


def test_app_validator_rejects_garbage_tokens() -> None:
    """An unknown (non-Fidelity) token is also rejected for every trust-tier config field."""
    with pytest.raises(NonTierTrustError):
        AthleteSourcePreference(
            athlete_id=uuid.uuid4(),
            source_descriptor_id=uuid.uuid4(),
            channel="*",
            trust_tier="not-a-tier",
        )
    with pytest.raises(NonTierTrustError):
        SourceDescriptor(
            source_key="s",
            display_name="S",
            kind=SourceKind.OAUTH_API,
            default_fidelity="not-a-tier",
        )


def test_app_validator_allows_none_and_empty_descriptor_config() -> None:
    """The seeded default (no default_fidelity, empty trust_profile) is always valid."""
    desc = SourceDescriptor(
        source_key="s",
        display_name="S",
        kind=SourceKind.OAUTH_API,
        trust_profile={},
        default_fidelity=None,
    )
    assert desc.default_fidelity is None
    assert desc.trust_profile == {}


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
