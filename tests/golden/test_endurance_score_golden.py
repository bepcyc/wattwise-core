"""Golden-reference tests for the endurance score (doc 40 §7C, ES-R1/R2/R3).

Fixture origin / derivation (TEST-R4) — all expected values are hand-derived from the
DOCUMENTED normalization in ``defaults.toml`` ``[analytics] endurance_score_*`` (the
declared composition, ES-R1), independent of the implementation:

  Config (packaged defaults): weights w = (ctl 0.4, durability 0.3, decoupling 0.3),
  ctl_full_scale = 100, durability band [0.5, 1.0], decoupling_full_penalty = 10 %,
  partial_confidence_penalty = 0.7.

  Case A — full composition: CTL=70, ratio=0.85, drift=5 %:
    f_ctl = 70/100 = 0.7;  f_dur = (0.85-0.5)/0.5 = 0.7;  f_dec = 1 - 5/10 = 0.5
    score = 100·(0.4·0.7 + 0.3·0.7 + 0.3·0.5)/1.0 = 28 + 21 + 15 = 64.0

  Case B — CTL-only partial (ES-R2b): CTL=70, both power components Unavailable:
    score = 100·(0.4·0.7)/0.4 = 70.0, confidence = 0.7, components recorded.

  Case C — saturation clamp (ES-R3): CTL=250, ratio=1.5, drift=-3 % ⇒ every
    component clamps to 1 ⇒ score = 100.0 exactly (the named normalization bound).

  Durability ratio: MMP(1200 s)=255 W, MMP(300 s)=300 W ⇒ 255/300 = 0.85.
"""

from __future__ import annotations

import pytest

from wattwise_core.analytics.endurance_score import durability_ratio, endurance_score
from wattwise_core.analytics.result import (
    Computed,
    Unavailable,
    UnavailableReason,
)

TOL = 1e-9

_MISSING = Unavailable(UnavailableReason.MISSING_REQUIRED_INPUT, "absent upstream")


@pytest.mark.golden
def test_es_golden_full_composition() -> None:
    """ES-R1 Case A: CTL=70, ratio=0.85, drift=5% ⇒ score == 64.0 (hand-derived)."""
    result = endurance_score(
        Computed(value=70.0), Computed(value=0.85), Computed(value=5.0), sport="cycling"
    )
    assert isinstance(result, Computed)
    assert result.value == pytest.approx(64.0, abs=TOL)
    assert result.quality.confidence == 1.0
    assert result.quality.extra["components_missing"] == ()
    assert result.quality.extra["components_present"] == ("ctl", "decoupling", "durability")
    assert result.provenance.sport == "cycling"


@pytest.mark.golden
def test_es_golden_partial_ctl_only_renormalizes_not_zero_substitutes() -> None:
    """ES-R2b Case B: CTL-only subset ⇒ 70.0 (renormalized), reduced confidence, never 0-filled.

    A silent-0 substitution would give 100·0.4·0.7 = 28; the declared-valid partial
    composition renormalizes the weights instead: 100·(0.4·0.7)/0.4 = 70.0 (ANL-R4).
    """
    result = endurance_score(Computed(value=70.0), _MISSING, _MISSING)
    assert isinstance(result, Computed)
    assert result.value == pytest.approx(70.0, abs=TOL)
    assert result.value != pytest.approx(28.0, abs=1.0)  # the banned 0-substituted number
    assert result.quality.confidence == pytest.approx(0.7, abs=TOL)
    assert result.quality.extra["components_present"] == ("ctl",)
    assert result.quality.extra["components_missing"] == ("decoupling", "durability")


@pytest.mark.golden
def test_es_golden_saturation_clamps_to_100() -> None:
    """ES-R3 Case C: every component beyond its band ⇒ score == 100.0 exactly (the bound)."""
    result = endurance_score(
        Computed(value=250.0), Computed(value=1.5), Computed(value=-3.0)
    )
    assert isinstance(result, Computed)
    assert result.value == pytest.approx(100.0, abs=TOL)


@pytest.mark.golden
def test_es_golden_missing_ctl_fails_closed() -> None:
    """ES-R2(a)/ES-T3: missing non-substitutable CTL ⇒ Unavailable, never a 0-CTL score."""
    result = endurance_score(_MISSING, Computed(value=0.85), Computed(value=5.0))
    assert isinstance(result, Unavailable)
    assert result.reason is UnavailableReason.MISSING_REQUIRED_INPUT


@pytest.mark.golden
def test_es_golden_durability_ratio() -> None:
    """ES-R1 durability input: MMP(long)=255, MMP(short)=300 ⇒ ratio == 0.85 exactly."""
    result = durability_ratio(Computed(value=255.0), Computed(value=300.0))
    assert isinstance(result, Computed)
    assert result.value == pytest.approx(0.85, abs=TOL)


@pytest.mark.golden
def test_es_golden_durability_ratio_fails_closed() -> None:
    """Durability ratio fail-closed: a missing point or MMP(short) <= 0 is typed Unavailable."""
    missing = durability_ratio(_MISSING, Computed(value=300.0))
    assert isinstance(missing, Unavailable)
    assert missing.reason is UnavailableReason.MISSING_REQUIRED_INPUT

    degenerate = durability_ratio(Computed(value=255.0), Computed(value=0.0))
    assert isinstance(degenerate, Unavailable)
    assert degenerate.reason is UnavailableReason.OUT_OF_DOMAIN
