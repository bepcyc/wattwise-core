"""Unit tests for PMC per-day load fidelity / substitution surfacing (DEGR-R2).

Pins the pure :func:`wattwise_core.analytics.pmc.pmc` contract that when a day's load is
threaded with a SUBSTITUTED equivalence-class coverage, the day's :class:`PmcDay` carries
that coverage and its :class:`QualityReport` shows reduced confidence + ``substituted`` /
``from_fidelity`` — never presenting a substituted load as full raw-stream fidelity — while
the numeric CTL/ATL/TSB recurrence is UNCHANGED (annotation-only, ANL-R2 determinism).
"""

from __future__ import annotations

import pytest

from wattwise_core.analytics.pmc import pmc
from wattwise_core.analytics.result import is_computed
from wattwise_core.domain.coverage import Coverage, Substitution
from wattwise_core.domain.enums import Fidelity

pytestmark = pytest.mark.unit

_SUBSTITUTED = Coverage(
    present=True,
    fidelity=Fidelity.SUBSTITUTED,
    substitution=Substitution(equivalence_class="training_load", from_fidelity=Fidelity.RAW_STREAM),
)
_TOP = Coverage(present=True, fidelity=Fidelity.RAW_STREAM)


def test_substituted_day_surfaces_downgrade_with_reduced_confidence() -> None:
    """A SUBSTITUTED-load day carries coverage + reduced confidence + from_fidelity (DEGR-R2)."""
    loads = [100.0, 50.0, 75.0]
    coverage = [_TOP, _SUBSTITUTED, _TOP]
    series = pmc(loads, day_load_coverage=coverage)

    sub_day = series[1]
    assert is_computed(sub_day)
    cov = sub_day.value.load_coverage
    assert cov is not None
    assert cov.fidelity is Fidelity.SUBSTITUTED
    assert cov.fidelity is not Fidelity.MODELED  # never the displaced member's own tier
    assert cov.substitution is not None
    assert cov.substitution.from_fidelity is Fidelity.RAW_STREAM
    assert sub_day.quality.confidence < 1.0
    assert sub_day.quality.extra["substituted"] is True
    assert sub_day.quality.extra["from_fidelity"] == Fidelity.RAW_STREAM.value

    # A non-substituted (top-tier) day is NOT flagged and stays full confidence.
    top_day = series[0]
    assert is_computed(top_day)
    assert top_day.quality.confidence == 1.0
    assert "substituted" not in top_day.quality.extra


def test_load_coverage_is_annotation_only_recurrence_unchanged() -> None:
    """Threading coverage must NOT change the numeric CTL/ATL/TSB recurrence (ANL-R2)."""
    loads = [120.0, 0.0, None, 80.0, 60.0]
    coverage = [_SUBSTITUTED, _TOP, None, _SUBSTITUTED, _TOP]
    bare = pmc(loads)
    annotated = pmc(loads, day_load_coverage=coverage)
    assert len(bare) == len(annotated) == len(loads)
    for b, a in zip(bare, annotated, strict=True):
        assert is_computed(b) and is_computed(a)
        assert a.value.ctl == pytest.approx(b.value.ctl, abs=1e-12)
        assert a.value.atl == pytest.approx(b.value.atl, abs=1e-12)
        assert a.value.tsb == pytest.approx(b.value.tsb, abs=1e-12)
    # Bare-recurrence days carry no per-day coverage (back-compat).
    assert all(is_computed(d) and d.value.load_coverage is None for d in bare)


def test_positional_coverage_length_mismatch_is_a_caller_error() -> None:
    """A positional day_load_coverage that does not match the grid length fails closed."""
    with pytest.raises(ValueError, match="match the daily_load grid length"):
        pmc([1.0, 2.0, 3.0], day_load_coverage=[_TOP])
