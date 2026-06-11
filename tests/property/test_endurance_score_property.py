"""Property-based tests for the endurance score (doc 40 §7C, ES-T1..ES-T4).

Covers, per the per-metric property list (doc 40 §11.1):

* ES-T1 — composition only: fixed upstream ``MetricResult``\\ s yield a deterministic
          score; the module never touches a raw stream (asserted structurally: no
          stream/DB imports in the module).
* ES-T2 — monotonicity (ES-R3): perturbing one component in the documented direction
          moves the score the documented way (higher CTL ⇒ not-lower; higher
          curve-shape ratio ⇒ not-lower; higher decoupling drift ⇒ not-higher).
* ES-T3 — missing component: an ``Unavailable`` non-substitutable input (CTL) yields
          ``Unavailable``; a missing optional component is renormalized away, never
          silently scored as ``0`` (ANL-R4); the configured ``allow_partial = False``
          policy fails closed instead.
* ES-T4 — bounds: the score is ALWAYS within ``[0, 100]`` (ES-R3).

Generators follow TEST-R2: realistic CTL / ratio / drift ranges with shrinking.
"""

from __future__ import annotations

import inspect

import pytest
from hypothesis import given
from hypothesis import strategies as st

from wattwise_core.analytics import endurance_score as es_mod
from wattwise_core.analytics.endurance_score import endurance_score
from wattwise_core.analytics.result import (
    Computed,
    Unavailable,
    UnavailableReason,
)

_MISSING = Unavailable(UnavailableReason.MISSING_REQUIRED_INPUT, "absent upstream")

_ctl = st.floats(min_value=0.0, max_value=200.0, allow_nan=False, allow_infinity=False)
_ratio = st.floats(min_value=0.0, max_value=2.0, allow_nan=False, allow_infinity=False)
_drift = st.floats(min_value=-20.0, max_value=40.0, allow_nan=False, allow_infinity=False)
# A strictly positive perturbation for the monotonicity probes.
_delta = st.floats(min_value=1e-3, max_value=50.0, allow_nan=False, allow_infinity=False)


def _score(ctl: float, ratio: float, drift: float) -> float:
    """The composed score for three present components (test shorthand)."""
    result = endurance_score(Computed(value=ctl), Computed(value=ratio), Computed(value=drift))
    assert isinstance(result, Computed)
    return result.value


@pytest.mark.property
@given(ctl=_ctl, ratio=_ratio, drift=_drift)
def test_es_t1_deterministic_composition(ctl: float, ratio: float, drift: float) -> None:
    """ES-T1: the same upstream MetricResults always yield the identical score (ANL-R30)."""
    assert _score(ctl, ratio, drift) == _score(ctl, ratio, drift)


@pytest.mark.property
def test_es_t1_no_raw_stream_access() -> None:
    """ES-T1: the module is a pure composition — no stream, DB, or loader imports (ES-R2)."""
    source = inspect.getsource(es_mod)
    for banned in ("analytics.series", "sqlalchemy", "persistence", "resample_to_1hz"):
        assert banned not in source, f"endurance_score must not reference {banned!r}"


@pytest.mark.property
@given(ctl=_ctl, ratio=_ratio, drift=_drift, delta=_delta)
def test_es_t2_monotonic_in_ctl(ctl: float, ratio: float, drift: float, delta: float) -> None:
    """ES-T2: higher CTL at fixed other inputs ⇒ not-lower score (ES-R3)."""
    assert _score(ctl + delta, ratio, drift) >= _score(ctl, ratio, drift)


@pytest.mark.property
@given(ctl=_ctl, ratio=_ratio, drift=_drift, delta=_delta)
def test_es_t2_monotonic_in_curve_shape(
    ctl: float, ratio: float, drift: float, delta: float
) -> None:
    """ES-T2: higher curve-shape ratio at fixed other inputs ⇒ not-lower score (ES-R3)."""
    assert _score(ctl, ratio + delta, drift) >= _score(ctl, ratio, drift)


@pytest.mark.property
@given(ctl=_ctl, ratio=_ratio, drift=_drift, delta=_delta)
def test_es_t2_antitonic_in_decoupling(
    ctl: float, ratio: float, drift: float, delta: float
) -> None:
    """ES-T2: higher decoupling drift at fixed other inputs ⇒ not-HIGHER score (ES-R3)."""
    assert _score(ctl, ratio, drift + delta) <= _score(ctl, ratio, drift)


@pytest.mark.property
@given(ctl=_ctl, ratio=_ratio, drift=_drift)
def test_es_t4_bounds(ctl: float, ratio: float, drift: float) -> None:
    """ES-T4: the score is always within [0, 100], whatever the (finite) inputs (ES-R3)."""
    score = _score(ctl, ratio, drift)
    assert 0.0 <= score <= 100.0


@pytest.mark.property
@given(ctl=_ctl)
def test_es_t3_missing_ctl_is_unavailable(ctl: float) -> None:
    """ES-T3: missing non-substitutable CTL ⇒ Unavailable, regardless of other components."""
    result = endurance_score(_MISSING, Computed(value=0.8), Computed(value=ctl / 10.0))
    assert isinstance(result, Unavailable)
    assert result.reason is UnavailableReason.MISSING_REQUIRED_INPUT


@pytest.mark.property
@given(ctl=st.floats(min_value=10.0, max_value=200.0, allow_nan=False, allow_infinity=False))
def test_es_t3_partial_never_zero_substitutes(ctl: float) -> None:
    """ES-T3: a missing optional component is renormalized away, never scored as 0 (ANL-R4).

    For any positive CTL, the partial (CTL-only) score is STRICTLY greater than the
    wrong-but-plausible score a silent 0-substitution would have produced — proving the
    missing components were excluded from the weighting, not zero-filled.
    """
    partial = endurance_score(Computed(value=ctl), _MISSING, _MISSING)
    assert isinstance(partial, Computed)
    zero_substituted = endurance_score(
        Computed(value=ctl), Computed(value=0.0), Computed(value=1e9)
    )
    assert isinstance(zero_substituted, Computed)
    assert partial.value > zero_substituted.value
    assert partial.quality.confidence < 1.0
    assert partial.quality.extra["components_missing"] == ("curve_shape", "decoupling")


@pytest.mark.property
def test_es_t3_allow_partial_false_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """ES-R2: with the partial policy NOT declared valid, any missing component fails closed."""
    monkeypatch.setattr(es_mod, "ES_ALLOW_PARTIAL", False)
    result = endurance_score(Computed(value=70.0), _MISSING, Computed(value=5.0))
    assert isinstance(result, Unavailable)
    assert result.reason is UnavailableReason.MISSING_REQUIRED_INPUT


@pytest.mark.property
def test_es_degenerate_weights_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """A zero weight-sum over present components is OUT_OF_DOMAIN, never a 0/0 score."""
    monkeypatch.setattr(es_mod, "_WEIGHTS", {"ctl": 0.0, "curve_shape": 0.0, "decoupling": 0.0})
    result = endurance_score(Computed(value=70.0), Computed(value=0.8), Computed(value=5.0))
    assert isinstance(result, Unavailable)
    assert result.reason is UnavailableReason.OUT_OF_DOMAIN


@pytest.mark.property
def test_es_non_finite_component_treated_as_absent() -> None:
    """A non-finite upstream value is treated as absent (fail-closed), never composed."""
    result = endurance_score(
        Computed(value=70.0), Computed(value=float("nan")), Computed(value=5.0)
    )
    assert isinstance(result, Computed)
    assert result.quality.extra["components_missing"] == ("curve_shape",)
