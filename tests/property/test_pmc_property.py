"""Property-based tests for the Performance Management Chart (doc 40 section 3).

Per-metric property IDs (digest section 11.1, PMC-T1..T6):

* PMC-T1 -- type/Unavailable: result is always a list of typed MetricResult; the
  mid-history-no-seed degenerate path is Unavailable(NOT_SEEDED) (fail-closed).
* PMC-T2 -- windowed-equivalence ORACLE (the DEFINING property): a seeded window
  equals the full-from-origin computation restricted to that window to
  ``abs <= 1e-9 * max(1, |value|)`` (PMC-R4).
* PMC-T3 -- convergence: constant L=c for t >> tau drives CTL,ATL -> c, TSB -> 0.
* PMC-T4 -- decay monotonicity: after the last load, with all subsequent L=0,
  CTL and ATL strictly decrease toward 0 (PMC section 3.3).
* PMC-T5 -- calendar-gap completeness by local_date: a dict input materializes
  EVERY calendar day origin..max, never skipping a day (PMC-R6).
* PMC-T6 -- rest vs provisional: a known L=0 day is a clean true-rest day; a
  None (open-gap) day decays identically but is flagged provisional (PMC-R6).

Generators (TEST-R2): variable-length daily-load series, rest days (L=0),
provisional days (None), spikes, athlete tau over realistic ranges; with shrinking.
"""

from __future__ import annotations

import datetime as _dt
import math

import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from wattwise_core.analytics.pmc import PmcDay, PmcSeed, ewma_alpha, pmc
from wattwise_core.analytics.result import Computed, Unavailable, UnavailableReason

pytestmark = pytest.mark.property

# A daily load: finite non-negative float, or None (provisional open-gap day).
_loads = st.lists(
    st.one_of(
        st.none(),
        st.floats(min_value=0.0, max_value=400.0, allow_nan=False, allow_infinity=False),
    ),
    min_size=1,
    max_size=60,
)
# Realistic per-athlete tau ranges (default 42/7; allow configured spread).
_tau_ctl = st.floats(min_value=20.0, max_value=70.0, allow_nan=False)
_tau_atl = st.floats(min_value=3.0, max_value=14.0, allow_nan=False)


def _val(r: Computed[PmcDay] | Unavailable) -> PmcDay:
    assert isinstance(r, Computed)
    return r.value


# --- PMC-T1: type / fail-closed -------------------------------------------------


@given(loads=_loads, tau_ctl=_tau_ctl, tau_atl=_tau_atl)
def test_t1_always_list_of_metric_results(
    loads: list[float | None], tau_ctl: float, tau_atl: float
) -> None:
    out = pmc(loads, tau_ctl=tau_ctl, tau_atl=tau_atl)
    assert isinstance(out, list)
    # Full-from-origin: one entry per calendar day, never a bare number (ANL-R3).
    assert len(out) == len(loads)
    for r in out:
        assert isinstance(r, Computed | Unavailable)
        if isinstance(r, Computed):
            assert isinstance(r.value, PmcDay)
            assert math.isfinite(r.value.ctl)
            assert math.isfinite(r.value.atl)
            assert math.isfinite(r.value.tsb)


@given(
    loads=st.lists(st.floats(0.0, 300.0, allow_nan=False), min_size=3, max_size=30),
    d0=st.integers(min_value=1, max_value=29),
)
def test_t1_mid_history_no_seed_is_not_seeded(loads: list[float], d0: int) -> None:
    """PMC-R5: a non-origin window with no seed fails closed, never zero-seeds."""
    assume(d0 < len(loads))
    res = pmc(loads, window=(d0, len(loads) - 1), seed=None)
    assert len(res) == 1
    assert isinstance(res[0], Unavailable)
    assert res[0].reason is UnavailableReason.NOT_SEEDED


# --- PMC-T2: windowed-equivalence oracle (DEFINING) -----------------------------


@settings(suppress_health_check=[HealthCheck.filter_too_much])
@given(
    loads=st.lists(
        st.floats(0.0, 400.0, allow_nan=False, allow_infinity=False),
        min_size=2,
        max_size=60,
    ),
    tau_ctl=_tau_ctl,
    tau_atl=_tau_atl,
    data=st.data(),
)
def test_t2_windowed_equivalence_oracle(
    loads: list[float], tau_ctl: float, tau_atl: float, data: st.DataObject
) -> None:
    """PMC-R4/PMC-T2: seeded window == full-history restricted to window, to 1e-9."""
    n = len(loads)
    full = pmc(loads, tau_ctl=tau_ctl, tau_atl=tau_atl)

    d0 = data.draw(st.integers(min_value=0, max_value=n - 1))
    d1 = data.draw(st.integers(min_value=d0, max_value=n - 1))

    if d0 == 0:
        seed: PmcSeed | None = None
    else:
        prev = _val(full[d0 - 1])
        seed = PmcSeed(ctl_prev=prev.ctl, atl_prev=prev.atl)

    win = pmc(loads, tau_ctl=tau_ctl, tau_atl=tau_atl, window=(d0, d1), seed=seed)
    assert len(win) == d1 - d0 + 1

    for k in range(d1 - d0 + 1):
        f = _val(full[d0 + k])
        w = _val(win[k])
        assert w.ctl == pytest.approx(f.ctl, abs=1e-9 * max(1.0, abs(f.ctl)))
        assert w.atl == pytest.approx(f.atl, abs=1e-9 * max(1.0, abs(f.atl)))
        assert w.tsb == pytest.approx(f.tsb, abs=1e-9 * max(1.0, abs(f.tsb)))


# --- PMC-T3: convergence --------------------------------------------------------


@settings(deadline=None)
@given(c=st.floats(1.0, 300.0, allow_nan=False), tau_ctl=_tau_ctl, tau_atl=_tau_atl)
def test_t3_constant_load_converges(c: float, tau_ctl: float, tau_atl: float) -> None:
    """Constant L=c for t >> tau drives CTL,ATL -> c and TSB -> 0."""
    n = 2000  # >> both tau upper bounds
    out = pmc([c] * n, tau_ctl=tau_ctl, tau_atl=tau_atl)
    last = _val(out[-1])
    assert last.ctl == pytest.approx(c, rel=1e-6, abs=1e-6)
    assert last.atl == pytest.approx(c, rel=1e-6, abs=1e-6)
    assert last.tsb == pytest.approx(0.0, abs=1e-4)


# --- PMC-T4: decay monotonicity -------------------------------------------------


@given(
    spike=st.floats(10.0, 300.0, allow_nan=False),
    rest_days=st.integers(min_value=2, max_value=40),
    tau_ctl=_tau_ctl,
    tau_atl=_tau_atl,
)
def test_t4_decay_strictly_decreasing_after_load(
    spike: float, rest_days: int, tau_ctl: float, tau_atl: float
) -> None:
    """With L=0 after a positive load, CTL and ATL strictly decrease toward 0."""
    out = pmc([spike] + [0.0] * rest_days, tau_ctl=tau_ctl, tau_atl=tau_atl)
    ctls = [_val(r).ctl for r in out]
    atls = [_val(r).atl for r in out]
    # From index 0 (the spike) onward, every rest day decays strictly downward.
    for i in range(1, len(ctls)):
        assert ctls[i] < ctls[i - 1]
        assert atls[i] < atls[i - 1]
        assert ctls[i] > 0.0  # decays toward, never reaches, 0
        assert atls[i] > 0.0


# --- PMC-T5: calendar completeness by local_date --------------------------------


@settings(suppress_health_check=[HealthCheck.filter_too_much])
@given(
    span=st.integers(min_value=1, max_value=120),
    present=st.sets(st.integers(min_value=0, max_value=120), min_size=1),
)
def test_t5_dict_materializes_every_calendar_day(span: int, present: set[int]) -> None:
    """PMC-R6: dict input fills EVERY calendar day origin..max, never skipping."""
    present_in = {p for p in present if p <= span}
    assume(present_in)
    origin = _dt.date(2026, 1, 1)
    loads = {origin + _dt.timedelta(days=i): 50.0 for i in present_in}
    lo, hi = min(present_in), max(present_in)
    out = pmc(loads)
    # Span is from the min present key to the max present key (inclusive).
    assert len(out) == (hi - lo) + 1
    dates = [r.provenance.reference_params["local_date"] for r in out if isinstance(r, Computed)]
    expected = [(origin + _dt.timedelta(days=lo + i)).isoformat() for i in range((hi - lo) + 1)]
    assert dates == expected


# --- PMC-T6: rest vs provisional ------------------------------------------------


@given(
    head=st.floats(10.0, 300.0, allow_nan=False),
    tau_ctl=_tau_ctl,
    tau_atl=_tau_atl,
)
def test_t6_rest_and_provisional_decay_identically_but_flagged(
    head: float, tau_ctl: float, tau_atl: float
) -> None:
    """A None (provisional) day and a 0.0 (true-rest) day produce identical values.

    The numbers must match (both decay as L=0); only the provisional FLAG differs.
    """
    rest = pmc([head, 0.0], tau_ctl=tau_ctl, tau_atl=tau_atl)
    prov = pmc([head, None], tau_ctl=tau_ctl, tau_atl=tau_atl)

    r1 = _val(rest[1])
    p1 = _val(prov[1])
    assert p1.ctl == pytest.approx(r1.ctl, abs=1e-12)
    assert p1.atl == pytest.approx(r1.atl, abs=1e-12)
    assert p1.tsb == pytest.approx(r1.tsb, abs=1e-12)

    assert isinstance(rest[1], Computed)
    assert isinstance(prov[1], Computed)
    assert rest[1].quality.extra["provisional"] is False
    assert prov[1].quality.extra["provisional"] is True
    assert rest[1].quality.confidence == 1.0
    assert prov[1].quality.confidence < 1.0


# --- Determinism (ANL-R30) ------------------------------------------------------


@given(loads=_loads, tau_ctl=_tau_ctl, tau_atl=_tau_atl)
def test_determinism_bit_stable(loads: list[float | None], tau_ctl: float, tau_atl: float) -> None:
    a = pmc(loads, tau_ctl=tau_ctl, tau_atl=tau_atl)
    b = pmc(loads, tau_ctl=tau_ctl, tau_atl=tau_atl)
    for ra, rb in zip(a, b, strict=True):
        assert isinstance(ra, Computed)
        assert isinstance(rb, Computed)
        assert ra.value == rb.value


def test_alpha_rejects_nonpositive_tau() -> None:
    """ewma_alpha guards its domain (PMC-R2): tau must be positive and finite."""
    for bad in (0.0, -1.0, math.inf, math.nan):
        with pytest.raises(ValueError, match="tau"):
            ewma_alpha(bad)
