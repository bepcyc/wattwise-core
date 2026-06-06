"""Golden-reference tests for the Performance Management Chart (doc 40 section 3).

Requirement coverage: PMC-R1 (named EWMA model + previous-day TSB), PMC-R2
(``alpha = 1 - exp(-1/tau)`` impulse-response form), PMC-R4 (windowed equivalence),
PMC-R6 (calendar completeness), TEST-R1/R4 (golden with cited origin).

Fixture origin / derivation note
--------------------------------
The expected CTL/ATL/TSB values are HAND-DERIVED directly from the spec recurrence
(doc 40 section 3.1, also digest .digests/40.md section 3 "PMC EWMA"):

    alpha   = 1 - exp(-1/tau)
    CTL(d)  = CTL(d-1) + alpha_CTL * (L(d) - CTL(d-1))   tau_CTL = 42 days
    ATL(d)  = ATL(d-1) + alpha_ATL * (L(d) - ATL(d-1))   tau_ATL = 7 days
    TSB(d)  = CTL(d-1) - ATL(d-1)                        (previous-day balance)

seeded from origin ``(CTL(-1), ATL(-1)) = (0, 0)``. The first-day closed forms
``CTL(0) = L0 * alpha_CTL`` and ``ATL(0) = L0 * alpha_ATL`` (with ``CTL(-1)=ATL(-1)=0``)
are computed from python ``math.exp`` constants alone, INDEPENDENT of the module
under test. The constant-load convergence case uses the closed form
``CTL(d) = c * (1 - (1 - alpha)^(d+1))`` (standard EWMA-from-zero geometric-series
solution), again derived without reference to ``pmc.py``.

This matches the model named in Coggan & Allen's Performance Management Chart and
Banister's impulse-response model (the canonical TrainingPeaks CTL/ATL/TSB), whose
publicly documented defaults are tau_CTL=42 d and tau_ATL=7 d.
"""

from __future__ import annotations

import datetime as _dt
import math

import pytest

from wattwise_core.analytics.pmc import PmcDay, PmcSeed, ewma_alpha, pmc
from wattwise_core.analytics.result import Computed, Unavailable, UnavailableReason

pytestmark = pytest.mark.golden

# Hand-derived smoothing factors from the spec formula (independent of pmc.py).
ALPHA_CTL = 1.0 - math.exp(-1.0 / 42.0)
ALPHA_ATL = 1.0 - math.exp(-1.0 / 7.0)

# Closed-form numeric values pinned as goldens (printed from math.exp directly):
GOLD_ALPHA_CTL = 0.023528313347756735
GOLD_ALPHA_ATL = 0.1331221002498184

TOL = 1e-12  # closed-form recurrence; far tighter than the PMC-R4 1e-9 floor.


def _value(result: Computed[PmcDay] | Unavailable) -> PmcDay:
    assert isinstance(result, Computed)
    return result.value


def test_alpha_is_impulse_response_form() -> None:
    """PMC-R2: alpha = 1 - exp(-1/tau), NOT 2/(N+1)."""
    assert ewma_alpha(42.0) == pytest.approx(GOLD_ALPHA_CTL, abs=TOL)
    assert ewma_alpha(7.0) == pytest.approx(GOLD_ALPHA_ATL, abs=TOL)
    # Explicitly distinct from the forbidden simple-EWMA form alpha = 2/(N+1).
    assert ewma_alpha(42.0) != pytest.approx(2.0 / (42.0 + 1.0), abs=1e-3)


def test_first_day_closed_form_single_impulse() -> None:
    """PMC-R1/R2 golden: day-0 impulse from a zero origin seed.

    With seed (0,0): CTL(0) = 100*alpha_CTL, ATL(0) = 100*alpha_ATL, TSB(0) = 0-0.
    """
    series = pmc([100.0, 0.0, 0.0, 0.0, 0.0])
    assert len(series) == 5

    d0 = _value(series[0])
    assert d0.ctl == pytest.approx(100.0 * GOLD_ALPHA_CTL, abs=TOL)
    assert d0.atl == pytest.approx(100.0 * GOLD_ALPHA_ATL, abs=TOL)
    assert d0.ctl == pytest.approx(2.3528313347756735, abs=TOL)
    assert d0.atl == pytest.approx(13.312210024981841, abs=TOL)
    assert d0.tsb == pytest.approx(0.0, abs=TOL)

    # Day 1: TSB(1) = CTL(0) - ATL(0) (previous-day balance, PMC-R1).
    d1 = _value(series[1])
    assert d1.tsb == pytest.approx(d0.ctl - d0.atl, abs=TOL)
    # Hand-derived recurrence values from the reference computation.
    assert d1.ctl == pytest.approx(2.2974731818766507, abs=TOL)
    assert d1.atl == pytest.approx(11.54006066748957, abs=TOL)
    assert d1.tsb == pytest.approx(-10.959378690206167, abs=TOL)


def test_constant_load_converges_to_load() -> None:
    """PMC golden: constant L=c for t >> tau drives CTL -> c, ATL -> c, TSB -> 0.

    Closed form CTL(d) = c*(1 - (1-alpha)^(d+1)) from a zero origin seed.
    """
    c = 50.0
    n = 400
    series = pmc([c] * n)
    assert len(series) == n

    last = _value(series[-1])
    expected_ctl = c * (1.0 - (1.0 - ALPHA_CTL) ** n)
    expected_atl = c * (1.0 - (1.0 - ALPHA_ATL) ** n)
    assert last.ctl == pytest.approx(expected_ctl, abs=1e-9)
    assert last.atl == pytest.approx(expected_atl, abs=1e-9)

    # After 400 days (>> tau), both have all-but-converged to c; TSB ~ 0.
    assert last.ctl == pytest.approx(c, abs=1e-2)
    assert last.atl == pytest.approx(c, abs=1e-6)
    assert last.tsb == pytest.approx(0.0, abs=1e-2)


def test_windowed_equivalence_exact() -> None:
    """PMC-R4 (defining): windowed seeded result == full-history restricted to window.

    Golden loads are a fixed deterministic sequence (no RNG, ANL-R30).
    """
    loads = [126.663, 113.693, 63.086, 38.838, 76.691, 60.74, 117.57, 45.497, 71.49, 87.507]
    full = pmc(loads)

    # Seed for window [3, 6] is the carried-forward (CTL(2), ATL(2)) (PMC-R3).
    prev = _value(full[2])
    seed = PmcSeed(ctl_prev=prev.ctl, atl_prev=prev.atl)
    win = pmc(loads, window=(3, 6), seed=seed)
    assert len(win) == 4

    for k in range(4):
        f = _value(full[3 + k])
        w = _value(win[k])
        # PMC-R4 tolerance abs <= 1e-9 * max(1, |value|); here exact equality holds.
        assert w.ctl == pytest.approx(f.ctl, abs=1e-9 * max(1.0, abs(f.ctl)))
        assert w.atl == pytest.approx(f.atl, abs=1e-9 * max(1.0, abs(f.atl)))
        assert w.tsb == pytest.approx(f.tsb, abs=1e-9 * max(1.0, abs(f.tsb)))


def test_calendar_completeness_dict_input() -> None:
    """PMC-R6: a dict with a date gap materializes EVERY calendar day (no skips).

    Days with a missing key fall in an open coverage gap -> provisional, decay as 0.
    """
    d = _dt.date(2026, 1, 1)
    loads = {
        d: 100.0,
        d + _dt.timedelta(days=1): 80.0,
        # day +2 missing -> provisional
        d + _dt.timedelta(days=3): 60.0,
    }
    series = pmc(loads)
    assert len(series) == 4  # Jan 1..4 inclusive, no calendar day skipped.

    # Day +2 (index 2) is provisional and decays as if L=0.
    prov = series[2]
    assert isinstance(prov, Computed)
    assert prov.quality.extra["provisional"] is True
    # Its CTL/ATL equal the previous day's value decayed toward 0.
    prev = _value(series[1])
    assert prov.value.ctl == pytest.approx(prev.ctl + ALPHA_CTL * (0.0 - prev.ctl), abs=TOL)
    assert prov.value.atl == pytest.approx(prev.atl + ALPHA_ATL * (0.0 - prev.atl), abs=TOL)

    # Local dates are carried in lineage in calendar order with no skip.
    iso = [
        r.provenance.reference_params["local_date"]
        for r in series
        if isinstance(r, Computed)
    ]
    assert iso == ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"]


def test_mid_history_window_without_seed_is_not_seeded() -> None:
    """PMC-R5: a mid-history window with no derivable seed fails closed (NOT_SEEDED)."""
    loads = [10.0, 20.0, 30.0, 40.0, 50.0]
    res = pmc(loads, window=(2, 4), seed=None)
    assert len(res) == 1
    assert isinstance(res[0], Unavailable)
    assert res[0].reason is UnavailableReason.NOT_SEEDED
