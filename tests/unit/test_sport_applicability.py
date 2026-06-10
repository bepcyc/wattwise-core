"""SPORT-T1 (doc 40 §11.1, ANL-R11): every metric declares its applicable sports.

Each metric module exposes an ``APPLICABLE_SPORTS`` declaration as METADATA on the
metric (ANL-R11 — not a branch inside formula code):

* the cycling-power family (NP/IF/power-TSS via ``np_if_tss``, W'balance, CP/W',
  MMP/power best-efforts) is **sport-specific** — an enumerated tuple, cycling first;
* the HR family (``trimp``/HRLoad, time- and freq-domain HRV) is **sport-agnostic**
  — declared as ``None`` (meaningful for every sport supplying the HR channel);
* aerobic decoupling is **sport-parameterized** (ANL-R13) — declared for the union of
  its per-sport output-channel mappings;
* the endurance score composes upstream metrics, so its own declaration is
  sport-agnostic while its power components gate upstream (§7C).

The sport-MISMATCH behavior these declarations gate (NOT_APPLICABLE_FOR_SPORT,
ANL-R12) is proven separately by SPORT-T2/T3 in the sport-gating golden/property
tiers — this test pins the declarations themselves.
"""

from __future__ import annotations

import pytest

from wattwise_core.analytics import (
    cp,
    decoupling,
    endurance_score,
    hrv,
    hrv_freq,
    mmp_cp,
    np_if_tss,
    trimp,
    wbal,
)

_POWER_FAMILY_MODULES = (np_if_tss, wbal, mmp_cp, cp)
_SPORT_AGNOSTIC_MODULES = (trimp, hrv, hrv_freq, endurance_score)


@pytest.mark.unit
def test_sport_t1_every_metric_module_declares_applicability() -> None:
    """SPORT-T1: every metric module exposes an ``APPLICABLE_SPORTS`` declaration."""
    for module in (*_POWER_FAMILY_MODULES, *_SPORT_AGNOSTIC_MODULES, decoupling):
        assert hasattr(module, "APPLICABLE_SPORTS"), module.__name__
        assert "APPLICABLE_SPORTS" in module.__all__, module.__name__


@pytest.mark.unit
def test_sport_t1_power_family_is_sport_specific_cycling_first() -> None:
    """SPORT-T1: the cycling-power family declares an enumerated set containing cycling."""
    for module in _POWER_FAMILY_MODULES:
        declared = module.APPLICABLE_SPORTS
        assert isinstance(declared, tuple), module.__name__
        assert "cycling" in declared, module.__name__


@pytest.mark.unit
def test_sport_t1_hr_family_is_sport_agnostic() -> None:
    """SPORT-T1: the HR family (TRIMP/HRV) and the ES composition declare sport-agnostic."""
    for module in _SPORT_AGNOSTIC_MODULES:
        assert module.APPLICABLE_SPORTS is None, module.__name__


@pytest.mark.unit
def test_sport_t1_decoupling_is_sport_parameterized() -> None:
    """SPORT-T1/ANL-R13: decoupling declares the union of its per-sport output mappings."""
    declared = decoupling.APPLICABLE_SPORTS
    assert isinstance(declared, frozenset)
    assert "cycling" in declared  # the power (Pw:Hr) realization
    assert "running" in declared  # the pace (Pa:Hr) realization
    # The declaration is DERIVED from the same sets the runtime gate consumes, so a
    # sport outside it is exactly a sport the gate rejects (no drift possible).
    assert declared == decoupling._POWER_SPORTS | decoupling._PACE_SPORTS
