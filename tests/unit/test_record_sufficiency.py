"""Unit tests for the pure record-sufficiency axis (GROUND-R6 / DEGR-R2).

Pins the deterministic :func:`wattwise_core.analytics.sufficiency.assess_record_sufficiency`
contract: staleness is the clamped whole-day gap to the most recent OBSERVED activity, and the
three derived zones (fresh / stale-disclose / insufficient-abstain) plus the source-agnostic
fidelity label classify the record without re-deriving it. The function is pure (ANL-R2): a fixed
reference date in, a typed envelope out — no DB, no wall-clock, no RNG.
"""

from __future__ import annotations

import datetime as _dt

import pytest

from wattwise_core.analytics import constants
from wattwise_core.analytics.sufficiency import RecordSufficiency, assess_record_sufficiency

pytestmark = pytest.mark.unit

_REF = _dt.date(2026, 6, 10)
_FRESH = 2
_MAX = 14


def _assess(
    last: _dt.date | None, *, substituted: bool = False, sync_suspect: bool = False
) -> RecordSufficiency:
    return assess_record_sufficiency(
        reference_date=_REF,
        last_observed_date=last,
        fresh_within_days=_FRESH,
        max_staleness_days=_MAX,
        substituted=substituted,
        sync_suspect=sync_suspect,
    )


def test_fresh_record_is_full_fidelity_and_not_stale() -> None:
    """A 1-day-old observation is inside the caveat-free zone: full fidelity, not stale."""
    suff = _assess(_REF - _dt.timedelta(days=1), sync_suspect=True)
    assert suff.staleness_days == 1
    assert not suff.stale
    assert not suff.insufficient
    assert suff.fidelity == "full"


def test_disclose_zone_is_stale_but_sufficient() -> None:
    """The 7-day kill-chain gap with a suspect sync is STALE (disclose + block GO), sufficient."""
    suff = _assess(_REF - _dt.timedelta(days=7), sync_suspect=True)
    assert suff.staleness_days == 7
    assert suff.stale
    assert not suff.insufficient
    assert suff.fidelity == "partial"


def test_beyond_hard_floor_with_suspect_sync_is_insufficient() -> None:
    """A suspect-sync gap past 2*tau_ATL is INSUFFICIENT: assumed-rest tail dominates -> abstain."""
    suff = _assess(_REF - _dt.timedelta(days=40), sync_suspect=True)
    assert suff.staleness_days == 40
    assert suff.stale
    assert suff.insufficient
    assert suff.fidelity == "degraded"


def test_healthy_sync_gap_is_a_legitimate_taper_not_stale() -> None:
    """The MNAR disambiguator: a long gap with HEALTHY sync is a real taper -> trust it, no block.

    This is the case that a raw activity-gap signal would wrongly abstain: 40 days since the last
    ride but the pipeline is fine, so the fresh form is genuine and the verdict must ship.
    """
    suff = _assess(_REF - _dt.timedelta(days=40), sync_suspect=False)
    assert suff.staleness_days == 40
    assert not suff.stale
    assert not suff.insufficient
    assert suff.fidelity == "full"


def test_boundary_at_max_is_still_sufficient() -> None:
    """Exactly at the hard floor is the last tolerated day; one past it abstains (edge)."""
    assert not _assess(_REF - _dt.timedelta(days=_MAX), sync_suspect=True).insufficient
    assert _assess(_REF - _dt.timedelta(days=_MAX + 1), sync_suspect=True).insufficient


def test_never_observed_is_insufficient_regardless_of_sync() -> None:
    """No observed activity at all -> no freshness anchor -> insufficient/degraded (fail-closed)."""
    suff = _assess(None)
    assert suff.staleness_days is None
    assert not suff.observed
    assert suff.insufficient
    assert suff.fidelity == "degraded"


def test_future_observation_clamps_to_zero_staleness() -> None:
    """A future-dated observation (cross-source clock skew) never reads as negative staleness."""
    suff = _assess(_REF + _dt.timedelta(days=3), sync_suspect=True)
    assert suff.staleness_days == 0
    assert not suff.stale


def test_substituted_fresh_record_is_partial_not_full() -> None:
    """A fresh but HR-substituted load is PARTIAL fidelity (DEGR-R2), even when not stale."""
    suff = _assess(_REF, substituted=True)
    assert not suff.stale
    assert not suff.insufficient
    assert suff.substituted
    assert suff.fidelity == "partial"


def test_hard_floor_stays_keyed_to_the_atl_time_constant() -> None:
    """The stale-abstain floor is 2x tau_ATL (PMC-R1); drifting apart is an explicit decision.

    The value itself lives in config (CFG-R1a); this pin makes a change to EITHER the ATL
    time constant or the configured floor a loud, deliberate edit instead of a silent skew
    of the sufficiency model's scientific keying.
    """
    assert int(2 * constants.ATL_TIME_CONSTANT_DAYS) == constants.READINESS_MAX_STALENESS_DAYS
