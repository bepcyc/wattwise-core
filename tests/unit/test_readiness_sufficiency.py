"""Unit tests for the readiness record-freshness gate (GROUND-R6, the issue #12 kill-chain).

Drives the REAL :func:`wattwise_core.agent.readiness_deliverable.readiness_assessment` with a
sufficiency envelope and asserts the asymmetric fail-closed policy: a stale record never ships the
most-aggressive GO (the manufactured-freshness failure), a record past the hard floor abstains
truthfully, a less-aggressive verdict on a stale record still ships but DEGRADED + caveated, and —
critically — omitting the envelope preserves the prior inputs-only behaviour verbatim (so every
existing caller and test is unaffected). No model/grounder is wired: the deliverable falls back to
its deterministic per-verdict state sentence, isolating the freshness policy under test.
"""

from __future__ import annotations

import datetime as _dt

import pytest

from wattwise_core.agent.readiness_deliverable import (
    STALE_DATA_CLAUSE,
    readiness_assessment,
)
from wattwise_core.analytics.sufficiency import assess_record_sufficiency
from wattwise_core.domain.enums import ReadinessVerdict

pytestmark = pytest.mark.unit

_REF = _dt.date(2026, 6, 10)
_FRESH = 2
_MAX = 14

# A deep-positive form (TSB) the oracle bands as GO — the value a broken-sync EWMA inflates to.
_FRESH_FORM = 30.0
# A deep-negative form the oracle bands as REST — a safe-side (more-cautious) verdict.
_FATIGUED_FORM = -30.0


def _suff(days: int | None, *, substituted: bool = False, sync_suspect: bool = True):
    # Default sync_suspect=True for the freshness-gate tests: an old observation only blocks a
    # verdict when corroborated by a broken/stalled connector (the MNAR disambiguator). The
    # healthy-sync taper case is covered explicitly below.
    last = None if days is None else _REF - _dt.timedelta(days=days)
    return assess_record_sufficiency(
        reference_date=_REF,
        last_observed_date=last,
        fresh_within_days=_FRESH,
        max_staleness_days=_MAX,
        substituted=substituted,
        sync_suspect=sync_suspect,
    )


async def _assess(form: float, suff):
    # Present, non-suppressed HRV (60 vs 58 baseline) so the oracle's verdict is read from BOTH
    # inputs and a DEGRADED status is attributable to the freshness gate alone, not missing HRV.
    return await readiness_assessment(
        "ath-1",
        form=form,
        as_of="2026-06-10",
        hrv_rmssd=60.0,
        hrv_baseline=58.0,
        narrate=None,
        grounder=None,
        sufficiency=suff,
    )


async def test_stale_record_never_ships_aggressive_go() -> None:
    """The kill-chain: inflated GO form on a 7-day-stale record -> STALE ABSTAIN, no verdict."""
    readiness = await _assess(_FRESH_FORM, _suff(7))
    assert readiness.verdict is None  # GO is never emitted on a stale record
    assert readiness.status.value == "degraded"
    assert "sync" in readiness.summary_text.lower()  # honest, names possible sync loss
    assert readiness.coverage is not None
    assert readiness.coverage.get("stale") is True
    assert readiness.coverage.get("staleness_days") == 7
    assert readiness.coverage.get("fidelity") == "partial"


async def test_record_past_hard_floor_abstains_for_any_verdict() -> None:
    """Beyond 2*tau_ATL, even a safe-side REST cannot be read -> truthful stale abstain."""
    readiness = await _assess(_FATIGUED_FORM, _suff(40))
    assert readiness.verdict is None
    assert readiness.status.value == "degraded"
    assert readiness.coverage is not None
    assert readiness.coverage.get("fidelity") == "degraded"


async def test_stale_safe_side_verdict_ships_but_degraded_and_caveated() -> None:
    """A less-aggressive REST on a merely-stale record still ships, DEGRADED + disclosed."""
    readiness = await _assess(_FATIGUED_FORM, _suff(7))
    assert readiness.verdict is ReadinessVerdict.REST  # safe-side verdict survives
    assert readiness.status.value == "degraded"
    assert STALE_DATA_CLAUSE in readiness.summary_text  # currency disclosed in the lead
    assert readiness.coverage is not None
    assert readiness.coverage.get("stale") is True


async def test_fresh_record_ships_go_completed_no_caveat() -> None:
    """A fresh (1-day) record reads the aggressive GO verbatim: COMPLETED, full fidelity."""
    readiness = await _assess(_FRESH_FORM, _suff(1))
    assert readiness.verdict is ReadinessVerdict.GO
    assert readiness.status.value == "completed"
    assert STALE_DATA_CLAUSE not in readiness.summary_text
    assert readiness.coverage is not None
    assert readiness.coverage.get("fidelity") == "full"
    assert "stale" not in readiness.coverage


async def test_substituted_fidelity_surfaces_without_blocking_verdict() -> None:
    """A fresh but HR-substituted record ships its verdict but discloses PARTIAL fidelity."""
    readiness = await _assess(_FRESH_FORM, _suff(1, substituted=True))
    assert readiness.verdict is ReadinessVerdict.GO
    assert readiness.coverage is not None
    assert readiness.coverage.get("substituted") is True
    assert readiness.coverage.get("fidelity") == "partial"


async def test_healthy_sync_taper_ships_go_even_with_old_observations() -> None:
    """A long gap with HEALTHY sync is a real taper: fresh form -> GO ships, no abstain, no caveat.

    The false-abstain regression guard — a raw activity-gap signal would wrongly block this. With
    ``sync_suspect=False`` the 40-day gap is trusted as genuine rest, so the GO verdict ships.
    """
    readiness = await _assess(_FRESH_FORM, _suff(40, sync_suspect=False))
    assert readiness.verdict is ReadinessVerdict.GO
    assert readiness.status.value == "completed"
    assert STALE_DATA_CLAUSE not in readiness.summary_text
    assert readiness.coverage is not None
    assert readiness.coverage.get("fidelity") == "full"


async def test_no_sufficiency_envelope_preserves_prior_behaviour() -> None:
    """Omitting the envelope (the inputs-only contract) leaves the verdict + status unchanged."""
    readiness = await _assess(_FRESH_FORM, None)
    assert readiness.verdict is ReadinessVerdict.GO
    assert readiness.status.value == "completed"
    assert readiness.coverage is not None
    assert "fidelity" not in readiness.coverage  # no sufficiency keys when no envelope
