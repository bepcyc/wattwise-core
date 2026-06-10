"""Golden-reference tests for cross-sport gating of the cycling-power family.

Pins the spec-mandated honest outcomes of the sport-applicability rule for the WHOLE
cycling-power family — NP/IF/power-TSS/W'balance, the mean-maximal-power curve
(``mmp``) and its derived ``best_effort``, and the per-activity bundle:

- **SPORT-T2 / ANL-R12** — a power-family metric requested on an ``activity`` whose
  ``sport`` has no mechanical power returns ``Unavailable(NOT_APPLICABLE_FOR_SPORT)`` —
  never ``0``, never a cross-sport surrogate, and DISTINCT from ``MISSING_REQUIRED_INPUT``
  (channel absent for a sport that COULD have it) and ``OUT_OF_DOMAIN``.
- **LM-R1 / LM-R2** — the per-activity bundle's load field is ``tss`` on the power path
  OR the labeled ``hr_load`` on the HR path; an HR-only activity yields a bundle carrying
  the HR-load value + ``duration_valid_s`` and ``load_model = hr_load`` (never an empty
  whole-bundle ``Unavailable``); a power-less HR-less activity has an ``Unavailable`` load
  field while still reporting a populated ``load_model``.

The inputs are valid, well-formed power/HR traces so the ONLY reason a power-family
metric is unavailable is the sport — proving the gate is on ``sport``, not on data.
"""

from __future__ import annotations

import numpy as np
import pytest

from wattwise_core.analytics.constants import MMP_DURATION_GRID_S
from wattwise_core.analytics.mmp_cp import best_effort, mmp
from wattwise_core.analytics.np_if_tss import (
    LOAD_MODEL_POWER_TSS,
    intensity_factor,
    load_metrics_bundle,
    normalized_power,
    power_tss,
)
from wattwise_core.analytics.result import Computed, Unavailable, UnavailableReason, is_computed
from wattwise_core.analytics.series import Stream
from wattwise_core.analytics.trimp import (
    LOAD_MODEL_HR_LOAD,
    banister_hr_load,
)
from wattwise_core.analytics.wbal import wbal

# A clean 120 s constant ride: a fully-valid power channel so the metric CAN compute —
# the only reason for unavailability under a non-power sport is the sport itself.
CONST_W = 250.0
RIDE_S = 120
FTP_W = 250.0
CP_W = 240.0
W_PRIME_J = 20000.0

# Sports without a true mechanical-power channel (SEED_SPORTS has_mechanical_power=False)
# plus a never-seen code; cycling/rowing carry mechanical power and are NOT gated here.
NON_POWER_SPORTS = ("running", "swimming", "xc_ski", "strength", "other", "made_up_sport")


def _power_stream() -> Stream:
    return Stream.from_values([CONST_W] * RIDE_S)


def _hr_stream() -> Stream:
    return Stream.from_values([140.0] * RIDE_S)


@pytest.mark.golden
@pytest.mark.parametrize("sport", NON_POWER_SPORTS)
def test_np_not_applicable_for_non_power_sport(sport: str) -> None:
    """SPORT-T2: NP on a non-power sport ⇒ NOT_APPLICABLE_FOR_SPORT (not a number)."""
    result = normalized_power(_power_stream(), sport=sport)
    assert isinstance(result, Unavailable)
    assert result.reason == UnavailableReason.NOT_APPLICABLE_FOR_SPORT


@pytest.mark.golden
@pytest.mark.parametrize("sport", NON_POWER_SPORTS)
def test_if_propagates_sport_mismatch(sport: str) -> None:
    """SPORT-T2: IF propagates the NP NOT_APPLICABLE_FOR_SPORT verbatim (IF-R1)."""
    result = intensity_factor(normalized_power(_power_stream(), sport=sport), FTP_W)
    assert isinstance(result, Unavailable)
    assert result.reason == UnavailableReason.NOT_APPLICABLE_FOR_SPORT


@pytest.mark.golden
@pytest.mark.parametrize("sport", NON_POWER_SPORTS)
def test_power_tss_propagates_sport_mismatch(sport: str) -> None:
    """SPORT-T2: power-TSS propagates the NP NOT_APPLICABLE_FOR_SPORT (no fabricated load)."""
    result = power_tss(normalized_power(_power_stream(), sport=sport), FTP_W, RIDE_S)
    assert isinstance(result, Unavailable)
    assert result.reason == UnavailableReason.NOT_APPLICABLE_FOR_SPORT


@pytest.mark.golden
@pytest.mark.parametrize("sport", NON_POWER_SPORTS)
def test_wbal_not_applicable_for_non_power_sport(sport: str) -> None:
    """SPORT-T2: W'balance on a non-power sport ⇒ NOT_APPLICABLE_FOR_SPORT."""
    result = wbal(_power_stream().values, CP_W, W_PRIME_J, sport=sport)
    assert isinstance(result, Unavailable)
    assert result.reason == UnavailableReason.NOT_APPLICABLE_FOR_SPORT


@pytest.mark.golden
@pytest.mark.parametrize("sport", NON_POWER_SPORTS)
def test_mmp_not_applicable_for_non_power_sport(sport: str) -> None:
    """SPORT-T2/ANL-R12: the WHOLE MMP curve on a non-power sport is per-duration gated.

    Every grid duration maps to ``Unavailable(NOT_APPLICABLE_FOR_SPORT)`` — never a
    Computed power number, never ``INSUFFICIENT_DATA`` (the window IS long enough), and
    DISTINCT from the ``MISSING_REQUIRED_INPUT`` an empty cycling channel would yield.
    Deleting the ``sport`` gate at the top of ``mmp`` would surface ``Computed`` peaks
    here (the trace is a valid 120 s power channel) and break this test.
    """
    results = mmp(_power_stream().values, MMP_DURATION_GRID_S, sport=sport)
    assert set(results) == {int(d) for d in MMP_DURATION_GRID_S}
    for res in results.values():
        assert isinstance(res, Unavailable)
        assert res.reason == UnavailableReason.NOT_APPLICABLE_FOR_SPORT


@pytest.mark.golden
@pytest.mark.parametrize("sport", NON_POWER_SPORTS)
def test_best_effort_not_applicable_for_non_power_sport(sport: str) -> None:
    """SPORT-T2/BEST-R1: best_effort (derived from MMP) inherits the sport gate."""
    res = best_effort(_power_stream().values, RIDE_S, sport=sport)
    assert isinstance(res, Unavailable)
    assert res.reason == UnavailableReason.NOT_APPLICABLE_FOR_SPORT


@pytest.mark.golden
def test_mmp_sport_mismatch_distinct_from_missing_required_input() -> None:
    """ANL-R12: MMP sport-mismatch (NOT_APPLICABLE) is DISTINCT from an absent channel.

    A cycling activity with NO power samples is ``MISSING_REQUIRED_INPUT`` (the sport
    COULD carry power); a running activity carrying a power channel is gated as
    ``NOT_APPLICABLE_FOR_SPORT`` (running is outside ``applicable_sports``).
    """
    d = RIDE_S
    missing = mmp(np.array([], dtype=np.float64), (d,), sport="cycling")[d]
    assert isinstance(missing, Unavailable)
    assert missing.reason == UnavailableReason.MISSING_REQUIRED_INPUT
    mismatch = mmp(_power_stream().values, (d,), sport="running")[d]
    assert isinstance(mismatch, Unavailable)
    assert mismatch.reason == UnavailableReason.NOT_APPLICABLE_FOR_SPORT


@pytest.mark.golden
def test_mmp_still_computes_for_cycling() -> None:
    """The gate does NOT block cycling: MMP/best_effort compute the real peak watts."""
    res = mmp(_power_stream().values, (RIDE_S,), sport="cycling")[RIDE_S]
    assert isinstance(res, Computed)
    assert res.value.mean_power_w == pytest.approx(CONST_W, abs=1e-6)
    be = best_effort(_power_stream().values, RIDE_S, sport="cycling")
    assert is_computed(be)


@pytest.mark.golden
def test_sport_mismatch_is_distinct_from_missing_required_input() -> None:
    """ANL-R12: sport-mismatch (NOT_APPLICABLE) is DISTINCT from absent input (MISSING)."""
    # Cycling but NO power channel: the sport COULD have power, the channel is absent.
    missing = normalized_power(Stream.from_values([]), sport="cycling")
    assert isinstance(missing, Unavailable)
    assert missing.reason == UnavailableReason.MISSING_REQUIRED_INPUT
    # Running WITH a (pod) power channel: the sport is not in the power family.
    mismatch = normalized_power(_power_stream(), sport="running")
    assert isinstance(mismatch, Unavailable)
    assert mismatch.reason == UnavailableReason.NOT_APPLICABLE_FOR_SPORT


@pytest.mark.golden
def test_cycling_power_family_still_computes() -> None:
    """The gate does NOT block the applicable sport: cycling still computes a real NP/TSS."""
    np_res = normalized_power(_power_stream(), sport="cycling")
    assert isinstance(np_res, Computed)
    assert np_res.value.np_w == pytest.approx(CONST_W, abs=1e-6)
    tss = power_tss(np_res, FTP_W, RIDE_S)
    assert isinstance(tss, Computed)
    assert tss.value > 0.0


@pytest.mark.golden
@pytest.mark.parametrize("sport", NON_POWER_SPORTS)
def test_bundle_power_fields_not_applicable_for_non_power_sport(sport: str) -> None:
    """ANL-R12/LM-R2: a non-power sport yields a bundle whose power fields are gated."""
    bundle = load_metrics_bundle(_power_stream(), _hr_stream(), FTP_W, CONST_W, 140.0, sport=sport)
    for field in (bundle.np, bundle.if_, bundle.tss, bundle.intensity_class):
        assert isinstance(field, Unavailable)
        assert field.reason == UnavailableReason.NOT_APPLICABLE_FOR_SPORT


@pytest.mark.golden
def test_bundle_hr_only_activity_surfaces_hr_load() -> None:
    """LM-R1/R2: an HR-only (power-less) cycling activity yields a bundle carrying hr_load.

    No power channel, but a clean HR trace with valid HR_max/HR_rest/sex: the load field
    is the labeled HR load (not the power TSS), ``load_model = hr_load``, and the power
    family is MISSING_REQUIRED_INPUT (channel absent) — never a whole-bundle absence.
    """
    hr_load = banister_hr_load(_hr_stream(), 190.0, 50.0, "male")
    assert isinstance(hr_load, Computed)  # the HR-load is genuinely computable
    bundle = load_metrics_bundle(
        Stream.from_values([]),  # no power channel
        _hr_stream(),
        None,
        None,
        140.0,
        sport="cycling",
        hr_load_result=hr_load,
    )
    assert bundle.load_model == LOAD_MODEL_HR_LOAD
    assert isinstance(bundle.hr_load, Computed)
    assert bundle.hr_load.value == pytest.approx(hr_load.value, abs=1e-9)
    # The power TSS is NOT relabeled as the HR load and is not both populated (LM-T2).
    assert isinstance(bundle.tss, Unavailable)
    assert bundle.tss.reason == UnavailableReason.MISSING_REQUIRED_INPUT


@pytest.mark.golden
def test_bundle_power_path_keeps_power_tss_label_and_no_hr_load() -> None:
    """LM-T2: a power activity carries load_model=power_tss; HR load is NOT also "the" load."""
    hr_load = banister_hr_load(_hr_stream(), 190.0, 50.0, "male")
    bundle = load_metrics_bundle(
        _power_stream(),
        _hr_stream(),
        FTP_W,
        CONST_W,
        140.0,
        sport="cycling",
        hr_load_result=hr_load,
    )
    assert bundle.load_model == LOAD_MODEL_POWER_TSS
    assert is_computed(bundle.tss)
    assert isinstance(bundle.hr_load, Unavailable)  # not both populated as "the" load


@pytest.mark.golden
def test_bundle_no_power_no_hr_load_unavailable_but_labeled() -> None:
    """LM-R2: neither power nor HR load ⇒ Unavailable load field with a populated load_model."""
    bundle = load_metrics_bundle(
        Stream.from_values([]),
        None,
        None,
        None,
        None,
        sport="cycling",
        hr_load_result=None,
    )
    assert isinstance(bundle.tss, Unavailable)
    assert isinstance(bundle.hr_load, Unavailable)
    assert bundle.load_model in {LOAD_MODEL_POWER_TSS, LOAD_MODEL_HR_LOAD}
