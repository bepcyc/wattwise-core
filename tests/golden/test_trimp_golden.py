"""Golden-reference tests for Banister-HRR TRIMP / HR-load (doc 40 §7D, TRIMP-T1).

TEST-R1/R4: the expected values below are HAND-DERIVED from the published
Banister-HRR formula (doc 40 §3) with a tiny, fully-enumerated HR series and known
sex constants. The derivation is recorded inline so the fixture origin is explicit.

Fixture origin / citation
-------------------------
Formula (TRIMP-R1, doc 40 §3)::

    HRR(t) = (HR(t) - HR_rest) / (HR_max - HR_rest)
    TRIMP  = Σ over valid seconds  Δt_min · HRR(t) · a · e^(b · HRR(t))

with Δt_min = 1/60 (1 Hz) and the published male Banister constants a = 0.64,
b = 1.92 (Banister 1991; doc 40 §3 / constants.py TRIMP_A_MALE / TRIMP_B_MALE).

Hand derivation of the Banister golden (male)
---------------------------------------------
HR series (1 Hz, 4 contiguous seconds): [120, 140, 160, 180] bpm.
HR_rest = 60, HR_max = 200  ->  reserve = 140.  All HRR lie within [0, 1] (no clamp).

    HR=120 -> HRR = 60/140  = 0.428571...  term = (1/60)·HRR·0.64·e^(1.92·HRR) = 0.010409125742
    HR=140 -> HRR = 80/140  = 0.571428...  term =                                0.018258864413
    HR=160 -> HRR = 100/140 = 0.714285...  term =                                0.030026488706
    HR=180 -> HRR = 120/140 = 0.857142...  term =                                0.047403080689
    TRIMP  = Σ terms = 0.10609755954890741

Hand derivation of the zonal golden (TRIMP-R2)
----------------------------------------------
Same HR series. zone_boundaries = [130, 150, 170] (4 zones), weights = [1, 2, 3, 4].
np.digitize(..., right=False): 120->z0, 140->z1, 160->z2, 180->z3 (1 s each).
minutes-in-zone = 1/60 each; load = (1·1 + 1·2 + 1·3 + 1·4)/60 = 10/60 = 0.16666666666666666.
"""

from __future__ import annotations

import math

import pytest

from wattwise_core.analytics.result import Computed
from wattwise_core.analytics.series import Stream
from wattwise_core.analytics.trimp import (
    LOAD_MODEL_HR_LOAD,
    LOAD_MODEL_HR_LOAD_ZONAL,
    banister_hr_load,
    hr_load_zonal,
)

# Declared golden tolerance for this closed-form sum (ANL-R31): abs/rel 1e-9.
_TOL = 1e-9

_HR_SERIES = [120.0, 140.0, 160.0, 180.0]
_HR_REST = 60.0
_HR_MAX = 200.0

_EXPECTED_BANISTER_MALE = 0.10609755954890741
_EXPECTED_ZONAL = 0.16666666666666666


@pytest.mark.golden
def test_banister_hr_load_golden_male() -> None:
    stream = Stream.from_values([float(x) for x in _HR_SERIES])
    result = banister_hr_load(stream, _HR_MAX, _HR_REST, "male", sport="cycling")

    assert isinstance(result, Computed)
    assert math.isclose(result.value, _EXPECTED_BANISTER_MALE, rel_tol=_TOL, abs_tol=_TOL)
    # Label honesty (TRIMP-R2/R4): canonical Banister carries load_model='hr_load'.
    assert result.quality.extra["load_model"] == LOAD_MODEL_HR_LOAD
    assert result.quality.extra["clamped_hrr_fraction"] == 0.0
    assert result.quality.extra["sex_neutral_constants"] is False
    assert result.quality.extra["trimp_a"] == 0.64
    assert result.quality.extra["trimp_b"] == 1.92
    assert result.quality.confidence == 1.0
    assert result.provenance.channels == ("heart_rate",)


@pytest.mark.golden
def test_banister_hr_load_golden_female_uses_distinct_constants() -> None:
    # Independent hand check for the female pair (a=0.86, b=1.67): same series,
    # the multiplicative a and exponential b must NOT be conflated (TRIMP-R1).
    a, b = 0.86, 1.67
    reserve = _HR_MAX - _HR_REST
    expected = sum(
        (1.0 / 60.0) * ((h - _HR_REST) / reserve) * a * math.exp(b * ((h - _HR_REST) / reserve))
        for h in _HR_SERIES
    )
    stream = Stream.from_values([float(x) for x in _HR_SERIES])
    result = banister_hr_load(stream, _HR_MAX, _HR_REST, "female")

    assert isinstance(result, Computed)
    assert math.isclose(result.value, expected, rel_tol=_TOL, abs_tol=_TOL)
    # Different sex => different value (constants genuinely applied, not stubbed).
    assert not math.isclose(result.value, _EXPECTED_BANISTER_MALE, abs_tol=1e-6)


@pytest.mark.golden
def test_hr_load_zonal_golden() -> None:
    stream = Stream.from_values([float(x) for x in _HR_SERIES])
    result = hr_load_zonal(stream, [130.0, 150.0, 170.0], [1.0, 2.0, 3.0, 4.0])

    assert isinstance(result, Computed)
    assert math.isclose(result.value, _EXPECTED_ZONAL, rel_tol=_TOL, abs_tol=_TOL)
    # Distinct label, never relabelled as hr_load or power_tss (TRIMP-R2/R4).
    assert result.quality.extra["load_model"] == LOAD_MODEL_HR_LOAD_ZONAL
    assert result.quality.extra["seconds_in_zone"] == (1.0, 1.0, 1.0, 1.0)
