"""Property-based tests for Banister-HRR TRIMP / HR-load (doc 40 §7D, TRIMP-T1..T5).

Covers, per the per-metric property list (doc 40 §11.1):

* TRIMP-T1 — golden Banister-HRR (replayed here as a closed-form oracle).
* TRIMP-T2 — invalid values (HR_max <= HR_rest, non-finite) -> OUT_OF_DOMAIN.
* TRIMP-T3 — missing input (absent HR / HR_max / HR_rest) -> MISSING_REQUIRED_INPUT.
* TRIMP-T4 — monotonic in intensity (higher HR everywhere => not-lower TRIMP).
* TRIMP-T5 — variant labelling (hr_load vs hr_load_zonal, never relabelled);
             out-of-band HRR sample clamp reported in QualityReport.

Plus the fail-closed degenerate cases (TEST-R3): empty stream, all-``null`` stream,
sub-data, and sex-required-but-absent. Determinism (ANL-R30) is checked by repeated
evaluation. Generators follow TEST-R2 (variable-length 1 Hz streams, ``null`` gaps,
realistic athlete param ranges, with shrinking).
"""

from __future__ import annotations

import math

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from wattwise_core.analytics.result import (
    Computed,
    Unavailable,
    UnavailableReason,
)
from wattwise_core.analytics.series import Stream, resample_to_1hz
from wattwise_core.analytics.trimp import (
    LOAD_MODEL_HR_LOAD,
    LOAD_MODEL_HR_LOAD_ZONAL,
    TRIMP_A_SEX_NEUTRAL,
    TRIMP_B_SEX_NEUTRAL,
    banister_hr_load,
    hr_load_zonal,
)

_TOL = 1e-9


# --- generators (TEST-R2) -------------------------------------------------------

# HR samples in a realistic range, with None marking gaps (ANL-R7).
_hr_sample = st.one_of(
    st.none(),
    st.floats(min_value=40.0, max_value=210.0, allow_nan=False, allow_infinity=False),
)
_hr_list = st.lists(_hr_sample, min_size=1, max_size=120)


def _oracle_banister(
    stream: Stream, hr_max: float, hr_rest: float, a: float, b: float
) -> float:
    """Independent closed-form oracle for the Banister-HRR sum at 1 Hz.

    Computed over the SAME resampled 1 Hz grid the implementation uses (ANL-R8), so
    interpolated gap-bridge seconds are accounted for identically.
    """
    reserve = hr_max - hr_rest
    grid = resample_to_1hz(stream)
    total = 0.0
    for h in grid:
        if math.isnan(h):
            continue
        hrr = min(1.0, max(0.0, (h - hr_rest) / reserve))
        total += (1.0 / 60.0) * hrr * a * math.exp(b * hrr)
    return total


# --- TRIMP-T1: golden / closed-form oracle agreement ----------------------------


@settings(max_examples=200)
@given(
    hr_values=_hr_list,
    hr_rest=st.floats(min_value=30.0, max_value=70.0),
    reserve=st.floats(min_value=60.0, max_value=150.0),
    sex=st.sampled_from(["male", "female"]),
)
def test_banister_matches_independent_oracle(
    hr_values: list[float | None], hr_rest: float, reserve: float, sex: str
) -> None:
    hr_max = hr_rest + reserve
    a, b = (0.64, 1.92) if sex == "male" else (0.86, 1.67)
    stream = Stream.from_values(hr_values)
    result = banister_hr_load(stream, hr_max, hr_rest, sex)

    # No valid second after resampling -> INSUFFICIENT_DATA (fail-closed, TEST-R3).
    grid = resample_to_1hz(stream)
    if not np.any(~np.isnan(grid)):
        assert isinstance(result, Unavailable)
        assert result.reason is UnavailableReason.INSUFFICIENT_DATA
        return

    assert isinstance(result, Computed)
    expected = _oracle_banister(stream, hr_max, hr_rest, a, b)
    assert math.isclose(result.value, expected, rel_tol=1e-9, abs_tol=1e-9)
    assert result.value >= 0.0  # HRR>=0, kernel>=0 => non-negative load.


# --- TRIMP-T2: invalid values -> OUT_OF_DOMAIN ----------------------------------


@settings(max_examples=100)
@given(
    hr_values=_hr_list,
    hr_rest=st.floats(min_value=40.0, max_value=200.0),
    delta=st.floats(min_value=0.0, max_value=80.0),  # hr_max <= hr_rest
)
def test_nonpositive_reserve_out_of_domain(
    hr_values: list[float | None], hr_rest: float, delta: float
) -> None:
    hr_max = hr_rest - delta  # hr_max <= hr_rest -> non-positive reserve
    result = banister_hr_load(Stream.from_values(hr_values), hr_max, hr_rest, "male")
    assert isinstance(result, Unavailable)
    assert result.reason is UnavailableReason.OUT_OF_DOMAIN


@settings(max_examples=50)
@given(hr_values=_hr_list, bad=st.sampled_from([math.inf, -math.inf, math.nan]))
def test_nonfinite_params_out_of_domain(
    hr_values: list[float | None], bad: float
) -> None:
    # Non-finite HR_max or HR_rest violates the domain precondition (ANL-R32).
    r1 = banister_hr_load(Stream.from_values(hr_values), bad, 60.0, "male")
    r2 = banister_hr_load(Stream.from_values(hr_values), 200.0, bad, "male")
    assert isinstance(r1, Unavailable) and r1.reason is UnavailableReason.OUT_OF_DOMAIN
    assert isinstance(r2, Unavailable) and r2.reason is UnavailableReason.OUT_OF_DOMAIN


# --- TRIMP-T3: missing input -> MISSING_REQUIRED_INPUT --------------------------


def test_missing_hr_stream_missing_required_input() -> None:
    r = banister_hr_load(None, 200.0, 60.0, "male")
    assert isinstance(r, Unavailable)
    assert r.reason is UnavailableReason.MISSING_REQUIRED_INPUT


def test_missing_hr_max_or_rest_missing_required_input() -> None:
    s = Stream.from_values([100.0, 120.0, 140.0])
    for r in (
        banister_hr_load(s, None, 60.0, "male"),
        banister_hr_load(s, 200.0, None, "male"),
    ):
        assert isinstance(r, Unavailable)
        assert r.reason is UnavailableReason.MISSING_REQUIRED_INPUT


def test_sex_required_but_absent_missing_required_input() -> None:
    # Per the config flag: require_sex -> absent sex is MISSING_REQUIRED_INPUT.
    s = Stream.from_values([100.0, 120.0, 140.0])
    r = banister_hr_load(s, 200.0, 60.0, None, require_sex=True)
    assert isinstance(r, Unavailable)
    assert r.reason is UnavailableReason.MISSING_REQUIRED_INPUT


def test_sex_absent_default_pair_reduced_confidence() -> None:
    # Default policy: sex-neutral pair, recorded, reduced confidence (TRIMP-R1).
    s = Stream.from_values([100.0, 120.0, 140.0])
    r = banister_hr_load(s, 200.0, 60.0, None)
    assert isinstance(r, Computed)
    assert r.quality.extra["sex_neutral_constants"] is True
    assert r.quality.extra["trimp_a"] == TRIMP_A_SEX_NEUTRAL
    assert r.quality.extra["trimp_b"] == TRIMP_B_SEX_NEUTRAL
    assert r.quality.confidence < 1.0


# --- TRIMP-T4: monotonic in intensity -------------------------------------------


@settings(max_examples=150)
@given(
    base=st.lists(
        st.floats(min_value=70.0, max_value=170.0), min_size=1, max_size=60
    ),
    bump=st.floats(min_value=0.0, max_value=25.0),
    hr_rest=st.floats(min_value=40.0, max_value=60.0),
    reserve=st.floats(min_value=100.0, max_value=140.0),
)
def test_monotonic_in_intensity(
    base: list[float], bump: float, hr_rest: float, reserve: float
) -> None:
    # Raising HR everywhere (still within the reserve, no clamping flips) must not
    # lower TRIMP: the kernel HRR·a·e^(b·HRR) is increasing in HRR on [0,1].
    hr_max = hr_rest + reserve
    higher = [min(hr_max, h + bump) for h in base]
    lower_r = banister_hr_load(Stream.from_values(base), hr_max, hr_rest, "male")
    higher_r = banister_hr_load(Stream.from_values(higher), hr_max, hr_rest, "male")
    assert isinstance(lower_r, Computed)
    assert isinstance(higher_r, Computed)
    assert higher_r.value >= lower_r.value - 1e-9


# --- TRIMP-T5: out-of-band clamp reported + variant labelling -------------------


@settings(max_examples=100)
@given(
    n_low=st.integers(min_value=0, max_value=20),
    n_mid=st.integers(min_value=1, max_value=20),
    n_high=st.integers(min_value=0, max_value=20),
)
def test_out_of_band_clamp_reported(n_low: int, n_mid: int, n_high: int) -> None:
    hr_rest, hr_max = 60.0, 200.0  # reserve 140
    # Below-rest (HRR<0) and above-max (HRR>1) samples are out-of-band.
    below = [40.0] * n_low  # HR < hr_rest -> HRR < 0
    mid = [130.0] * n_mid  # within band
    above = [230.0] * n_high  # HR > hr_max -> HRR > 1
    values: list[float | None] = [*below, *mid, *above]
    result = banister_hr_load(Stream.from_values(values), hr_max, hr_rest, "male")
    assert isinstance(result, Computed)
    n_valid = n_low + n_mid + n_high
    expected_fraction = (n_low + n_high) / n_valid
    assert math.isclose(
        float(result.quality.extra["clamped_hrr_fraction"]),  # type: ignore[arg-type]
        expected_fraction,
        abs_tol=_TOL,
    )


def test_variant_labels_distinct() -> None:
    s = Stream.from_values([100.0, 120.0, 140.0, 160.0])
    ban = banister_hr_load(s, 200.0, 60.0, "male")
    zon = hr_load_zonal(s, [130.0, 150.0], [1.0, 2.0, 3.0])
    assert isinstance(ban, Computed) and isinstance(zon, Computed)
    assert ban.quality.extra["load_model"] == LOAD_MODEL_HR_LOAD
    assert zon.quality.extra["load_model"] == LOAD_MODEL_HR_LOAD_ZONAL
    # The two labels are never the same token, and neither is power_tss.
    assert LOAD_MODEL_HR_LOAD != LOAD_MODEL_HR_LOAD_ZONAL
    assert LOAD_MODEL_HR_LOAD != "power_tss"
    assert LOAD_MODEL_HR_LOAD_ZONAL != "power_tss"


# --- zonal-specific fail-closed + oracle ----------------------------------------


def test_zonal_missing_inputs_missing_required_input() -> None:
    s = Stream.from_values([100.0, 120.0])
    for r in (
        hr_load_zonal(None, [130.0], [1.0, 2.0]),
        hr_load_zonal(s, None, [1.0, 2.0]),
        hr_load_zonal(s, [130.0], None),
    ):
        assert isinstance(r, Unavailable)
        assert r.reason is UnavailableReason.MISSING_REQUIRED_INPUT


def test_zonal_bad_shapes_out_of_domain() -> None:
    s = Stream.from_values([100.0, 120.0])
    bad_cases = [
        ([], [1.0]),  # empty boundaries
        ([150.0, 130.0], [1.0, 2.0, 3.0]),  # non-ascending
        ([130.0, 150.0], [1.0, 2.0]),  # weights length mismatch (need 3)
        ([130.0], [1.0, math.inf]),  # non-finite weight
        ([math.nan], [1.0, 2.0]),  # non-finite boundary
    ]
    for bounds, wts in bad_cases:
        r = hr_load_zonal(s, bounds, wts)
        assert isinstance(r, Unavailable)
        assert r.reason is UnavailableReason.OUT_OF_DOMAIN


@settings(max_examples=150)
@given(
    hr_values=_hr_list,
    weights=st.lists(
        st.floats(min_value=0.0, max_value=5.0), min_size=4, max_size=4
    ),
)
def test_zonal_matches_oracle(
    hr_values: list[float | None], weights: list[float]
) -> None:
    bounds = [120.0, 150.0, 180.0]  # 4 zones
    stream = Stream.from_values(hr_values)
    result = hr_load_zonal(stream, bounds, weights)
    grid = resample_to_1hz(stream)
    valid = grid[~np.isnan(grid)]
    if valid.size == 0:
        assert isinstance(result, Unavailable)
        assert result.reason is UnavailableReason.INSUFFICIENT_DATA
        return
    assert isinstance(result, Computed)
    # Oracle: digitize then weighted minutes-in-zone over the resampled grid.
    zone_idx = np.digitize(valid, np.asarray(bounds), right=False)
    counts = np.bincount(zone_idx, minlength=4).astype(float)
    expected = float(np.dot(counts / 60.0, np.asarray(weights)))
    assert math.isclose(result.value, expected, rel_tol=1e-9, abs_tol=1e-9)


# --- determinism (ANL-R30) ------------------------------------------------------


@settings(max_examples=50)
@given(hr_values=st.lists(_hr_sample, min_size=1, max_size=40))
def test_deterministic(hr_values: list[float | None]) -> None:
    s = Stream.from_values(hr_values)
    r1 = banister_hr_load(s, 200.0, 60.0, "male")
    r2 = banister_hr_load(s, 200.0, 60.0, "male")
    assert type(r1) is type(r2)
    if isinstance(r1, Computed) and isinstance(r2, Computed):
        assert r1.value == r2.value
