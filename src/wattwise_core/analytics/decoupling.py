"""Aerobic decoupling — Pw:Hr (cycling) / Pa:Hr (pace) drift (doc 40 §9, DEC-R1..R5).

Aerobic decoupling is the percentage drift in *output-per-heartbeat* between the
first and second halves of a steady effort: how much more (or less) heart rate it
costs to hold the same mechanical output (power) or speed (pace) late in the effort.

Contract (the exact spec mandates this module implements):

- **DEC-R1** — the half boundary is the midpoint of elapsed *time*
  ``t_mid = (t_start + t_end) / 2`` (NOT the sample-count midpoint); a sample is in
  the first half iff ``t <= t_mid``.
- **DEC-R2** — coasting/zero exclusion is mandatory and applied *after* the time
  split: drop ``output == 0`` seconds (cycling) / non-moving seconds (pace) from
  BOTH the output and the HR means. Each half needs a declared minimum number of
  *included* (non-coasting) samples, else ``Unavailable(INSUFFICIENT_DATA)``.
- **DEC-R3** — the output term uses 30 s smoothed / NP-style power (the declared
  smoothing window), never the raw 1 Hz signal.
- **DEC-R4** — requires synchronized output-or-pace AND HR over a declared minimum
  duration (default 20 min) plus a steadiness gate; a missing channel / too-short /
  too-variable effort fails closed with the matching :class:`UnavailableReason`.
- **DEC-R5** — properties: constant power + HR ⇒ 0 %; coasting-invariance (inserting
  ``output == 0`` samples does not change the result); sign convention (a second-half
  efficiency *drop* ⇒ positive decoupling); time-midpoint split.

Engine-wide invariants also bind here: pure deterministic function (ANL-R2/R30); a
typed :class:`MetricResult` envelope, never a bare number (ANL-R3); fail-closed with
the exact reason (ANL-R4, doc 40 §6); no NaN/Inf in a ``Computed`` (ANL-R32 ⇒
``OUT_OF_DOMAIN``); every ``Computed`` carries a :class:`QualityReport` +
:class:`InputLineage` (ANL-R5/R33). Aerobic decoupling is *sport-parameterized*
(doc 40 §5): the output channel is power for cycling, speed for pace — selected by
``sport`` metadata, never by a source-name branch (ANL-R11/R13).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt

from wattwise_core.analytics.constants import (
    DECOUPLING_MIN_DURATION_S,
    DECOUPLING_SMOOTHING_WINDOW_S,
    MAX_INTERP_GAP_S,
)
from wattwise_core.analytics.result import (
    Computed,
    InputLineage,
    MetricResult,
    QualityReport,
    Unavailable,
    UnavailableReason,
)
from wattwise_core.analytics.series import (
    FloatArray,
    Stream,
    resample_to_1hz,
    trailing_rolling_mean,
)

BoolArray = npt.NDArray[np.bool_]

# Sports for which a *power* output channel is the decoupling numerator (Pw:Hr).
_POWER_SPORTS: Final[frozenset[str]] = frozenset({"cycling", "ride", "virtualride"})
# Sports for which a *speed* output channel is the decoupling numerator (Pa:Hr).
_PACE_SPORTS: Final[frozenset[str]] = frozenset(
    {"running", "run", "virtualrun", "walking", "walk", "hiking", "hike"}
)

# ANL-R11/ANL-R13: decoupling is sport-PARAMETERIZED on its output channel — defined for
# any sport with a declared output mapping (power for _POWER_SPORTS, speed/pace for
# _PACE_SPORTS). The set is derived from the SAME per-sport mappings consumed by
# ``_output_channel_for_sport`` (the NOT_APPLICABLE_FOR_SPORT gate), so the declaration
# can never drift from the gate.
APPLICABLE_SPORTS: Final[frozenset[str]] = _POWER_SPORTS | _PACE_SPORTS

# Declared minimum number of included (non-coasting) samples per half (DEC-R2).
# Below this a half is statistically meaningless and the metric fails closed.
MIN_INCLUDED_SAMPLES_PER_HALF: Final = 60

# Steadiness gate (DEC-R4): the effort must be aerobically *steady*. We gate on the
# coefficient of variation of the smoothed output over the whole included window; an
# interval / sprint workout (huge CV) is not a valid decoupling target.
STEADINESS_MAX_OUTPUT_CV: Final = 0.50


def _output_channel_for_sport(sport: str) -> str | None:
    """Return ``"power"`` / ``"speed"`` for the sport, or ``None`` if inapplicable."""
    key = sport.strip().lower().replace(" ", "")
    if key in _POWER_SPORTS:
        return "power"
    if key in _PACE_SPORTS:
        return "speed"
    return None


@dataclass(frozen=True, slots=True)
class _Window:
    """Validated common-grid analysis window (both channels resampled to 1 Hz)."""

    channel: str
    out_1hz: FloatArray
    hr_1hz: FloatArray
    both_valid: BoolArray
    t_start: float
    t_end: float
    elapsed_s: float


def _prepare_window(
    output_stream: Stream,
    hr_stream: Stream,
    sport: str,
    min_duration_s: int,
    max_interp_gap_s: float,
) -> _Window | Unavailable:
    """Validate inputs and build the common-grid window, or fail closed (doc 40 §6).

    Returns a :class:`_Window` on success, else a typed :class:`Unavailable` with the
    exact reason: sport mismatch ⇒ ``NOT_APPLICABLE_FOR_SPORT``; a wholly absent
    output/HR channel or never-overlapping channels ⇒ ``MISSING_REQUIRED_INPUT``;
    an overlap window shorter than ``min_duration_s`` ⇒ ``INSUFFICIENT_DATA``.
    """
    channel = _output_channel_for_sport(sport)
    if channel is None:
        return Unavailable(
            UnavailableReason.NOT_APPLICABLE_FOR_SPORT,
            f"aerobic decoupling has no output channel for sport {sport!r}",
        )

    # A wholly absent channel is MISSING_REQUIRED_INPUT (doc 40 §6).
    if output_stream.values.size == 0 or not np.any(~np.isnan(output_stream.values)):
        return Unavailable(
            UnavailableReason.MISSING_REQUIRED_INPUT,
            f"no valid {channel} ({channel}-vs-HR) samples in output stream",
        )
    if hr_stream.values.size == 0 or not np.any(~np.isnan(hr_stream.values)):
        return Unavailable(
            UnavailableReason.MISSING_REQUIRED_INPUT,
            "no valid heart-rate samples in HR stream",
        )

    # Resample both onto a common 1 Hz grid (ANL-R8); align to the shorter grid.
    out_1hz = resample_to_1hz(output_stream, max_interp_gap_s=max_interp_gap_s)
    hr_1hz = resample_to_1hz(hr_stream, max_interp_gap_s=max_interp_gap_s)
    n = min(out_1hz.size, hr_1hz.size)
    out_1hz = out_1hz[:n]
    hr_1hz = hr_1hz[:n]
    both_valid = (~np.isnan(out_1hz)) & (~np.isnan(hr_1hz))
    if not np.any(both_valid):
        return Unavailable(
            UnavailableReason.MISSING_REQUIRED_INPUT,
            "no second has both output and HR valid (channels never overlap)",
        )

    # The analysis window spans the first..last second with both channels valid.
    valid_idx = np.flatnonzero(both_valid)
    t_start = float(valid_idx[0])
    t_end = float(valid_idx[-1])
    elapsed_s = t_end - t_start + 1.0
    if elapsed_s < float(min_duration_s):
        return Unavailable(
            UnavailableReason.INSUFFICIENT_DATA,
            f"analysis window {elapsed_s:.0f}s < required {min_duration_s}s (DEC-R4)",
        )

    return _Window(
        channel=channel,
        out_1hz=out_1hz,
        hr_1hz=hr_1hz,
        both_valid=both_valid,
        t_start=t_start,
        t_end=t_end,
        elapsed_s=elapsed_s,
    )


def _half_efficiency(
    smoothed_output: FloatArray,
    hr_1hz: FloatArray,
    included: BoolArray,
    in_half: BoolArray,
) -> tuple[float, int] | None:
    """Mean(included smoothed output) / mean(included HR) for one half (DEC-R1/R2).

    ``included`` and ``in_half`` are boolean masks over the common 1 Hz grid. Returns
    ``(efficiency, n_included)`` or ``None`` when too few samples / a non-positive
    HR mean would make the ratio undefined.
    """
    mask = included & in_half
    n = int(np.count_nonzero(mask))
    if n < MIN_INCLUDED_SAMPLES_PER_HALF:
        return None
    mean_out = float(np.mean(smoothed_output[mask]))
    mean_hr = float(np.mean(hr_1hz[mask]))
    if not (np.isfinite(mean_out) and np.isfinite(mean_hr)) or mean_hr <= 0.0:
        return None
    eff = mean_out / mean_hr
    if not np.isfinite(eff):
        return None
    return eff, n


@dataclass(frozen=True, slots=True)
class _HalfEfficiencies:
    """Per-half efficiencies + included-sample counts + steadiness CV for the window."""

    eff_first: float
    eff_second: float
    n_first: int
    n_second: int
    cv: float
    t_mid: float
    n_coasting: int


def _half_efficiencies(win: _Window, smoothing_window_s: int) -> _HalfEfficiencies | Unavailable:
    """Coasting-exclude, 30 s smooth, steadiness-gate, split, and compute both halves.

    Returns a :class:`_HalfEfficiencies` or a typed :class:`Unavailable`:
    no fully-seeded included second / a half with too few included samples ⇒
    ``INSUFFICIENT_DATA``; a non-steady effort ⇒ ``INSUFFICIENT_DATA``; a non-positive
    mean output ⇒ ``OUT_OF_DOMAIN`` (DEC-R2/R3/R4).
    """
    out_1hz = win.out_1hz
    hr_1hz = win.hr_1hz
    n = out_1hz.size

    # Coasting mask (DEC-R2): raw output == 0 is coasting/non-moving. Built from raw
    # 1 Hz output (pre-smoothing) so it is invariant to where a coasting second sits;
    # coasting seconds are dropped before smoothing too, so an inserted output==0
    # second cannot pollute the moving signal (coasting-invariance, DEC-R5).
    raw_moving = win.both_valid & (out_1hz != 0.0)

    # Smoothed output (DEC-R3): 30 s rolling mean of the MOVING signal. Compress to
    # moving-only seconds, smooth, then scatter back, so the smoothed value at each
    # moving second is a function only of the surrounding moving signal.
    moving_idx = np.flatnonzero(raw_moving)
    smoothed_moving = trailing_rolling_mean(out_1hz[moving_idx], smoothing_window_s)
    smoothed_output = np.full(n, np.nan, dtype=np.float64)
    smoothed_output[moving_idx] = smoothed_moving

    # A second is *included* only when it is moving AND its smoothed output is seeded.
    included = (~np.isnan(smoothed_output)) & raw_moving
    if not np.any(included):
        return Unavailable(
            UnavailableReason.INSUFFICIENT_DATA,
            f"no included second has a fully-seeded {smoothing_window_s}s smoothed "
            "output (effort too short after coasting exclusion)",
        )

    # Steadiness gate (DEC-R4): reject non-steady (interval) efforts.
    incl_smoothed = smoothed_output[included]
    mean_incl = float(np.mean(incl_smoothed))
    if not np.isfinite(mean_incl) or mean_incl <= 0.0:
        return Unavailable(
            UnavailableReason.OUT_OF_DOMAIN,
            "mean smoothed output over included window is non-positive/non-finite",
        )
    cv = float(np.std(incl_smoothed) / mean_incl)
    if not np.isfinite(cv) or cv > STEADINESS_MAX_OUTPUT_CV:
        return Unavailable(
            UnavailableReason.INSUFFICIENT_DATA,
            f"effort too variable (output CV {cv:.3f} > {STEADINESS_MAX_OUTPUT_CV}); "
            "not a steady aerobic effort (DEC-R4)",
        )

    # Time-midpoint split (DEC-R1): first half iff t <= t_mid.
    t = np.arange(n, dtype=np.float64)
    t_mid = (win.t_start + win.t_end) / 2.0
    in_first = t <= t_mid
    first = _half_efficiency(smoothed_output, hr_1hz, included, in_first)
    second = _half_efficiency(smoothed_output, hr_1hz, included, ~in_first)
    if first is None or second is None:
        return Unavailable(
            UnavailableReason.INSUFFICIENT_DATA,
            f"a half has < {MIN_INCLUDED_SAMPLES_PER_HALF} included samples after "
            "coasting exclusion (DEC-R2)",
        )

    eff_first, n_first = first
    eff_second, n_second = second
    return _HalfEfficiencies(
        eff_first=eff_first,
        eff_second=eff_second,
        n_first=n_first,
        n_second=n_second,
        cv=cv,
        t_mid=t_mid,
        n_coasting=int(np.count_nonzero(win.both_valid & ~raw_moving)),
    )


def _decoupling_pct(halves: _HalfEfficiencies) -> float | Unavailable:
    """Compute the decoupling % from per-half efficiencies, or fail closed (DEC-R1).

    ``((eff_first - eff_second) / eff_first) * 100`` — positive when second-half
    efficiency drops (DEC-R5). A zero first-half efficiency or a non-finite result
    is ``OUT_OF_DOMAIN`` (ANL-R32).
    """
    if halves.eff_first == 0.0:
        return Unavailable(
            UnavailableReason.OUT_OF_DOMAIN,
            "first-half efficiency is zero; decoupling ratio undefined",
        )
    decoupling_pct = ((halves.eff_first - halves.eff_second) / halves.eff_first) * 100.0
    if not np.isfinite(decoupling_pct):
        return Unavailable(
            UnavailableReason.OUT_OF_DOMAIN,
            "decoupling percentage is non-finite (ANL-R32)",
        )
    return decoupling_pct


def _build_decoupling_result(
    decoupling_pct: float,
    win: _Window,
    halves: _HalfEfficiencies,
    sport: str,
    min_duration_s: int,
    max_interp_gap_s: float,
    smoothing_window_s: int,
) -> Computed[float]:
    """Assemble the ``Computed`` envelope (QualityReport + InputLineage) for a result.

    Coverage is included-samples / analysis-window seconds (ANL-R5); the gap count is
    the number of seconds inside the window missing one/both channels.
    """
    n_included = halves.n_first + halves.n_second
    coverage = n_included / win.elapsed_s
    # Gaps inside the analysis window: seconds with one/both channels invalid.
    window_slice = win.both_valid[int(win.t_start) : int(win.t_end) + 1]
    quality = QualityReport(
        coverage_fraction=min(1.0, coverage),
        sample_rate_hz=1.0,
        gap_count=int(np.count_nonzero(~window_slice)),
        confidence=1.0,
        extra={
            "output_channel": win.channel,
            "smoothing_window_s": smoothing_window_s,
            "analysis_window_s": float(win.elapsed_s),
            "t_mid_s": float(halves.t_mid),
            "included_samples_first_half": halves.n_first,
            "included_samples_second_half": halves.n_second,
            "coasting_samples_excluded": halves.n_coasting,
            "eff_first_half": halves.eff_first,
            "eff_second_half": halves.eff_second,
            "output_cv": halves.cv,
        },
    )
    lineage = InputLineage(
        sport=sport,
        channels=(win.channel, "heart_rate"),
        reference_params={
            "min_duration_s": min_duration_s,
            "smoothing_window_s": smoothing_window_s,
            "max_interp_gap_s": max_interp_gap_s,
        },
    )
    return Computed(value=decoupling_pct, quality=quality, provenance=lineage)


def aerobic_decoupling(
    output_stream: Stream,
    hr_stream: Stream,
    sport: str,
    *,
    min_duration_s: int = DECOUPLING_MIN_DURATION_S,
    max_interp_gap_s: float = MAX_INTERP_GAP_S,
    smoothing_window_s: int = DECOUPLING_SMOOTHING_WINDOW_S,
) -> MetricResult[float]:
    """Aerobic decoupling % over a steady effort (doc 40 §9, DEC-R1..R5).

    ``output_stream`` carries mechanical power (cycling) or speed (pace) per
    ``sport``; ``hr_stream`` carries heart rate. Both are resampled to a common 1 Hz
    grid (ANL-R8); the analysis window is the seconds where BOTH channels are valid.
    The window is split at the elapsed-time midpoint (DEC-R1); coasting seconds
    (raw ``output == 0``) are excluded from both means *after* the split (DEC-R2);
    the output term is 30 s smoothed (DEC-R3). The result is
    ``((eff_first - eff_second) / eff_first) * 100`` (DEC-R1) — positive when
    second-half efficiency drops (DEC-R5).

    Fails closed with the exact reason (ANL-R4, doc 40 §6):

    - sport without a power/speed decoupling channel ⇒ ``NOT_APPLICABLE_FOR_SPORT``;
    - a wholly absent output or HR channel ⇒ ``MISSING_REQUIRED_INPUT``;
    - overlapping valid window shorter than ``min_duration_s``, or a half with too
      few included samples ⇒ ``INSUFFICIENT_DATA``;
    - a too-variable (non-steady) effort ⇒ ``INSUFFICIENT_DATA``;
    - a non-finite efficiency / first-half efficiency of 0 ⇒ ``OUT_OF_DOMAIN``.
    """
    if smoothing_window_s <= 0:
        raise ValueError("smoothing_window_s must be positive")

    win = _prepare_window(output_stream, hr_stream, sport, min_duration_s, max_interp_gap_s)
    if isinstance(win, Unavailable):
        return win

    halves = _half_efficiencies(win, smoothing_window_s)
    if isinstance(halves, Unavailable):
        return halves

    decoupling_pct = _decoupling_pct(halves)
    if isinstance(decoupling_pct, Unavailable):
        return decoupling_pct

    return _build_decoupling_result(
        decoupling_pct,
        win,
        halves,
        sport,
        min_duration_s,
        max_interp_gap_s,
        smoothing_window_s,
    )


__all__ = [
    "APPLICABLE_SPORTS",
    "MIN_INCLUDED_SAMPLES_PER_HALF",
    "STEADINESS_MAX_OUTPUT_CV",
    "aerobic_decoupling",
]
