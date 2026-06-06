"""Canonical stream representation + the 1 Hz resampler (ANL-R7, ANL-R8).

A canonical stream is an ordered sequence ``(t_seconds, value|null)`` with
``t_seconds`` monotonically non-decreasing seconds from start; gaps are explicit
``null`` (here represented as ``NaN`` in a float array), NEVER 0 and never silently
shortened (ANL-R7). Metrics that need uniform 1 Hz input resample first with the
declared, tested resampler (ANL-R8): forward-bounded linear interpolation across
gaps no longer than ``max_interp_gap_s`` (default 3 s); longer gaps stay ``null``.

These helpers are pure and deterministic (ANL-R2/R30) and are shared by every
stream metric (NP, MMP, W'bal, decoupling) so the gap semantics are identical.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

DEFAULT_MAX_INTERP_GAP_S = 3.0

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class Stream:
    """An ordered, possibly non-uniform canonical stream. ``NaN`` values are gaps."""

    t_seconds: FloatArray
    values: FloatArray

    def __post_init__(self) -> None:
        if self.t_seconds.shape != self.values.shape:
            raise ValueError("t_seconds and values must have the same shape")
        if self.t_seconds.ndim != 1:
            raise ValueError("stream arrays must be 1-D")
        if self.t_seconds.size and np.any(np.diff(self.t_seconds) < 0):
            raise ValueError("t_seconds must be monotonically non-decreasing (ANL-R7)")

    @classmethod
    def from_values(
        cls, values: list[float | None], *, t0: float = 0.0, dt: float = 1.0
    ) -> Stream:
        """Build a uniform stream from a list where ``None`` marks a gap."""
        v = np.array([np.nan if x is None else float(x) for x in values], dtype=np.float64)
        t = t0 + dt * np.arange(v.size, dtype=np.float64)
        return cls(t_seconds=t, values=v)


def resample_to_1hz(
    stream: Stream, *, max_interp_gap_s: float = DEFAULT_MAX_INTERP_GAP_S
) -> FloatArray:
    """Resample a stream onto a uniform 1 Hz grid ``[0, 1, ..., floor(t_max)]`` (ANL-R8).

    Each grid second is linearly interpolated from the nearest bracketing *valid*
    (non-``NaN``) samples, but only when the bracketing valid samples are no more
    than ``max_interp_gap_s`` apart; otherwise the grid second is ``NaN`` (a gap).
    A grid second outside the span of any valid sample is ``NaN``. Pure and
    deterministic.
    """
    t = stream.t_seconds
    v = stream.values
    valid = ~np.isnan(v)
    if not np.any(valid):
        # No valid samples at all: an all-gap stream.
        n = 0 if t.size == 0 else int(np.floor(t[-1])) + 1
        return np.full(n, np.nan, dtype=np.float64)

    tv = t[valid]
    vv = v[valid]
    grid = np.arange(0.0, np.floor(t[-1]) + 1.0, 1.0, dtype=np.float64)
    out = np.full(grid.size, np.nan, dtype=np.float64)

    # For each grid point, find the bracketing valid samples via searchsorted.
    idx = np.searchsorted(tv, grid, side="left")
    for i, g in enumerate(grid):
        j = idx[i]
        if j < tv.size and tv[j] == g:
            out[i] = vv[j]  # exact sample hit
            continue
        lo = j - 1
        hi = j
        if lo < 0 or hi >= tv.size:
            continue  # grid point lies outside the valid span -> gap
        gap = tv[hi] - tv[lo]
        if gap > max_interp_gap_s:
            continue  # bracketing valid samples too far apart -> gap
        frac = (g - tv[lo]) / gap if gap > 0 else 0.0
        out[i] = vv[lo] + frac * (vv[hi] - vv[lo])
    return out


def trailing_rolling_mean(values_1hz: FloatArray, window_s: int) -> FloatArray:
    """Seeded trailing ``window_s``-second rolling arithmetic mean (NP-R1/R2/R3).

    Output at index ``t`` is the mean of ``values_1hz[t-window_s+1 .. t]`` only when
    all ``window_s`` of those seconds are valid (non-``NaN``); otherwise ``NaN``.
    The window must be fully seeded — a partial early window or a window straddling
    a gap is ``NaN`` (not yet valid), never a short-window mean (NP-R2).
    """
    n = values_1hz.size
    out = np.full(n, np.nan, dtype=np.float64)
    if window_s <= 0:
        raise ValueError("window_s must be positive")
    valid = (~np.isnan(values_1hz)).astype(np.int64)
    # Count of valid seconds in each trailing window via cumulative sum.
    csum_valid = np.concatenate(([0], np.cumsum(valid)))
    filled = np.where(valid.astype(bool), values_1hz, 0.0)
    csum_vals = np.concatenate(([0.0], np.cumsum(filled)))
    for t in range(window_s - 1, n):
        lo = t - window_s + 1
        if csum_valid[t + 1] - csum_valid[lo] == window_s:
            out[t] = (csum_vals[t + 1] - csum_vals[lo]) / window_s
    return out


def longest_contiguous_valid(values_1hz: FloatArray) -> int:
    """Return the longest run of contiguous valid (non-``NaN``) seconds."""
    best = 0
    run = 0
    for x in values_1hz:
        if np.isnan(x):
            run = 0
        else:
            run += 1
            best = max(best, run)
    return best


__all__ = [
    "DEFAULT_MAX_INTERP_GAP_S",
    "FloatArray",
    "Stream",
    "longest_contiguous_valid",
    "resample_to_1hz",
    "trailing_rolling_mean",
]
